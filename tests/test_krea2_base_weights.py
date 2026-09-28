import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


class TestKrea2BaseWeightsLoading(unittest.TestCase):
    def test_raw_load_merges_base_weights_once_and_keeps_them_for_turbo(self):
        trainer = Krea2NetworkTrainer()
        trainer.convert_weight_keys = lambda weights, module: {f"converted_{module}": weights["name"]}
        args = SimpleNamespace(
            fp8_scaled=True,
            base_weights=["first.safetensors", "second.safetensors"],
            base_weights_multiplier=[0.25],
            network_module="networks.lora_krea2",
        )
        accelerator = SimpleNamespace(device="cpu")

        with (
            patch("musubi_tuner.krea2_train_network.load_file", side_effect=[{"name": "first"}, {"name": "second"}]),
            patch("musubi_tuner.krea2_train_network.krea2_utils.load_krea2_dit", return_value=object()) as load_dit,
        ):
            trainer.load_transformer(accelerator, args, "raw.safetensors", "torch", False, "cpu", None)

        expected_weights = [
            {"converted_networks.lora_krea2": "first"},
            {"converted_networks.lora_krea2": "second"},
        ]
        self.assertEqual(load_dit.call_args.kwargs["lora_weights"], expected_weights)
        self.assertEqual(load_dit.call_args.kwargs["lora_multipliers"], [0.25, 1.0])
        self.assertEqual(trainer._base_weights_for_merge, expected_weights)
        self.assertEqual(trainer._base_weights_multipliers, [0.25, 1.0])
        self.assertIsNone(args.base_weights)
        self.assertIsNone(args.base_weights_multiplier)

    def test_state_dict_merges_lora_before_optional_fp8_quantization(self):
        base = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.bfloat16)
        down = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        up = torch.tensor([[2.0], [-1.0]], dtype=torch.bfloat16)
        lora = {
            "lora_unet_blocks_0_proj.lora_down.weight": down,
            "lora_unet_blocks_0_proj.lora_up.weight": up,
            "lora_unet_blocks_0_proj.alpha": torch.tensor(1.0),
        }
        multiplier = 0.5
        expected = base.float() + multiplier * (up.float() @ down.float())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.safetensors")
            save_file({"blocks.0.proj.weight": base}, path)

            bf16_state_dict = krea2_utils.load_krea2_dit_state_dict(
                path,
                fp8_scaled=False,
                calc_device="cpu",
                result_device="cpu",
                lora_weights=[lora],
                lora_multipliers=[multiplier],
            )
            state_dict = krea2_utils.load_krea2_dit_state_dict(
                path,
                fp8_scaled=True,
                calc_device="cpu",
                result_device="cpu",
                lora_weights=[lora],
                lora_multipliers=[multiplier],
            )

        torch.testing.assert_close(bf16_state_dict["blocks.0.proj.weight"].float(), expected, rtol=0, atol=0)
        dequantized = state_dict["blocks.0.proj.weight"].float() * state_dict["blocks.0.proj.scale_weight"].float()
        torch.testing.assert_close(dequantized, expected, rtol=0.02, atol=0.02)


class TestKrea2BaseWeightsSwapping(unittest.TestCase):
    @staticmethod
    def _args(cache: bool):
        return SimpleNamespace(
            dit="raw.safetensors",
            turbo_dit="turbo.safetensors",
            turbo_dit_cache=cache,
            fp8_scaled=True,
        )

    @staticmethod
    def _state_dict(model: nn.Module, value: float) -> dict[str, torch.Tensor]:
        return {key: torch.full_like(tensor, value) for key, tensor in model.state_dict().items()}

    def test_cached_turbo_is_prepared_once_and_raw_restore_does_not_reload(self):
        trainer = Krea2NetworkTrainer()
        trainer._base_weights_for_merge = [{"adapter": torch.tensor(1.0)}]
        trainer._base_weights_multipliers = [0.8]
        base_weights = trainer._base_weights_for_merge
        model = nn.Linear(2, 2)
        accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda value: value)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        turbo = self._state_dict(model, 4.0)

        with patch("musubi_tuner.krea2_train_network.krea2_utils.load_krea2_dit_state_dict", return_value=turbo) as load_sd:
            for _ in range(2):
                trainer.on_before_sample_images(accelerator, self._args(True), 0, 0, None, model, None, None, None)
                self.assertTrue(torch.equal(model.weight, turbo["weight"]))
                trainer.on_after_sample_images(accelerator, self._args(True), 0, 0, None, model, None, None, None)
                self.assertTrue(torch.equal(model.weight, initial["weight"]))

        load_sd.assert_called_once()
        self.assertIs(load_sd.call_args.kwargs["lora_weights"], base_weights)
        self.assertEqual(load_sd.call_args.kwargs["lora_multipliers"], [0.8])
        self.assertIsNone(trainer._base_weights_for_merge)
        self.assertIsNone(trainer._base_weights_multipliers)

    def test_uncached_turbo_and_raw_reload_both_merge_base_weights(self):
        trainer = Krea2NetworkTrainer()
        trainer._base_weights_for_merge = [{"adapter": torch.tensor(1.0)}]
        trainer._base_weights_multipliers = [0.8]
        model = nn.Linear(2, 2)
        accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda value: value)
        turbo = self._state_dict(model, 4.0)
        raw = self._state_dict(model, 2.0)
        args = self._args(False)

        with patch("musubi_tuner.krea2_train_network.krea2_utils.load_krea2_dit_state_dict", side_effect=[turbo, raw]) as load_sd:
            trainer.on_before_sample_images(accelerator, args, 0, 0, None, model, None, None, None)
            self.assertTrue(torch.equal(model.weight, turbo["weight"]))
            trainer.on_after_sample_images(accelerator, args, 0, 0, None, model, None, None, None)
            self.assertTrue(torch.equal(model.weight, raw["weight"]))

        self.assertEqual([call.args[0] for call in load_sd.call_args_list], ["turbo.safetensors", "raw.safetensors"])
        for call in load_sd.call_args_list:
            self.assertIs(call.kwargs["lora_weights"], trainer._base_weights_for_merge)
            self.assertEqual(call.kwargs["lora_multipliers"], [0.8])


if __name__ == "__main__":
    unittest.main()
