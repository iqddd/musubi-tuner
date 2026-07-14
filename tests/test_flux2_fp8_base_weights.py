import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer


class TestFlux2Fp8BaseWeights(unittest.TestCase):
    def test_fp8_merges_base_weights_during_model_load_only(self):
        trainer = Flux2NetworkTrainer()
        trainer.model_version_info = object()
        trainer.convert_weight_keys = lambda weights, module: {f"converted_{module}": weights["name"]}
        args = SimpleNamespace(
            fp8_scaled=True,
            base_weights=["first.safetensors", "second.safetensors"],
            base_weights_multiplier=[0.25, 0.75],
            network_module="networks.boft_flux_2",
            disable_numpy_memmap=False,
        )
        accelerator = SimpleNamespace(device="cpu")

        with patch("musubi_tuner.flux_2_train_network.load_file", side_effect=[{"name": "first"}, {"name": "second"}]), patch(
            "musubi_tuner.flux_2_train_network.flux2_utils.load_flow_model", return_value=object()
        ) as load_flow_model:
            trainer.load_transformer(accelerator, args, "model.safetensors", "torch", False, "cpu", None)

        kwargs = load_flow_model.call_args.kwargs
        self.assertEqual(
            kwargs["lora_weights_list"],
            [{"converted_networks.boft_flux_2": "first"}, {"converted_networks.boft_flux_2": "second"}],
        )
        self.assertEqual(kwargs["lora_multipliers"], [0.25, 0.75])
        self.assertIsNone(args.base_weights)
        self.assertIsNone(args.base_weights_multiplier)


if __name__ == "__main__":
    unittest.main()
