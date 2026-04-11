import importlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.networks.boft import (
    BOFTInfModule,
    BOFTModule,
    create_network,
    boft_diff_from_weight,
    butterfly_factor,
    export_oft_blocks_to_skew,
    export_oft_blocks_to_packed,
    merge_weights_to_tensor,
    packed_oft_blocks_to_export,
)


def import_detect_network_type():
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub

    fp8_stub = types.ModuleType("musubi_tuner.modules.fp8_optimization_utils")
    fp8_stub.load_safetensors_with_fp8_optimization = lambda *args, **kwargs: {}
    sys.modules["musubi_tuner.modules.fp8_optimization_utils"] = fp8_stub

    device_stub = types.ModuleType("musubi_tuner.utils.device_utils")
    device_stub.synchronize_device = lambda *args, **kwargs: None
    sys.modules["musubi_tuner.utils.device_utils"] = device_stub

    safetensors_stub = types.ModuleType("musubi_tuner.utils.safetensors_utils")
    safetensors_stub.MemoryEfficientSafeOpen = object
    safetensors_stub.TensorWeightAdapter = object
    safetensors_stub.WeightTransformHooks = object
    safetensors_stub.get_split_weight_filenames = lambda *args, **kwargs: None
    sys.modules["musubi_tuner.utils.safetensors_utils"] = safetensors_stub

    if "musubi_tuner.utils.lora_utils" in sys.modules:
        del sys.modules["musubi_tuner.utils.lora_utils"]
    lora_utils = importlib.import_module("musubi_tuner.utils.lora_utils")
    return lora_utils.detect_network_type


class TestBOFT(unittest.TestCase):
    class DoubleStreamBlock(torch.nn.Module):
        def __init__(self, in_features=16, out_features=3072):
            super().__init__()
            self.proj = torch.nn.Linear(in_features, out_features, bias=False)

        def forward(self, x):
            return self.proj(x)

    class ToyNet(torch.nn.Module):
        def __init__(self, in_features=16, out_features=3072):
            super().__init__()
            self.block = TestBOFT.DoubleStreamBlock(in_features, out_features)

    def test_butterfly_factor_for_klein_shapes(self):
        self.assertEqual(butterfly_factor(3072, 32), (24, 128))
        self.assertEqual(butterfly_factor(4096, 32), (32, 128))

    def test_zero_init_is_noop(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(12, 3072, bias=True)
        x = torch.randn(2, 5, 12)
        expected = linear(x)

        module = BOFTModule("test", linear, multiplier=1.0, lora_dim=32, alpha=0.2, rescaled=True)
        module.apply_to()
        actual = linear(x)

        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_matches_weight_space_boft(self):
        torch.manual_seed(1)
        linear = torch.nn.Linear(10, 3072, bias=True)
        module = BOFTModule("test", linear, multiplier=1.0, lora_dim=32, alpha=0.2, rescaled=True)
        with torch.no_grad():
            module.oft_blocks.normal_(0, 0.01)
            module.rescale.fill_(1.01)

        x = torch.randn(4, 10)
        base_weight = linear.weight.detach().clone()
        diff = boft_diff_from_weight(base_weight, module.export_oft_blocks(), module.alpha, 1.0, module.rescale)
        expected = torch.nn.functional.linear(x, base_weight + diff, linear.bias)

        module.apply_to()
        actual = linear(x)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_training_module_rejects_scaled_multiplier(self):
        linear = torch.nn.Linear(10, 3072, bias=True)
        with self.assertRaises(ValueError):
            BOFTModule("test", linear, multiplier=0.7, lora_dim=32, alpha=0.2, rescaled=True)

    def test_merge_weights_to_tensor_consumes_export_keys(self):
        torch.manual_seed(2)
        linear = torch.nn.Linear(16, 3072, bias=False)
        module = BOFTModule("test", linear, multiplier=1.0, lora_dim=32, alpha=0.1, rescaled=True)
        with torch.no_grad():
            module.oft_blocks.normal_(0, 0.01)
            module.rescale.fill_(1.02)

        lora_name = "lora_unet_test"
        keys = {f"{lora_name}.oft_blocks", f"{lora_name}.rescale", f"{lora_name}.alpha"}
        state_dict = {
            f"{lora_name}.oft_blocks": module.export_oft_blocks().detach().clone(),
            f"{lora_name}.rescale": module.rescale.detach().clone(),
            f"{lora_name}.alpha": module.alpha.detach().clone(),
        }

        weight = linear.weight.detach().clone()
        expected = weight + boft_diff_from_weight(weight, module.export_oft_blocks(), module.alpha, 0.5, module.rescale)
        merged = merge_weights_to_tensor(weight, lora_name, state_dict, keys, 0.5, torch.device("cpu"))

        self.assertTrue(torch.allclose(expected, merged, atol=1e-5, rtol=1e-5))
        self.assertEqual(keys, set())

    def test_inference_module_scaled_merge_matches_weight_space_boft(self):
        torch.manual_seed(6)
        linear = torch.nn.Linear(16, 3072, bias=False)
        module = BOFTModule("test", linear, multiplier=1.0, lora_dim=32, alpha=0.1, rescaled=True)
        with torch.no_grad():
            module.oft_blocks.normal_(0, 0.01)
            module.rescale.fill_(1.02)

        inference_linear = torch.nn.Linear(16, 3072, bias=False)
        inference_linear.load_state_dict(linear.state_dict())
        inference_module = BOFTInfModule("test", inference_linear, multiplier=0.5, lora_dim=32, alpha=0.1, rescaled=True)
        state_dict = {
            "oft_blocks": module.export_oft_blocks().detach().clone(),
            "rescale": module.rescale.detach().clone(),
            "alpha": module.alpha.detach().clone(),
        }

        weight = linear.weight.detach().clone()
        expected = weight + boft_diff_from_weight(weight, state_dict["oft_blocks"], state_dict["alpha"], 0.5, state_dict["rescale"])
        inference_module.merge_to(state_dict, dtype=None, device=torch.device("cpu"))
        merged = inference_linear.weight.detach().clone()

        self.assertTrue(torch.allclose(expected, merged, atol=1e-5, rtol=1e-5))

    def test_detect_network_type_returns_boft(self):
        detect_network_type = import_detect_network_type()
        state_dict = {"lora_unet_test.oft_blocks": torch.zeros(2, 4, 8, 8)}
        self.assertEqual(detect_network_type(state_dict), "boft")

    def test_export_roundtrip_preserves_skew(self):
        torch.manual_seed(3)
        linear = torch.nn.Linear(10, 3072, bias=False)
        module = BOFTModule("test", linear, multiplier=1.0, lora_dim=32, alpha=0.3)
        with torch.no_grad():
            module.oft_blocks.normal_(0, 0.02)

        export_blocks = module.export_oft_blocks()
        repacked = export_oft_blocks_to_packed(export_blocks)
        restored_export = packed_oft_blocks_to_export(repacked, module.block_size)
        self.assertTrue(torch.allclose(export_oft_blocks_to_skew(restored_export), export_oft_blocks_to_skew(export_blocks)))

    def test_packed_representation_matches_full_forward(self):
        torch.manual_seed(4)
        linear_full = torch.nn.Linear(10, 3072, bias=True)
        linear_packed = torch.nn.Linear(10, 3072, bias=True)
        linear_packed.load_state_dict(linear_full.state_dict())

        module_full = BOFTModule("test_full", linear_full, multiplier=1.0, lora_dim=32, alpha=0.2, rescaled=True)
        module_packed = BOFTModule(
            "test_packed",
            linear_packed,
            multiplier=1.0,
            lora_dim=32,
            alpha=0.2,
            rescaled=True,
            train_representation="packed",
        )
        packed_blocks = torch.randn_like(module_packed.oft_blocks) * 0.01
        export_blocks = packed_oft_blocks_to_export(packed_blocks, module_packed.block_size)
        with torch.no_grad():
            module_full.oft_blocks.copy_(export_blocks)
            module_full.rescale.fill_(1.01)
            module_packed.oft_blocks.copy_(packed_blocks)
            module_packed.rescale.copy_(module_full.rescale)

        x = torch.randn(4, 10)
        module_full.apply_to()
        module_packed.apply_to()

        self.assertTrue(torch.allclose(linear_full(x), linear_packed(x), atol=1e-5, rtol=1e-5))

    def test_network_load_weights_converts_export_for_full_and_packed(self):
        torch.manual_seed(5)
        model_full = self.ToyNet()
        model_packed = self.ToyNet()
        model_packed.load_state_dict(model_full.state_dict())

        network_full = create_network(
            ["DoubleStreamBlock"], 1.0, 32, 0.2, None, [], model_full, module_kwargs={"train_representation": "full"}
        )
        network_packed = create_network(
            ["DoubleStreamBlock"], 1.0, 32, 0.2, None, [], model_packed, module_kwargs={"train_representation": "packed"}
        )
        network_full.apply_to(None, model_full, apply_text_encoder=False, apply_unet=True)
        network_packed.apply_to(None, model_packed, apply_text_encoder=False, apply_unet=True)

        export_key = "lora_unet_block_proj.oft_blocks"
        export_blocks = torch.randn(8, 128, 24, 24) * 0.01
        alpha = torch.tensor(0.2)

        with NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save({export_key: export_blocks, "lora_unet_block_proj.alpha": alpha}, tmp.name)
            self.assertFalse(network_full.load_weights(tmp.name).unexpected_keys)
            self.assertFalse(network_packed.load_weights(tmp.name).unexpected_keys)

        full_blocks = network_full.state_dict()[export_key]
        packed_blocks = network_packed.state_dict()[export_key]

        self.assertEqual(full_blocks.ndim, 4)
        self.assertEqual(packed_blocks.ndim, 3)
        self.assertTrue(torch.allclose(full_blocks, export_blocks))
        self.assertTrue(torch.allclose(packed_blocks, export_oft_blocks_to_packed(export_blocks)))


if __name__ == "__main__":
    unittest.main()
