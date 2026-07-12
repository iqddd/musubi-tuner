import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader


def _args(**overrides):
    values = dict(
        fp8_base=False,
        fp8_scaled=False,
        turbo_dit="turbo.safetensors",
        turbo_dit_cache=True,
        blocks_to_swap=2,
        block_swap_h2d_only=True,
        sample_prompts="prompts.txt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestKrea2TurboBlockSwapValidation(unittest.TestCase):
    def test_cached_h2d_only_combination_is_allowed(self):
        Krea2NetworkTrainer().handle_model_specific_args(_args())

    def test_disk_streaming_turbo_is_rejected_with_block_swap(self):
        with self.assertRaisesRegex(ValueError, "requires both --turbo_dit_cache and --block_swap_h2d_only"):
            Krea2NetworkTrainer().handle_model_specific_args(_args(turbo_dit_cache=False))

    def test_classic_block_swap_is_rejected_with_turbo(self):
        with self.assertRaisesRegex(ValueError, "requires both --turbo_dit_cache and --block_swap_h2d_only"):
            Krea2NetworkTrainer().handle_model_specific_args(_args(block_swap_h2d_only=False))


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 8, bias=False)
        self.register_buffer("scale_weight", torch.ones(1))
        self.requires_grad_(False)


class _TinyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
        self.head = nn.Linear(8, 8, bias=False)
        self.register_buffer("global_scale", torch.ones(1))
        self.requires_grad_(False)
        self.to(device)
        self.offloader = LoRAStreamOffloader(
            "test",
            self.blocks,
            num_blocks=len(self.blocks),
            blocks_to_swap=2,
            supports_backward=False,
            device=device,
            ring_size=1,
            use_pinned_memory=True,
        )
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def prepare_block_swap_before_forward(self):
        self.offloader.prepare_block_devices_before_forward(self.blocks)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the H2D ring test")
class TestH2DWeightBanks(unittest.TestCase):
    def test_switches_cpu_master_bank_and_invalidates_ring(self):
        device = torch.device("cuda")
        blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
        offloader = LoRAStreamOffloader(
            "test",
            blocks,
            num_blocks=len(blocks),
            blocks_to_swap=2,
            supports_backward=False,
            device=device,
            ring_size=1,
            use_pinned_memory=True,
        )
        offloader.prepare_block_devices_before_forward(blocks)

        raw = {
            key: module.weight.detach().cpu().clone()
            for block_idx in offloader.stream_idx
            for key, module in (
                (f"blocks.{block_idx}.fc1.weight", blocks[block_idx].fc1),
                (f"blocks.{block_idx}.fc2.weight", blocks[block_idx].fc2),
            )
        }
        turbo = {key: torch.full_like(value, 0.125 + index) for index, (key, value) in enumerate(raw.items())}

        used = offloader.register_weight_bank_from_state_dict("turbo", turbo)
        self.assertEqual(used, set(turbo))
        offloader.activate_weight_bank("turbo")
        self.assertEqual(offloader.active_weight_bank, "turbo")
        self.assertEqual(offloader.in_slot, [None])

        offloader.prepare_block_devices_before_forward(blocks)
        first = offloader.stream_idx[0]
        offloader.wait_for_block(first)
        self.assertTrue(torch.equal(blocks[first].fc1.weight.detach().cpu(), turbo[f"blocks.{first}.fc1.weight"]))
        self.assertTrue(torch.equal(blocks[first].fc2.weight.detach().cpu(), turbo[f"blocks.{first}.fc2.weight"]))

        offloader.activate_weight_bank("raw")
        self.assertEqual(offloader.active_weight_bank, "raw")
        self.assertEqual(offloader.in_slot, [None])
        offloader.prepare_block_devices_before_forward(blocks)
        offloader.wait_for_block(first)
        self.assertTrue(torch.equal(blocks[first].fc1.weight.detach().cpu(), raw[f"blocks.{first}.fc1.weight"]))
        self.assertTrue(torch.equal(blocks[first].fc2.weight.detach().cpu(), raw[f"blocks.{first}.fc2.weight"]))

    def test_trainer_switches_streamed_and_resident_tensors_together(self):
        device = torch.device("cuda")
        model = _TinyModel(device)
        # Krea compiles blocks after the offloader's first prepare. The offloader must keep
        # using the pre-compile checkpoint names while the trainer normalizes _orig_mod names.
        for index, block in enumerate(model.blocks):
            model.blocks[index] = torch.compile(block, backend="eager")
        trainer = Krea2NetworkTrainer()
        accelerator = SimpleNamespace(device=device, unwrap_model=lambda value: value)
        args = _args(fp8_scaled=True)

        raw = {key: value.detach().cpu().clone() for key, value in trainer._named_live_tensors(model).items()}
        trainer._turbo_stash = {
            key: torch.full_like(value, 2.0 if key.endswith("scale_weight") or key == "global_scale" else 0.25)
            for key, value in raw.items()
        }

        trainer.on_before_sample_images(accelerator, args, 0, 0, None, model, None, None, None)
        self.assertEqual(model.offloader.active_weight_bank, "turbo")
        model.prepare_block_swap_before_forward()
        first = model.offloader.stream_idx[0]
        model.offloader.wait_for_block(first)
        self.assertTrue(torch.equal(model.blocks[first].fc1.weight.detach().cpu(), torch.full((8, 8), 0.25)))
        self.assertEqual(model.blocks[first].scale_weight.item(), 2.0)
        self.assertTrue(torch.equal(model.head.weight.detach().cpu(), torch.full((8, 8), 0.25)))
        self.assertEqual(model.global_scale.item(), 2.0)

        trainer.on_after_sample_images(accelerator, args, 0, 0, None, model, None, None, None)
        self.assertEqual(model.offloader.active_weight_bank, "raw")
        restored = trainer._named_live_tensors(model)
        for key, expected in raw.items():
            self.assertTrue(torch.equal(restored[key].detach().cpu(), expected), key)


if __name__ == "__main__":
    unittest.main()
