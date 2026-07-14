import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from musubi_tuner.networks.lokr import LoKrModule


class TestLoKrForward(unittest.TestCase):
    def _make_module(self, rank_dropout=None):
        linear = torch.nn.Linear(4, 8, bias=True)
        module = LoKrModule("test", linear, multiplier=0.7, lora_dim=2, factor=2, rank_dropout=rank_dropout)
        with torch.no_grad():
            module.lokr_w1.normal_(0, 0.2)
            if module.use_w2:
                module.lokr_w2.normal_(0, 0.2)
            else:
                module.lokr_w2_a.normal_(0, 0.2)
                module.lokr_w2_b.normal_(0, 0.2)
        return linear, module

    def test_forward_matches_materialized_delta(self):
        torch.manual_seed(1)
        linear, module = self._make_module()
        x = torch.randn(3, 5, 4)
        expected = linear(x) + F.linear(x, module.get_diff_weight()) * module.multiplier

        module.apply_to()

        torch.testing.assert_close(linear(x), expected)

    def test_rank_dropout_matches_output_channel_mask(self):
        torch.manual_seed(2)
        linear, module = self._make_module(rank_dropout=0.5)
        module.train()
        x = torch.randn(2, 4)
        base = linear(x)
        delta = F.linear(x, module.get_diff_weight())

        torch.manual_seed(3)
        mask = (torch.rand(delta.size(-1)) > module.rank_dropout).to(delta.dtype)
        expected = base + delta * mask * module.multiplier / (1.0 - module.rank_dropout)

        module.apply_to()
        torch.manual_seed(3)
        torch.testing.assert_close(linear(x), expected)


if __name__ == "__main__":
    unittest.main()
