"""Reusable, non-random main-attention layout, prepared outside compiled blocks."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from musubi_tuner.krea2.log_bias import LogBiasPlan, make_log_bias_plan
from musubi_tuner.krea2.shared_kv import SharedKVPlan, make_shared_kv_plan
from musubi_tuner.modules.attention import AttentionParams


@dataclass(frozen=True)
class PackedAlphaPlan:
    image_mask: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    permutation: torch.Tensor
    original_length: int
    attention: AttentionParams
    shared_kv: SharedKVPlan | None
    log_bias: LogBiasPlan | None

    def pack(self, tensor):
        length = min(self.original_length, self.valid.shape[1])
        indices = self.permutation[:, :length]
        packed = tensor.gather(1, indices[..., None].expand(-1, -1, tensor.shape[-1]))
        return F.pad(packed, (0, 0, 0, self.valid.shape[1] - length))


def prepare_packed_alpha_plan(image_mask, text_mask, positions, probabilities, *,
                              attn_mode, split_attn=False, mode="native", gamma=None,
                              image_keep_kv=None):
    # Import lazily: SingleStreamDiT also consumes this plan type.
    from musubi_tuner.krea2.krea2_mmdit import pack_valid_prefix

    valid = torch.cat((image_mask, text_mask), dim=1)
    _, packed_pos, packed_valid, permutation, length = pack_valid_prefix(positions, positions, valid)
    attention = AttentionParams.create_attention_params_from_mask(attn_mode, split_attn, 0, packed_valid)
    plan = PackedAlphaPlan(image_mask, packed_pos, packed_valid, permutation, length, attention, None, None)
    shared_kv = log_bias = None
    if mode == "sharedkv":
        if image_keep_kv is None or image_keep_kv.shape != image_mask.shape:
            raise ValueError("sharedkv requires explicit per-image K/V decisions matching image_mask")
        keep = torch.cat((image_keep_kv & image_mask, text_mask), dim=1)
        packed_keep = plan.pack(keep[..., None]).squeeze(-1) & packed_valid
        shared_kv = make_shared_kv_plan(packed_valid, packed_keep)
    elif mode == "logbias":
        p = torch.cat((probabilities, torch.ones_like(text_mask, dtype=torch.float32)), dim=1)
        packed_p = plan.pack(p[..., None]).squeeze(-1)
        log_bias = make_log_bias_plan(packed_valid, packed_p, gamma)
    elif mode != "native":
        raise ValueError(f"Unknown Krea2 alpha attention mode: {mode}")
    return PackedAlphaPlan(image_mask, packed_pos, packed_valid, permutation, length, attention, shared_kv, log_bias)
