"""Krea2 shared-threshold K/V dropping on unmodified FlashAttention 2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from musubi_tuner.modules.attention import flash_attn_varlen_func


@dataclass(frozen=True)
class SharedKVPlan:
    """Fixed-storage varlen layout shared by every main DiT block."""

    order: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    valid_q: Tensor
    max_seqlen: int


def _segment_boundaries(lengths: Tensor, capacity: int) -> Tensor:
    """Describe each row as a real segment followed by an isolated dummy segment."""
    batch_size = lengths.shape[0]
    boundaries = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=lengths.device)
    starts = torch.arange(batch_size, dtype=torch.int32, device=lengths.device) * capacity
    boundaries[1::2] = starts + lengths.to(torch.int32)
    boundaries[2::2] = starts + capacity
    return boundaries


def make_shared_kv_plan(valid_q: Tensor, keep_kv: Tensor) -> SharedKVPlan:
    """Stably place enabled K/V first while retaining fixed BxN storage."""
    if valid_q.ndim != 2 or keep_kv.shape != valid_q.shape:
        raise ValueError(f"valid_q and keep_kv must have the same BxN shape, got {valid_q.shape} and {keep_kv.shape}")
    if valid_q.dtype != torch.bool or keep_kv.dtype != torch.bool:
        raise ValueError("valid_q and keep_kv must be boolean")
    if torch.any(keep_kv & ~valid_q):
        raise ValueError("shared K/V plan cannot retain padding keys")
    if torch.any(valid_q.sum(dim=1) == 0):
        raise ValueError("each shared K/V row must contain at least one valid query")
    if torch.any(keep_kv.sum(dim=1) == 0):
        raise ValueError("each shared K/V row must contain at least one retained key/value")

    batch_size, capacity = valid_q.shape
    # Stable sorting preserves the original image raster order and text order.
    local_order = torch.argsort((~keep_kv).to(torch.int8), dim=1, stable=True)
    offsets = torch.arange(batch_size, device=valid_q.device)[:, None] * capacity
    order = (local_order + offsets).flatten()
    return SharedKVPlan(
        order=order,
        cu_seqlens_q=_segment_boundaries(valid_q.sum(dim=1), capacity),
        cu_seqlens_k=_segment_boundaries(keep_kv.sum(dim=1), capacity),
        valid_q=valid_q,
        max_seqlen=capacity,
    )


def shared_kv_flash_attention(q: Tensor, k: Tensor, v: Tensor, plan: SharedKVPlan) -> Tensor:
    """Attend every valid Q to only retained K/V and return [B,N,H,D]."""
    if flash_attn_varlen_func is None:
        raise ImportError("Krea2 sharedkv requires FlashAttention 2 (--flash_attn)")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape BxNxHxD")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError(f"shared K/V requires equal BxN storage, got {q.shape}, {k.shape}, {v.shape}")
    if plan.valid_q.shape != q.shape[:2] or plan.max_seqlen != q.shape[1]:
        raise ValueError(f"shared K/V plan shape {plan.valid_q.shape} does not match Q shape {q.shape[:2]}")

    batch_size, capacity, query_heads, head_dim = q.shape
    key_heads = k.shape[2]
    flat_k = k.reshape(batch_size * capacity, key_heads, head_dim).index_select(0, plan.order)
    flat_v = v.reshape(batch_size * capacity, key_heads, head_dim).index_select(0, plan.order)
    output = flash_attn_varlen_func(
        q.reshape(batch_size * capacity, query_heads, head_dim),
        flat_k,
        flat_v,
        plan.cu_seqlens_q,
        plan.cu_seqlens_k,
        capacity,
        capacity,
        dropout_p=0.0,
        causal=False,
    )
    return output.reshape(batch_size, capacity, query_heads, head_dim).masked_fill(
        ~plan.valid_q[:, :, None, None], 0
    )
