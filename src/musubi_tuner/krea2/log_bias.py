"""Fixed per-key log-alpha bias for Krea2's main FlashAttention blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

import torch
from torch import Tensor

_alpha_attention: Callable | None = None
_FA2_ALPHA_IMPORT_ERROR: Exception | None = None
_FA2_ALPHA_IMPORT_ATTEMPTED = False


@dataclass(frozen=True)
class LogBiasPlan:
    """Fixed-storage log-key weights shared by every main DiT block."""

    log_key_weights: Tensor
    cu_seqlens: Tensor
    valid_tokens: Tensor
    max_seqlen: int


def require_fa2_alpha() -> Callable:
    """Return the optional CUDA operator or fail with an actionable message."""
    global _alpha_attention, _FA2_ALPHA_IMPORT_ERROR, _FA2_ALPHA_IMPORT_ATTEMPTED
    if not _FA2_ALPHA_IMPORT_ATTEMPTED:
        _FA2_ALPHA_IMPORT_ATTEMPTED = True
        try:
            _alpha_attention = import_module("fa2_alpha").alpha_attention
        except (ImportError, OSError, RuntimeError, AttributeError) as error:
            # Importing a compiled PyTorch extension can fail with ImportError,
            # OSError, or an ABI-related RuntimeError. This path is reached only
            # when logbias was explicitly requested, so native/sharedkv remain
            # independent of the optional wheel.
            _FA2_ALPHA_IMPORT_ERROR = error
    if _alpha_attention is None:
        detail = f" ({_FA2_ALPHA_IMPORT_ERROR})" if _FA2_ALPHA_IMPORT_ERROR is not None else ""
        raise ImportError(
            "Krea2 logbias requires a fa2-alpha wheel compatible with the installed Python, PyTorch, "
            f"CUDA, and GPU architecture{detail}. Install the compatible 'fa2-alpha' wheel before training."
        ) from _FA2_ALPHA_IMPORT_ERROR
    return _alpha_attention


def _segment_boundaries(lengths: Tensor, capacity: int) -> Tensor:
    """Describe each row as a valid prefix followed by an isolated padding segment."""
    batch_size = lengths.shape[0]
    boundaries = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=lengths.device)
    starts = torch.arange(batch_size, dtype=torch.int32, device=lengths.device) * capacity
    boundaries[1::2] = starts + lengths.to(torch.int32)
    boundaries[2::2] = starts + capacity
    return boundaries


def make_log_bias_plan(valid_tokens: Tensor, probabilities: Tensor, gamma: float) -> LogBiasPlan:
    """Build ``gamma * log(p)`` for a stably packed image/text sequence."""
    if valid_tokens.ndim != 2 or probabilities.shape != valid_tokens.shape:
        raise ValueError(
            "valid_tokens and probabilities must have the same BxN shape, "
            f"got {valid_tokens.shape} and {probabilities.shape}"
        )
    if valid_tokens.dtype != torch.bool:
        raise ValueError(f"valid_tokens must be boolean, got {valid_tokens.dtype}")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"Krea2 logbias gamma must be finite and positive, got {gamma}")

    probabilities = probabilities.to(device=valid_tokens.device, dtype=torch.float32)
    if not torch.isfinite(probabilities).all() or torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Krea2 logbias probabilities must be finite and in [0, 1]")
    if torch.any(valid_tokens.sum(dim=1) == 0):
        raise ValueError("each Krea2 logbias row must contain at least one valid token")
    if torch.any(probabilities[valid_tokens] <= 0):
        raise ValueError("Krea2 logbias requires strictly positive alpha for every retained token")

    # Padding is isolated by cu_seqlens, and a neutral finite value keeps the
    # fixed BxN storage safe for the custom CUDA operator.
    safe_probabilities = probabilities.masked_fill(~valid_tokens, 1.0)
    log_key_weights = (safe_probabilities.log() * gamma).contiguous()
    if not torch.isfinite(log_key_weights).all():
        raise ValueError("Krea2 logbias produced non-finite key weights; reduce gamma or check alpha values")
    capacity = valid_tokens.shape[1]
    return LogBiasPlan(
        log_key_weights=log_key_weights,
        cu_seqlens=_segment_boundaries(valid_tokens.sum(dim=1), capacity),
        valid_tokens=valid_tokens,
        max_seqlen=capacity,
    )


def log_bias_flash_attention(q: Tensor, k: Tensor, v: Tensor, plan: LogBiasPlan) -> Tensor:
    """Run the optional BF16/D128 log-bias FA2 operator and return BxNxHxD."""
    alpha_attention = require_fa2_alpha()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape BxNxHxD")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError(f"Krea2 logbias requires equal BxN storage, got {q.shape}, {k.shape}, {v.shape}")
    if plan.valid_tokens.shape != q.shape[:2] or plan.max_seqlen != q.shape[1]:
        raise ValueError(f"Krea2 logbias plan shape {plan.valid_tokens.shape} does not match Q shape {q.shape[:2]}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("Krea2 logbias fa2-alpha supports CUDA tensors only")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("Krea2 logbias fa2-alpha supports BF16 Q/K/V only")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("Krea2 logbias fa2-alpha supports head_dim=128 only")
    if q.shape[2] % k.shape[2] != 0 or k.shape[2] != v.shape[2]:
        raise ValueError(f"Krea2 logbias requires compatible GQA heads, got {q.shape[2]}, {k.shape[2]}, {v.shape[2]}")

    output = alpha_attention(q, k, v, plan.log_key_weights, None, plan.cu_seqlens)
    if output.shape != q.shape:
        raise RuntimeError(f"fa2-alpha returned shape {output.shape}, expected {q.shape}")
    # Dummy padding is a separate varlen segment and cannot interact with the
    # valid prefix. Its output is discarded by the model's final unpack/scatter,
    # so avoid an extra full-size mask kernel in every DiT block.
    return output
