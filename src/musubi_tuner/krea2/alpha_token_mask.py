"""Krea2 alpha alignment shared by training and token length indexing."""

import torch

from musubi_tuner.utils import train_utils


def align_alpha_mask_to_token_grid(
    alpha_mask: torch.Tensor, latent_size: tuple[int, int], patch: int
) -> torch.Tensor:
    """Exclude entire image tokens touched by exact-zero alpha, without a margin."""
    if alpha_mask.ndim == 4 and alpha_mask.shape[1] == 1:
        return align_alpha_mask_to_token_grid(alpha_mask[:, 0], latent_size, patch).unsqueeze(1)
    if alpha_mask.ndim != 3:
        raise ValueError(f"Krea2 alpha token drop expects alpha_mask BxHxW, got {tuple(alpha_mask.shape)}")
    if patch <= 0 or any(size <= 0 or size % patch != 0 for size in latent_size):
        raise ValueError(f"latent size {latent_size} must be divisible by positive DiT patch size {patch}")
    expected_size = (latent_size[0] * 8, latent_size[1] * 8)
    if alpha_mask.shape[-2:] != expected_size:
        raise ValueError(f"Krea2 alpha mask must have image size {expected_size}, got {tuple(alpha_mask.shape[-2:])}")

    batch_size, height, width = alpha_mask.shape
    cell = patch * 8
    zero = (alpha_mask == 0).reshape(batch_size, height // cell, cell, width // cell, cell)
    touched = zero.any(dim=(2, 4))
    excluded = touched.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
    return alpha_mask.masked_fill(excluded, 0)


def make_alpha_token_keep_mask(
    alpha_mask: torch.Tensor, latent_size: tuple[int, int], patch: int, device: torch.device
) -> torch.Tensor:
    """Return token keep flags from a prealigned pixel alpha mask."""
    if alpha_mask.ndim == 4 and alpha_mask.shape[1] == 1:
        alpha_mask = alpha_mask[:, 0]
    if alpha_mask.ndim != 3:
        raise ValueError(f"Krea2 alpha token drop expects alpha_mask BxHxW, got {tuple(alpha_mask.shape)}")
    if latent_size[0] % patch != 0 or latent_size[1] % patch != 0:
        raise ValueError(f"latent size {latent_size} must be divisible by DiT patch size {patch}")

    alpha_mask = alpha_mask.to(device=device, dtype=torch.float32)
    latent_weights = train_utils.resize_spatial_mask(alpha_mask, latent_size)
    zero = latent_weights == 0
    b, h, w = zero.shape
    blocks = zero.reshape(b, h // patch, patch, w // patch, patch).permute(0, 1, 3, 2, 4)
    any_zero = blocks.any(dim=(-1, -2))
    all_zero = blocks.all(dim=(-1, -2))
    mixed = any_zero & ~all_zero
    if torch.any(mixed):
        count = int(mixed.sum().item())
        raise ValueError(
            f"--alpha_masked_token_drop requires a loss mask aligned to complete "
            f"{patch}x{patch} latent / {patch * 8}x{patch * 8} image token cells; "
            f"found {count} token cells containing both exact-zero and positive weights"
        )
    return ~all_zero.flatten(1)
