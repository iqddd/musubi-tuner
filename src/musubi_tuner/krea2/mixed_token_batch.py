"""Training one Krea2 DiT batch containing different latent geometries."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from musubi_tuner.krea2.alpha_token_mask import align_alpha_mask_to_token_grid, make_alpha_token_keep_mask
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3
from musubi_tuner.utils import train_utils


def process_mixed_token_batch(trainer, args, accelerator, transformer, batch, latents, noise,
                              noise_scheduler, dit_dtype, network_dtype):
    """Prepare spatial tensors separately, then perform one packed DiT forward."""
    patch = transformer.config.patch
    device = accelerator.device
    images, positions, image_masks, targets, aligned_alphas, timesteps = [], [], [], [], [], []
    original_shapes, kept_indices = [], []
    preset_timesteps = batch.get("timesteps")
    for i, (latent, eps) in enumerate(zip(latents, noise)):
        if latent.ndim != 4 or latent.shape[1] != 1:
            raise ValueError(f"Krea2 expects a Cx1xHxW latent, got item {i} with shape {tuple(latent.shape)}")
        latent = latent.unsqueeze(0)
        eps = eps.unsqueeze(0)
        original_h, original_w = latent.shape[-2:]
        if original_h % patch or original_w % patch:
            raise ValueError(f"Krea2 latent {i} shape {(original_h, original_w)} is not divisible by patch {patch}")
        own_timestep = None if preset_timesteps is None else [preset_timesteps[i]]
        noisy, timestep = trainer.get_noisy_model_input_and_timesteps(
            args, eps, latent, own_timestep, noise_scheduler, device, dit_dtype
        )
        h, w = original_h // patch, original_w // patch
        tokens = rearrange(noisy.squeeze(2), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        pos = torch.stack((torch.zeros_like(grid_y), grid_y, grid_x), dim=-1).reshape(1, h * w, 3).float()

        alpha = batch["alpha_mask"][i]
        if alpha is not None and args.alpha_masked_token_drop:
            alpha = align_alpha_mask_to_token_grid(alpha[None].to(device=device, dtype=torch.float32),
                                                   (original_h, original_w), patch)
            keep = make_alpha_token_keep_mask(alpha, (original_h, original_w), patch, device)
        else:
            keep = torch.ones((1, h * w), device=device, dtype=torch.bool)
            if alpha is not None:
                alpha = alpha[None].to(device=device, dtype=torch.float32)
        selected = keep[0].nonzero(as_tuple=True)[0]
        images.append(tokens[0, selected].to(dtype=network_dtype))
        positions.append(pos[0, selected])
        image_masks.append(torch.ones(selected.numel(), device=device, dtype=torch.bool))
        kept_indices.append(selected)
        # Match Krea2.call_dit: latents are promoted to the trainable network
        # dtype before forming the velocity target (cached noise may be bf16).
        targets.append(eps - latent.to(dtype=network_dtype))
        aligned_alphas.append(alpha)
        timesteps.append(timestep.reshape(()))
        original_shapes.append((h, w))

    max_image = max(tokens.shape[0] for tokens in images)
    image = torch.stack([F.pad(tokens, (0, 0, 0, max_image - tokens.shape[0])) for tokens in images])
    image_pos = torch.stack([F.pad(pos, (0, 0, 0, max_image - pos.shape[0])) for pos in positions])
    image_mask = torch.stack([F.pad(mask, (0, max_image - mask.shape[0]), value=False) for mask in image_masks])

    embeds = batch["krea2_vl_embed"]
    max_text = max(embed.shape[0] for embed in embeds)
    context = torch.stack([F.pad(embed, (0, 0, 0, 0, 0, max_text - embed.shape[0])) for embed in embeds])
    context = context.to(device=device, dtype=network_dtype)
    text_mask = torch.arange(max_text, device=device)[None] < torch.tensor(
        [embed.shape[0] for embed in embeds], device=device
    )[:, None]
    text_pos = torch.zeros(len(images), max_text, 3, device=device)
    pos = torch.cat((image_pos, text_pos), dim=1)
    mask = torch.cat((image_mask, text_mask), dim=1)
    if args.gradient_checkpointing:
        image.requires_grad_(True)
        context.requires_grad_(True)

    with accelerator.autocast():
        predictions = transformer(
            img=image, context=context, t=torch.stack(timesteps).to(device=device) / 1000.0,
            pos=pos, mask=mask, image_mask=image_mask
        )

    losses = []
    for i, ((h, w), selected, target, alpha, timestep) in enumerate(
        zip(original_shapes, kept_indices, targets, aligned_alphas, timesteps)
    ):
        selected_prediction = predictions[i:i + 1, :selected.numel()]
        full_prediction = selected_prediction.new_zeros(1, h * w, selected_prediction.shape[-1])
        full_prediction = full_prediction.scatter(
            1, selected[None, :, None].expand_as(selected_prediction), selected_prediction
        )
        pred = rearrange(full_prediction,
                         "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=patch, pw=patch).unsqueeze(2)
        loss = F.mse_loss(pred.to(network_dtype), target, reduction="none")
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, noise_scheduler, timestep.reshape(1), device, dit_dtype
        )
        if weighting is not None:
            loss = loss * weighting
        if alpha is not None:
            loss = train_utils.apply_alpha_masked_loss(loss, {"alpha_mask": alpha})
        losses.append(loss.mean())  # denominator is the full latent, including dropped cells
    return torch.stack(losses).mean(), {}
