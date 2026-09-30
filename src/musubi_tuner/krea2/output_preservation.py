"""Complementary masked output preservation with sequential student backwards."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from musubi_tuner.krea2.alpha_token_mask import align_alpha_mask_to_token_grid
from musubi_tuner.krea2.packed_alpha_plan import prepare_packed_alpha_plan
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3


def get_output_preservation_loss_balance(args):
    balance = float(getattr(args, "alpha_masked_output_preservation_loss_balance", 0.0))
    if not math.isfinite(balance) or not -1.0 <= balance <= 1.0:
        raise ValueError("--alpha_masked_output_preservation_loss_balance must be finite and in [-1, 1]")
    return balance


def validate_output_preservation_args(args):
    enabled = getattr(args, "alpha_masked_output_preservation", False)
    extremes_only = getattr(args, "alpha_masked_output_preservation_extremes_only", False)
    if extremes_only and not enabled:
        raise ValueError(
            "--alpha_masked_output_preservation_extremes_only requires "
            "--alpha_masked_output_preservation"
        )
    if not enabled:
        return
    get_output_preservation_loss_balance(args)
    if not getattr(args, "alpha_masked_token_drop", False):
        raise ValueError("--alpha_masked_output_preservation requires --alpha_masked_token_drop")
    if getattr(args, "blocks_to_swap", 0):
        raise ValueError("--alpha_masked_output_preservation does not support --blocks_to_swap")
    module = getattr(args, "network_module", None)
    if module not in (None, "networks.lora_krea2", "musubi_tuner.networks.lora_krea2"):
        raise ValueError("--alpha_masked_output_preservation supports the standard networks.lora_krea2 only")


def make_output_preservation_probabilities(alpha, latent_size, patch, device, *, extremes_only=False):
    """Return target and preservation weights for each DiT image token."""
    h, w = latent_size
    if alpha is None:
        target = torch.ones((h // patch) * (w // patch), device=device)
        return target, torch.zeros_like(target)
    if alpha.ndim == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    if alpha.ndim != 2:
        raise ValueError(f"Krea2 output preservation expects a 2D alpha mask, got {tuple(alpha.shape)}")
    alpha = alpha.to(device=device, dtype=torch.float32)

    def token_probabilities(mask):
        aligned = align_alpha_mask_to_token_grid(mask.unsqueeze(0), latent_size, patch)
        # Alignment already computed the mean; do not introduce a second reduction.
        return aligned[0, ::patch * 8, ::patch * 8].flatten()

    target = token_probabilities(alpha)
    if not extremes_only:
        return target, 1 - target

    endpoints = (alpha == 0) | (alpha == 1)
    preservation_alpha = torch.where(endpoints, 1 - alpha, alpha)
    return target, token_probabilities(preservation_alpha)


@contextmanager
def base_model_teacher(network):
    """Disable training LoRA, not the inference-only `enabled` attribute.

    Restore every module's exact state before any checkpointed student forward.
    Keeping base modules in training mode avoids unrelated global mode changes.
    """
    from musubi_tuner.networks.lora import LoRANetwork

    if not isinstance(network, LoRANetwork):
        raise ValueError("Output preservation requires the standard Krea2 LoRANetwork")
    modules = network.text_encoder_loras + network.unet_loras
    states = [(module, module.multiplier, module.training) for module in modules]
    try:
        for module, _, _ in states:
            module.multiplier = 0.0
            module.training = False  # LoRA dropout must not consume teacher RNG.
        with torch.no_grad():
            yield
    finally:
        for module, multiplier, training in states:
            module.multiplier = multiplier
            module.training = training


@dataclass(frozen=True)
class PreservationBranch:
    kwargs: dict
    indices: tuple
    weights: tuple
    full_lengths: tuple


def prepare_branch(model, images, positions, probabilities, context, text_mask, timesteps,
                   mode, gamma, keep_kv=None):
    """Select image Q once; preserve original coordinates and full-loss denominators."""
    selected = tuple((p > 0).nonzero(as_tuple=True)[0] for p in probabilities)
    if not any(index.numel() for index in selected):
        return None
    capacity = max(index.numel() for index in selected)
    image = torch.stack([F.pad(x[index], (0, 0, 0, capacity - index.numel()))
                         for x, index in zip(images, selected)])
    image_pos = torch.stack([F.pad(pos[index], (0, 0, 0, capacity - index.numel()))
                             for pos, index in zip(positions, selected)])
    p = torch.stack([F.pad(prob[index], (0, capacity - index.numel()))
                     for prob, index in zip(probabilities, selected)])
    image_mask = p > 0
    kv = None if keep_kv is None else torch.stack([
        F.pad(keep[index], (0, capacity - index.numel()), value=False)
        for keep, index in zip(keep_kv, selected)
    ])
    pos = torch.cat((image_pos, image_pos.new_zeros(context.shape[0], context.shape[1], 3)), dim=1)
    plan = prepare_packed_alpha_plan(
        image_mask, text_mask, pos, p, attn_mode=model.attn_mode, split_attn=model.split_attn,
        mode=mode, gamma=gamma, image_keep_kv=kv,
    )
    if model.gradient_checkpointing:
        image.requires_grad_(True)
        context.requires_grad_(True)
    kwargs = dict(img=image, context=context, t=timesteps / 1000.0, pos=pos,
                  mask=torch.cat((image_mask, text_mask), dim=1), packed_alpha_plan=plan)
    return PreservationBranch(kwargs, selected, tuple(prob[index] for prob, index in zip(probabilities, selected)),
                              tuple(x.shape[0] for x in images))


def branch_loss(prediction, targets, branch, timestep_weights, network_dtype, *, targets_selected=False):
    """Sum weighted errors over retained Q, but divide by each *full* image size."""
    losses = []
    for i, (index, p, full_length) in enumerate(zip(branch.indices, branch.weights, branch.full_lengths)):
        pred = prediction[i, :index.numel()].to(network_dtype)
        target = targets[i, :index.numel()] if targets_selected else targets[i][index]
        error = F.mse_loss(pred, target, reduction="none")
        losses.append((error * p[:, None] * timestep_weights[i]).sum() / (full_length * prediction.shape[-1]))
    return torch.stack(losses).mean()


def process_preservation_batch(trainer, args, accelerator, transformer, network, batch, latents, noise,
                               noise_scheduler, dit_dtype, network_dtype, *, backward=None):
    """Reuse one teacher target, optionally backpropagating each student immediately.

    Training supplies ``backward`` to keep only one student graph alive at a time.
    Without it, return a differentiable combined loss for numerical diagnostics.
    """
    device, patch = accelerator.device, transformer.config.patch
    loss_balance = get_output_preservation_loss_balance(args)
    target_loss_weight = 1.0 + loss_balance
    preservation_loss_weight = 1.0 - loss_balance
    mixed = isinstance(latents, list)
    preset = batch.get("timesteps")
    if mixed:
        noisy_and_t = [trainer.get_noisy_model_input_and_timesteps(
            args, eps.unsqueeze(0), latent.unsqueeze(0), None if preset is None else [preset[i]],
            noise_scheduler, device, dit_dtype,
        ) for i, (latent, eps) in enumerate(zip(latents, noise))]
        noisy_images = [pair[0][0] for pair in noisy_and_t]
        timesteps = torch.cat([pair[1].reshape(1) for pair in noisy_and_t])
    else:
        noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
            args, noise, latents, preset, noise_scheduler, device, dit_dtype,
        )
        noisy_images = list(noisy)

    images, positions, probabilities, preservation_probabilities, targets = [], [], [], [], []
    alpha_batch = batch.get("alpha_mask")
    extremes_only = getattr(args, "alpha_masked_output_preservation_extremes_only", False)
    for i, (noisy, latent, eps) in enumerate(zip(noisy_images, latents, noise)):
        if noisy.ndim != 4 or noisy.shape[1] != 1:
            raise ValueError("Krea2 output preservation expects single-frame Cx1xHxW latents")
        h, w = noisy.shape[-2:]
        if h % patch or w % patch:
            raise ValueError("Krea2 latent dimensions must be divisible by the DiT patch size")
        image = rearrange(noisy[:, 0], "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch)
        images.append(image.to(device=device, dtype=network_dtype))
        y, x = torch.meshgrid(torch.arange(h // patch, device=device),
                              torch.arange(w // patch, device=device), indexing="ij")
        positions.append(torch.stack((torch.zeros_like(y), y, x), dim=-1).reshape(-1, 3).float())
        alpha = None if alpha_batch is None else alpha_batch[i]
        p, preservation_p = make_output_preservation_probabilities(
            alpha, (h, w), patch, device, extremes_only=extremes_only
        )
        probabilities.append(p)
        preservation_probabilities.append(preservation_p)
        target = eps.to(device) - latent.to(device=device, dtype=network_dtype)
        targets.append(rearrange(target[:, 0], "c (h ph) (w pw) -> (h w) (c ph pw)", ph=patch, pw=patch))

    embeds = batch["krea2_vl_embed"]
    max_text = max(x.shape[0] for x in embeds)
    context = torch.stack([F.pad(x, (0, 0, 0, 0, 0, max_text - x.shape[0])) for x in embeds]).to(
        device=device, dtype=network_dtype)
    text_mask = torch.arange(max_text, device=device)[None] < torch.tensor([x.shape[0] for x in embeds], device=device)[:, None]
    timesteps = timesteps.to(device)
    mode = getattr(args, "alpha_masked_attention_mode", "native")
    gamma = getattr(args, "alpha_masked_attention_gamma", None)
    target_kv = preservation_kv = None
    if mode == "sharedkv":
        uniforms = torch.rand(len(images), device=device)
        target_kv = [u < p for u, p in zip(uniforms, probabilities)]
        if extremes_only:
            # Apply the transformed alpha mask through the same thresholding mechanism as the target mask.
            preservation_kv = [u < p for u, p in zip(uniforms, preservation_probabilities)]
        else:
            # Preserve the existing strictly complementary decisions in the default full-inversion mode.
            preservation_kv = [u >= p for u, p in zip(uniforms, probabilities)]
    primary = prepare_branch(transformer, images, positions, probabilities, context, text_mask,
                             timesteps, mode, gamma, target_kv)
    preservation = prepare_branch(transformer, images, positions, preservation_probabilities, context, text_mask,
                                  timesteps, mode, gamma, preservation_kv)
    weights = []
    for timestep in timesteps:
        weight = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler,
                                                timestep.reshape(1), device, dit_dtype)
        weights.append(1.0 if weight is None else weight.reshape(()))

    unwrapped_network = accelerator.unwrap_model(network)
    if preservation is not None:
        # Never let teacher's no-grad casts of trainable FP32 LoRA weights
        # populate the autocast cache reused by a student forward.
        with base_model_teacher(unwrapped_network):
            with accelerator.autocast():
                teacher = transformer(**preservation.kwargs).detach()
        torch.clear_autocast_cache()  # Also safe when a caller owns an outer autocast scope.
    loss_target = torch.zeros((), device=device)
    loss_preservation = torch.zeros((), device=device)
    if primary is not None:
        with accelerator.autocast():
            prediction = transformer(**primary.kwargs)
            loss_target = branch_loss(prediction, targets, primary, weights, network_dtype)
        if backward is not None:
            # Leave autocast before backward, so the next student cannot reuse a
            # cached trainable-weight cast whose graph has already been freed.
            backward(target_loss_weight * loss_target)
            loss_target = loss_target.detach()
            del prediction
        # Keep the two student cast graphs independent even under an enclosing
        # autocast scope (also in the combined-loss diagnostic reference).
        torch.clear_autocast_cache()
    if preservation is not None:
        with accelerator.autocast():
            prediction = transformer(**preservation.kwargs)
            loss_preservation = branch_loss(prediction, teacher, preservation, weights, network_dtype,
                                            targets_selected=True)
        if backward is not None:
            backward(preservation_loss_weight * loss_preservation)
            loss_preservation = loss_preservation.detach()
            del prediction
    return target_loss_weight * loss_target + preservation_loss_weight * loss_preservation, {
        "loss_target": loss_target.detach(), "loss_preservation": loss_preservation.detach(),
    }
