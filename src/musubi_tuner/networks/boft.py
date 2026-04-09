# BOFT (Butterfly Orthogonal Fine-Tuning) network module
# Linear layers only.
#
# Export format is compatible with BOFT weights that store:
# - oft_blocks
# - optional rescale
# - alpha

import ast
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from musubi_tuner.networks import lora as lora_module

logger = logging.getLogger(__name__)


def power2factorization(dimension: int, factor: int = -1) -> tuple[Optional[int], int]:
    if factor == -1:
        factor = dimension

    m = n = 0
    while m <= factor:
        m += 2
        while dimension % m != 0 and m < dimension:
            m += 2
        if m > factor:
            break
        if sum(int(i) for i in f"{dimension // m:b}") == 1:
            n = dimension // m

    if n == 0:
        return None, n
    return dimension // n, n


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    block_size, block_num = power2factorization(dimension, factor)
    if block_num == 0 or block_size is None:
        raise ValueError(f"BOFT cannot decompose dimension={dimension} with network_dim={factor}")
    return block_size, block_num


def _bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() == "true"


def _stage_count(block_num: int) -> int:
    return sum(int(bit) for bit in f"{block_num - 1:b}") + 1


def _packed_block_size(block_size: int) -> int:
    return block_size * (block_size - 1) // 2


def _alpha_tensor(alpha: float | torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(alpha, torch.Tensor):
        return alpha.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    return torch.tensor(float(alpha), device=ref_tensor.device, dtype=ref_tensor.dtype)


def export_oft_blocks_to_packed(oft_blocks: torch.Tensor) -> torch.Tensor:
    block_size = oft_blocks.shape[-1]
    upper_rows, upper_cols = torch.triu_indices(block_size, block_size, offset=1, device=oft_blocks.device)
    skew = oft_blocks - oft_blocks.transpose(-1, -2)
    return skew[..., upper_rows, upper_cols]


def packed_oft_blocks_to_export(packed_oft_blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    upper_rows, upper_cols = torch.triu_indices(block_size, block_size, offset=1, device=packed_oft_blocks.device)
    export_blocks = packed_oft_blocks.new_zeros(*packed_oft_blocks.shape[:-1], block_size, block_size)
    export_blocks[..., upper_rows, upper_cols] = packed_oft_blocks
    return export_blocks


def export_oft_blocks_to_skew(oft_blocks: torch.Tensor) -> torch.Tensor:
    return oft_blocks - oft_blocks.transpose(-1, -2)


def build_rotation(skew_blocks: torch.Tensor, alpha: float | torch.Tensor) -> torch.Tensor:
    block_size = skew_blocks.shape[-1]
    work_dtype = torch.float32
    q = skew_blocks.to(work_dtype)
    identity = torch.eye(block_size, device=skew_blocks.device, dtype=work_dtype)
    alpha_tensor = _alpha_tensor(alpha, q)
    q_norm = torch.linalg.vector_norm(q)
    clip_scale = torch.minimum(alpha_tensor / (q_norm + 1e-8), torch.ones_like(alpha_tensor))
    clip_scale = torch.where(alpha_tensor > 0, clip_scale, torch.ones_like(clip_scale))
    normed_q = q * clip_scale
    rotation = (identity + normed_q) @ torch.linalg.inv(identity - normed_q)
    return rotation.to(skew_blocks.dtype)


def apply_boft_lastdim(
    tensor: torch.Tensor,
    skew_blocks: torch.Tensor,
    alpha: float | torch.Tensor,
    scale: float = 1.0,
    rescale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    boft_m, _, block_size, _ = skew_blocks.shape
    rotation = build_rotation(skew_blocks, alpha).to(device=tensor.device, dtype=tensor.dtype)
    identity = torch.eye(block_size, device=tensor.device, dtype=tensor.dtype)

    inp = tensor
    half_block = block_size // 2
    for i in range(boft_m):
        stage = rotation[i]
        g = 2
        k = (2**i) * half_block
        if scale != 1.0:
            stage = torch.lerp(identity, stage, scale)
        inp = inp.unflatten(-1, (-1, g, k)).transpose(-2, -1).flatten(-3).unflatten(-1, (-1, block_size))
        inp = torch.einsum("b i j, ... b j -> ... b i", stage, inp)
        inp = inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)

    if rescale is not None:
        inp = inp * rescale.reshape(-1).to(device=inp.device, dtype=inp.dtype)

    return inp


def apply_boft_to_weight(
    weight: torch.Tensor,
    skew_blocks: torch.Tensor,
    alpha: float | torch.Tensor,
    scale: float = 1.0,
    rescale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    transformed = apply_boft_lastdim(weight.transpose(0, 1), skew_blocks, alpha, scale=scale, rescale=rescale)
    return transformed.transpose(0, 1)


def boft_diff_from_weight(
    weight: torch.Tensor,
    oft_blocks: torch.Tensor,
    alpha: float | torch.Tensor,
    scale: float = 1.0,
    rescale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    transformed = apply_boft_to_weight(weight, export_oft_blocks_to_skew(oft_blocks), alpha, scale=scale, rescale=rescale)
    return (transformed - weight) * scale


class BOFTModule(torch.nn.Module):
    """BOFT module for training. Replaces the forward method of the original Linear."""

    _dropout_warning_emitted = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=0.0,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        rescaled=False,
        train_representation="full",
        module_layouts: Optional[Dict[str, Dict[str, int]]] = None,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ != "Linear":
            raise ValueError("BOFT only supports Linear in this implementation")

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().item()
        alpha = 0.0 if alpha is None else float(alpha)

        layout = None if module_layouts is None else module_layouts.get(lora_name)
        if layout is None:
            block_size, block_num = butterfly_factor(org_module.out_features, int(lora_dim))
            rescaled = _bool_arg(rescaled)
            train_representation = str(train_representation).lower()
        else:
            block_size = int(layout["block_size"])
            block_num = int(layout["block_num"])
            rescaled = bool(layout.get("rescaled", False))
            train_representation = str(layout.get("train_representation", train_representation)).lower()

        if train_representation not in {"full", "packed"}:
            raise ValueError(f"Unsupported BOFT train_representation={train_representation!r}, expected 'full' or 'packed'")

        if block_size * block_num != org_module.out_features:
            raise ValueError(
                f"BOFT invalid layout for {lora_name}: out_features={org_module.out_features}, "
                f"block_size={block_size}, block_num={block_num}"
            )

        self.block_size = block_size
        self.block_num = block_num
        self.boft_m = _stage_count(block_num)
        self.rescaled = rescaled
        self.train_representation = train_representation

        if self.train_representation == "packed":
            self.packed_block_size = _packed_block_size(self.block_size)
            self.oft_blocks = nn.Parameter(torch.zeros(self.boft_m, self.block_num, self.packed_block_size))
            upper_rows, upper_cols = torch.triu_indices(self.block_size, self.block_size, offset=1)
            self.register_buffer("_upper_flat_idx", upper_rows * self.block_size + upper_cols, persistent=False)
            self.register_buffer("_lower_flat_idx", upper_cols * self.block_size + upper_rows, persistent=False)
        else:
            self.packed_block_size = None
            self.oft_blocks = nn.Parameter(torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size))
        if self.rescaled:
            self.rescale = nn.Parameter(torch.ones(org_module.out_features, 1))

        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if not BOFTModule._dropout_warning_emitted:
            if (dropout is not None and float(dropout) > 0) or (rank_dropout is not None and float(rank_dropout) > 0):
                logger.warning("BOFT ignores neuron dropout and rank dropout in this implementation")
                BOFTModule._dropout_warning_emitted = True

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _rescale_param(self):
        return self.rescale if self.rescaled else None

    def _expand_oft_blocks(self, signed: bool) -> torch.Tensor:
        if self.train_representation == "full":
            return export_oft_blocks_to_skew(self.oft_blocks) if signed else self.oft_blocks
        flat = packed_oft_blocks_to_export(self.oft_blocks, self.block_size).flatten(-2)
        if signed:
            flat[..., self._lower_flat_idx] = -self.oft_blocks
        return flat.unflatten(-1, (self.block_size, self.block_size))

    def export_oft_blocks(self) -> torch.Tensor:
        return self._expand_oft_blocks(signed=False)

    def skew_oft_blocks(self) -> torch.Tensor:
        return self._expand_oft_blocks(signed=True)

    def get_diff_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        org_weight = self.org_module_ref[0].weight.to(torch.float)
        return (
            apply_boft_to_weight(
            org_weight,
            self.skew_oft_blocks().to(torch.float),
            self.alpha,
            scale=multiplier,
            rescale=self._rescale_param(),
            )
            - org_weight
        ) * multiplier

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1, device=x.device) < self.module_dropout:
                return org_forwarded

        bias = self.org_module_ref[0].bias
        projected = org_forwarded if bias is None else org_forwarded - bias.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
        transformed = apply_boft_lastdim(
            projected,
            self.skew_oft_blocks(),
            self.alpha,
            scale=self.multiplier,
            rescale=self._rescale_param(),
        )
        return org_forwarded + (transformed - projected) * self.multiplier


class BOFTInfModule(BOFTModule):
    """BOFT module for inference. Supports merge_to and get_weight."""

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=0.0, **kwargs):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, **kwargs)
        self.enabled = True
        self.network = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        oft_blocks = sd["oft_blocks"].to(device, dtype=torch.float, non_blocking=non_blocking)
        rescale = sd.get("rescale")
        if rescale is not None:
            rescale = rescale.to(device, dtype=torch.float, non_blocking=non_blocking)
        alpha = sd.get("alpha", torch.tensor(0.0))
        if isinstance(alpha, torch.Tensor):
            alpha = float(alpha.item())
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)
        diff_weight = boft_diff_from_weight(weight, oft_blocks, alpha, self.multiplier, rescale)
        org_sd["weight"] = (weight + diff_weight).to(org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        return self.get_diff_weight(self.multiplier if multiplier is None else multiplier)

    def default_forward(self, x):
        return super().forward(x)

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class BOFTNetwork(lora_module.LoRANetwork):
    def _export_state_dict(self, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        state_dict: Dict[str, torch.Tensor] = {}
        for module in self.text_encoder_loras + self.unet_loras:
            state_dict[f"{module.lora_name}.oft_blocks"] = module.export_oft_blocks().detach().clone().to("cpu")
            state_dict[f"{module.lora_name}.alpha"] = module.alpha.detach().clone().to("cpu")
            if module.rescaled:
                state_dict[f"{module.lora_name}.rescale"] = module.rescale.detach().clone().to("cpu")

        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(dtype)
        return state_dict

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self._export_state_dict(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from musubi_tuner.utils import model_utils

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        expected_state = self.state_dict()
        converted_sd = {}
        for key, value in weights_sd.items():
            expected_value = expected_state.get(key)
            if key.endswith(".oft_blocks") and expected_value is not None:
                if expected_value.ndim == 3 and value.ndim == 4:
                    value = export_oft_blocks_to_packed(value)
                elif expected_value.ndim == 4 and value.ndim == 3:
                    value = packed_oft_blocks_to_export(value, int(expected_value.shape[-1]))
            converted_sd[key] = value
        return self.load_state_dict(converted_sd, False)


def create_network(
    target_replace_modules: List[str],
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    module_kwargs: Optional[Dict] = None,
    **kwargs,
):
    kwargs = dict(kwargs)
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 0.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = _bool_arg(kwargs.get("verbose", False))

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    return BOFTNetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=BOFTModule,
        module_kwargs=module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
    )


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    modules_dim = {}
    modules_alpha = {}
    module_layouts: Dict[str, Dict[str, int]] = {}
    train_representation = str(kwargs.get("train_representation", "full")).lower()
    if for_inference:
        train_representation = "full"

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name, tensor_name = key.split(".", 1)
        if tensor_name == "alpha":
            modules_alpha[lora_name] = value
        elif tensor_name == "oft_blocks":
            if value.ndim != 4:
                raise ValueError(f"BOFT weight {key} must be rank-4, got {value.ndim}")
            modules_dim[lora_name] = int(value.shape[2])
            modules_alpha.setdefault(lora_name, torch.tensor(0.0))
            layout = module_layouts.setdefault(lora_name, {})
            layout.update(
                {
                    "boft_m": int(value.shape[0]),
                    "block_num": int(value.shape[1]),
                    "block_size": int(value.shape[2]),
                }
            )
            layout.setdefault("rescaled", False)
            layout.setdefault("train_representation", train_representation)
        elif tensor_name == "rescale":
            module_layouts.setdefault(lora_name, {})["rescaled"] = True

    module_class = BOFTInfModule if for_inference else BOFTModule
    network = BOFTNetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs={"module_layouts": module_layouts, "train_representation": train_representation},
    )
    return network


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    target_replace_modules: List[str],
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    kwargs = dict(kwargs)
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    kwargs["exclude_patterns"] = exclude_patterns

    rescaled = _bool_arg(kwargs.pop("rescaled", False))
    train_representation = str(kwargs.pop("train_representation", "full")).lower()
    module_kwargs = {"rescaled": rescaled, "train_representation": train_representation}

    return create_network(
        target_replace_modules,
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        module_kwargs=module_kwargs,
        **kwargs,
    )


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    oft_blocks_key = lora_name + ".oft_blocks"
    rescale_key = lora_name + ".rescale"
    alpha_key = lora_name + ".alpha"

    if oft_blocks_key not in lora_weight_keys:
        return model_weight
    if model_weight.dim() != 2:
        raise ValueError(f"BOFT native merge only supports Linear weights, got shape={tuple(model_weight.shape)} for {lora_name}")

    oft_blocks = lora_sd[oft_blocks_key].to(calc_device)
    rescale = lora_sd.get(rescale_key)
    if rescale is not None:
        rescale = rescale.to(calc_device)
    alpha = lora_sd.get(alpha_key, torch.tensor(0.0))
    if isinstance(alpha, torch.Tensor):
        alpha = float(alpha.item())

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(torch.float16)
        oft_blocks = oft_blocks.to(torch.float16)
        if rescale is not None:
            rescale = rescale.to(torch.float16)

    model_weight = model_weight.to(calc_device)
    model_weight = model_weight + boft_diff_from_weight(model_weight, oft_blocks, alpha, multiplier, rescale)

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    lora_weight_keys.discard(oft_blocks_key)
    lora_weight_keys.discard(rescale_key)
    lora_weight_keys.discard(alpha_key)
    return model_weight
