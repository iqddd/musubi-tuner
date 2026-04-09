# BOFT module for FLUX.2 Klein

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from musubi_tuner.networks import boft


FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*"]
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.append(r".*(norm).*")
    kwargs["exclude_patterns"] = exclude_patterns

    return boft.create_arch_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        FLUX_2_TARGET_REPLACE_MODULES,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    return boft.create_network_from_weights(
        FLUX_2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
