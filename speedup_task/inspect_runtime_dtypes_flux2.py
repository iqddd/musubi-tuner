import argparse
import importlib
import json
import math
import os
import random
import sys
from collections import Counter
from multiprocessing import Value

import torch
from accelerate.utils import set_seed

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import collator_class, prepare_accelerator, read_config_from_file, setup_parser_common
from safetensors.torch import load_file


def summarize_params(module: torch.nn.Module, label: str) -> dict:
    counts_all = Counter()
    counts_trainable = Counter()
    counts_frozen = Counter()
    tensors_all = 0
    tensors_trainable = 0
    tensors_frozen = 0
    numel_all = 0
    numel_trainable = 0
    numel_frozen = 0

    for p in module.parameters():
        dt = str(p.dtype)
        n = p.numel()
        tensors_all += 1
        numel_all += n
        counts_all[dt] += n
        if p.requires_grad:
            tensors_trainable += 1
            numel_trainable += n
            counts_trainable[dt] += n
        else:
            tensors_frozen += 1
            numel_frozen += n
            counts_frozen[dt] += n

    return {
        "label": label,
        "tensors": {
            "all": tensors_all,
            "trainable": tensors_trainable,
            "frozen": tensors_frozen,
        },
        "numel": {
            "all": numel_all,
            "trainable": numel_trainable,
            "frozen": numel_frozen,
        },
        "dtype_numel": {
            "all": dict(counts_all),
            "trainable": dict(counts_trainable),
            "frozen": dict(counts_frozen),
        },
    }


def summarize_param_list(params: list[torch.nn.Parameter], label: str) -> dict:
    counts = Counter()
    tensors = 0
    numel = 0
    for p in params:
        tensors += 1
        numel += p.numel()
        counts[str(p.dtype)] += p.numel()
    return {
        "label": label,
        "tensors": tensors,
        "numel": numel,
        "dtype_numel": dict(counts),
    }


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser.add_argument("--skip_sample_prompt_cache", action="store_true")

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None
    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    trainer = Flux2NetworkTrainer()

    if torch.cuda.is_available():
        if args.cuda_allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if args.cuda_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    if args.dataset_config is None:
        raise ValueError("dataset_config is required")
    if args.dit is None:
        raise ValueError("path to DiT model is required")

    trainer.handle_model_specific_args(args)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    current_epoch = Value("i", 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=trainer.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, training=True, num_timestep_buckets=args.num_timestep_buckets, shared_epoch=current_epoch
    )

    if train_dataset_group.num_train_items == 0:
        raise ValueError("No training items found in dataset")

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)

    accelerator = prepare_accelerator(args)
    if args.mixed_precision is None:
        args.mixed_precision = accelerator.mixed_precision

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    dit_dtype = torch.bfloat16 if args.dit_dtype is None else torch.float32
    dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else dit_dtype

    # Optional: same path as training, but can be skipped to reduce startup time.
    vae = None
    if args.sample_prompts and not args.skip_sample_prompt_cache:
        _ = trainer.process_sample_prompts(args, accelerator, args.sample_prompts)
        vae_dtype = torch.float16 if args.vae_dtype is None else torch.float32
        vae = trainer.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()

    blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
    trainer.blocks_to_swap = blocks_to_swap
    loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

    if args.sdpa:
        attn_mode = "torch"
    elif args.flash_attn:
        attn_mode = "flash"
    elif args.sage_attn:
        attn_mode = "sageattn"
    elif args.xformers:
        attn_mode = "xformers"
    elif args.flash3:
        attn_mode = "flash3"
    else:
        raise ValueError("one of sdpa/flash_attn/flash3/sage_attn/xformers must be enabled")

    transformer = trainer.load_transformer(
        accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
    )
    transformer.eval()
    transformer.requires_grad_(False)

    if blocks_to_swap > 0:
        transformer.enable_block_swap(
            blocks_to_swap, accelerator.device, supports_backward=True, use_pinned_memory=args.use_pinned_memory_for_block_swap
        )
        transformer.move_to_device_except_swap_blocks(accelerator.device)

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "musubi_tuner"))
    network_module = importlib.import_module(args.network_module)

    if args.base_weights is not None:
        for i, weight_path in enumerate(args.base_weights):
            if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                multiplier = 1.0
            else:
                multiplier = args.base_weights_multiplier[i]
            weights_sd = load_file(weight_path)
            module = network_module.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
            module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=")
            net_kwargs[key] = value

    if args.dim_from_weights:
        weights_sd = load_file(args.dim_from_weights)
        network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
    else:
        if hasattr(network_module, "create_arch_network"):
            network = network_module.create_arch_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                None,
                transformer,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        else:
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                None,
                transformer,
                **net_kwargs,
            )

    if hasattr(network_module, "prepare_network"):
        network.prepare_network(args)

    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

    if args.network_weights is not None:
        network.load_weights(args.network_weights)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
        network.enable_gradient_checkpointing()

    trainable_params, _ = network.prepare_optimizer_params(unet_lr=args.learning_rate)
    _, _, optimizer, _, _ = trainer.get_optimizer(args, trainable_params)

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = trainer.get_lr_scheduler(args, optimizer, accelerator.num_processes)

    if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
        transformer.to(dit_weight_dtype)

    pre_accel_transformer = summarize_params(transformer, "transformer_pre_accelerate")
    pre_accel_network = summarize_params(network, "network_pre_accelerate")

    if blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
    else:
        transformer = accelerator.prepare(transformer)

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)

    if args.gradient_checkpointing:
        transformer.train()
    else:
        transformer.eval()

    accelerator.unwrap_model(network).prepare_grad_etc(transformer)

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_network = accelerator.unwrap_model(network)

    post_accel_transformer = summarize_params(unwrapped_transformer, "transformer_post_accelerate")
    post_accel_network = summarize_params(unwrapped_network, "network_post_accelerate")

    if hasattr(unwrapped_network, "get_trainable_params"):
        net_trainable_list = list(unwrapped_network.get_trainable_params())
    else:
        net_trainable_list = [p for p in unwrapped_network.parameters() if p.requires_grad]

    trainable_list_stats = summarize_param_list(net_trainable_list, "network_trainable_param_list")

    optimizer_dtype_counts = Counter()
    optimizer_tensors = 0
    optimizer_numel = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_tensors += 1
            optimizer_numel += p.numel()
            optimizer_dtype_counts[str(p.dtype)] += p.numel()

    payload = {
        "device": str(accelerator.device),
        "mixed_precision": args.mixed_precision,
        "dit_dtype_target": str(dit_dtype),
        "dit_weight_dtype_target": str(dit_weight_dtype),
        "network_module": args.network_module,
        "model_version": args.model_version,
        "train_items": train_dataset_group.num_train_items,
        "batches_per_epoch": len(train_dataloader),
        "max_train_steps": args.max_train_steps,
        "stats": [
            pre_accel_transformer,
            pre_accel_network,
            post_accel_transformer,
            post_accel_network,
            trainable_list_stats,
            {
                "label": "optimizer_param_groups",
                "tensors": optimizer_tensors,
                "numel": optimizer_numel,
                "dtype_numel": dict(optimizer_dtype_counts),
            },
        ],
    }

    print("DTYPE_REPORT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("DTYPE_REPORT_END")


if __name__ == "__main__":
    main()
