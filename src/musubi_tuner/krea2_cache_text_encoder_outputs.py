"""Cache text encoder (Qwen3-VL-4B) outputs for Krea 2 (K2) training.

K2's text encoder returns a stack of *selected* Qwen3-VL hidden-state layers
(shape (B, seq, 12, 2560)) plus an attention mask. The layerwise fusion
(TextFusionTransformer) is trainable and lives inside the DiT, so we cache the raw
selected-layer stack. Padding tokens are dropped per item (varlen) — K2 gives text
tokens zero RoPE position and masks padding in attention, so this is lossless for
the image outputs.
"""

import argparse
import json
import logging
import os

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_krea2
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import remove_dtype_suffix

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


KREA2_MULTI_CAPTION_KEY_PREFIX = "varlen_krea2_vl_embed_caption_"


def caption_variants(item: ItemInfo) -> list[str]:
    return item.caption_variants if item.caption_variants is not None else [item.caption]


def encode_and_save_batch(encoder, batch: list[ItemInfo], prompt_batch_size: int | None, multi_caption: bool = False):
    flattened_prompts: list[tuple[int, str]] = []
    variants_by_item: list[list[str]] = []
    for i, item in enumerate(batch):
        variants = caption_variants(item) if multi_caption else [item.caption]
        variants_by_item.append(variants)
        print(f"Item {i}: {item.item_key}, prompts: {variants}")
        flattened_prompts.extend((i, prompt) for prompt in variants)

    embeds_by_item: list[list[torch.Tensor]] = [[] for _ in batch]
    prompt_batch_size = prompt_batch_size or len(flattened_prompts)
    for start in range(0, len(flattened_prompts), prompt_batch_size):
        prompt_batch = flattened_prompts[start : start + prompt_batch_size]
        hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [prompt for _, prompt in prompt_batch])

        for (item_index, _), hidden_i, mask_i in zip(prompt_batch, hiddens, mask):
            valid = mask_i.bool()
            embeds_by_item[item_index].append(hidden_i[valid])  # (valid_len, L, D)

    for item, embeds, variants in zip(batch, embeds_by_item, variants_by_item):
        save_text_encoder_output_cache_krea2(item, embeds, variants, multi_caption=multi_caption)


def is_krea2_text_encoder_cache_current(item: ItemInfo) -> bool:
    """Validate Krea's caption list and tensor set for --skip_existing."""
    cache_path = item.text_encoder_output_cache_path
    if cache_path is None or not os.path.exists(cache_path):
        return False

    expected_captions = caption_variants(item)
    try:
        with safetensors_utils.MemoryEfficientSafeOpen(cache_path) as cache_file:
            metadata = cache_file.metadata()
            if metadata.get("architecture") != ARCHITECTURE_KREA2_FULL:
                return False

            cached_captions_json = metadata.get("captions")
            if cached_captions_json is None:
                # Legacy one-caption cache remains usable without re-caching.
                if len(expected_captions) != 1 or metadata.get("caption1") != expected_captions[0]:
                    return False
                return any(remove_dtype_suffix(key) == "varlen_krea2_vl_embed" for key in cache_file.keys())

            if json.loads(cached_captions_json) != expected_captions:
                return False

            expected_bases = {f"{KREA2_MULTI_CAPTION_KEY_PREFIX}{index:04d}" for index in range(len(expected_captions))}
            cached_bases = {
                remove_dtype_suffix(key)
                for key in cache_file.keys()
                if remove_dtype_suffix(key).startswith(KREA2_MULTI_CAPTION_KEY_PREFIX)
            }
            return cached_bases == expected_bases and metadata.get("caption_count") == str(len(expected_captions))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Krea 2 TE cache is unreadable or invalid and will be rebuilt: {cache_path} ({exc})")
        return False


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3-VL-4B text encoder safetensors path (official or ComfyUI key layout)",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for the text encoder, default is bfloat16")
    parser.add_argument(
        "--multi_caption",
        action="store_true",
        help="treat each non-empty line in directory caption files as a separate Krea 2 caption alternative",
    )
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_KREA2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=te_dtype, device=device)

    logger.info("Encoding with Qwen3-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal encoder
        encode_and_save_batch(encoder, batch, args.batch_size, multi_caption=args.multi_caption)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        is_cache_valid=is_krea2_text_encoder_cache_current,
        multi_caption=args.multi_caption,
    )
    del encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
