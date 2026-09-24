"""Krea2 batches grouped by surviving image-token count across resolutions."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.krea2.alpha_token_mask import align_alpha_mask_to_token_grid, make_alpha_token_keep_mask
from musubi_tuner.utils.model_utils import remove_dtype_suffix


def cache_token_info(path: str, drop_alpha_tokens: bool, patch: int = 2) -> tuple[tuple[int, int], int]:
    """Read only the latent shape and, when requested, the small alpha mask."""
    with safe_open(path, framework="pt", device="cpu") as cache:
        keys = [key for key in cache.keys() if remove_dtype_suffix(key).startswith("latents_")]
        if len(keys) != 1:
            raise ValueError(f"Expected one Krea2 latent in {path}, found {keys}")
        shape = cache.get_slice(keys[0]).get_shape()
        if len(shape) != 4 or shape[1] != 1 or shape[2] % patch or shape[3] % patch:
            raise ValueError(f"Invalid Krea2 latent shape {shape} in {path}")
        h, w = shape[-2:]
        total = (h // patch) * (w // patch)
        if not drop_alpha_tokens or "alpha_mask" not in cache.keys():
            return (h, w), total
        alpha = cache.get_tensor("alpha_mask").float().unsqueeze(0)
    aligned = align_alpha_mask_to_token_grid(alpha, (h, w), patch)
    kept = make_alpha_token_keep_mask(aligned, (h, w), patch, torch.device("cpu"))
    return (h, w), int(kept.sum().item())


class Krea2TokenBucketBatchManager(BucketBatchManager):
    """Carry short bucket remainders down, using every item once per epoch."""

    def __init__(self, bucketed_item_info, batch_size, *, multiple, drop_alpha_tokens,
                 num_timestep_buckets=None, caption_selection_seed=None,
                 caption_dropout_rate=0.0, loss_multiplier=1.0, dry=False):
        if not isinstance(multiple, int) or multiple < 1:
            raise ValueError("--image_token_bucket_multiple must be a positive integer")
        super().__init__(bucketed_item_info, batch_size, num_timestep_buckets,
                         caption_selection_seed, caption_dropout_rate, loss_multiplier)
        self.multiple = multiple
        self.width = multiple * 256
        self.dry = dry
        self.items = [item for bucket in bucketed_item_info.values() for item in bucket]
        self.token_info = {}
        for item in self.items:
            if item.latent_cache_path not in self.token_info:
                self.token_info[item.latent_cache_path] = cache_token_info(item.latent_cache_path, drop_alpha_tokens)
        self.batch_items = []
        self.group_counts = {}
        self.carry_counts = {}
        self.remainder = []
        self.shuffle(raise_on_remainder=not dry)

    def _keep_count(self, item):
        return self.token_info[item.latent_cache_path][1]

    def shuffle(self, raise_on_remainder=True):
        seed = 0 if self.caption_selection_seed is None else self.caption_selection_seed
        rng = random.Random(seed + self.current_epoch)
        groups = defaultdict(list)
        for item in self.items:
            groups[max(0, (self._keep_count(item) - 1) // self.width)].append(item)
        self.group_counts = {key: len(value) for key, value in groups.items()}
        self.batch_items = []
        self.carry_counts = {}
        carry = []
        for group in sorted(groups, reverse=True):
            candidates = groups[group] + carry
            rng.shuffle(candidates)
            full = len(candidates) // self.batch_size
            self.batch_items.extend(
                candidates[i * self.batch_size:(i + 1) * self.batch_size] for i in range(full)
            )
            carry = candidates[full * self.batch_size:]
            self.carry_counts[group] = len(carry)
        self.remainder = carry
        rng.shuffle(self.batch_items)
        self._make_timestep_pool(rng)
        if raise_on_remainder:
            self.assert_full_batches()

    def assert_full_batches(self):
        # A fixed batch dimension is required to avoid an extra compile shape.
        assert not self.remainder, (
            f"Krea2 image-token bucketing: {len(self.items)} items with batch_size={self.batch_size} "
            f"leave {len(self.remainder)} unbatched items. Adjust repeats or batch_size."
        )

    def _make_timestep_pool(self, rng):
        self.timestep_pool = None
        if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
            return
        needed = len(self.batch_items) * self.batch_size
        per_bucket = math.ceil(needed / self.num_timestep_buckets)
        values = [rng.uniform(i / self.num_timestep_buckets, (i + 1) / self.num_timestep_buckets)
                  for i in range(self.num_timestep_buckets) for _ in range(per_bucket)]
        rng.shuffle(values)
        values = values[:needed]
        self.timestep_pool = [values[i:i + self.batch_size] for i in range(0, needed, self.batch_size)]

    def __len__(self):
        return len(self.batch_items)

    def __getitem__(self, idx):
        items = self.batch_items[idx]
        batch = {"latents": [], "alpha_mask": [], "krea2_vl_embed": [], "item_keys": [],
                 "loss_multiplier": self.loss_multiplier,
                 "timesteps": None if self.timestep_pool is None else self.timestep_pool[idx]}
        for offset, item in enumerate(items):
            latent_cache = load_file(item.latent_cache_path)
            latent_keys = [key for key in latent_cache if remove_dtype_suffix(key).startswith("latents_")]
            if len(latent_keys) != 1:
                raise ValueError(f"Expected one latent in {item.latent_cache_path}")
            batch["latents"].append(latent_cache[latent_keys[0]])
            batch["alpha_mask"].append(latent_cache.get("alpha_mask"))
            batch["item_keys"].append(item.item_key)

            if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
                if self.caption_dropout_embedding is None:
                    raise RuntimeError("Krea2 caption dropout embedding was not prepared")
                text = self.caption_dropout_embedding
            else:
                te_cache = load_file(item.text_encoder_output_cache_path)
                self._select_krea2_caption_embed(te_cache, item, idx, offset)
                text_keys = [key for key in te_cache if remove_dtype_suffix(key) == "varlen_krea2_vl_embed"]
                if len(text_keys) != 1:
                    raise ValueError(f"Expected one Krea2 text embedding in {item.text_encoder_output_cache_path}")
                text = te_cache[text_keys[0]]
            batch["krea2_vl_embed"].append(text)
        return batch

    def report(self, dataset_name: str, seed: int):
        """Print the same epoch-one plan used by training, without loading models."""
        if not self.items:
            print(f"Dataset {dataset_name}: no cached training items (seed={seed})")
            return
        by_resolution = Counter(item.bucket_size for item in self.items)
        keep_counts = [self._keep_count(item) for item in self.items]
        image_padding = sum(len(batch) * max(map(self._keep_count, batch))
                            - sum(map(self._keep_count, batch)) for batch in self.batch_items)
        capacity = sum(len(batch) * max(map(self._keep_count, batch)) for batch in self.batch_items)
        old_padding = 0
        for resolution in by_resolution:
            source = [item for item in self.items if item.bucket_size == resolution]
            for offset in range(0, len(source), self.batch_size):
                lengths = [self._keep_count(item) for item in source[offset:offset + self.batch_size]]
                old_padding += len(lengths) * max(lengths) - sum(lengths)
        print(f"Dataset {dataset_name}: unique={len(self.token_info)} items={len(self.items)} "
              f"batch_size={self.batch_size} seed={seed} K={self.multiple} width={self.width}")
        print(f"  resolution buckets: {dict(sorted(by_resolution.items()))}")
        print(f"  image keep-count: min={min(keep_counts)} max={max(keep_counts)} "
              f"groups={dict(sorted(self.group_counts.items()))}")
        print(f"  carry after groups: {dict(sorted(self.carry_counts.items()))}; "
              f"full batches={len(self.batch_items)} remainder={len(self.remainder)}")
        print(f"  image padding={image_padding}/{capacity} ({image_padding / capacity:.2%})" if capacity else
              "  image padding=0/0")
        for number, batch in enumerate(self.batch_items):
            lengths = [self._keep_count(item) for item in batch]
            members = [f"{item.item_key}:{item.bucket_size}:{length}"
                       for item, length in zip(batch, lengths)]
            print(f"  batch {number}: keep={min(lengths)}..{max(lengths)} " + ", ".join(members))
        if self.remainder:
            print("  unbatched: " + ", ".join(item.item_key for item in self.remainder))
        old_batches = sum(math.ceil(count / self.batch_size) for count in by_resolution.values())
        old_partial = sum(count % self.batch_size != 0 for count in by_resolution.values())
        print(f"  old resolution buckets: batches={old_batches} partial={old_partial} "
              f"image_padding={old_padding}")
        print("  Image-only statistics; caption length and final 256-token rounding are excluded.")
