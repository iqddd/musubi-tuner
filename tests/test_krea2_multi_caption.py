"""CPU-only regression tests for Krea 2 multi-caption TE caches."""

import random

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2_FULL
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_krea2
from musubi_tuner.dataset.datasources import split_caption_file_variants
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner import krea2_cache_text_encoder_outputs
from musubi_tuner.krea2_cache_text_encoder_outputs import is_krea2_text_encoder_cache_current
from musubi_tuner.utils import safetensors_utils


def _item(tmp_path, name="sample"):
    item = ItemInfo(name, "", (16, 16), (16, 16))
    item.latent_cache_path = str(tmp_path / f"{name}_latent.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / f"{name}_te.safetensors")
    safetensors_utils.mem_eff_save_file(
        {"latents_1x1x1_float32": torch.zeros(1, 1, 1, 1)}, item.latent_cache_path, metadata={}
    )
    return item


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("first\nsecond\nthird", ["first", "second", "third"]),
        ("first\rsecond\rthird", ["first", "second", "third"]),
        ("first\r\nsecond\r\nthird", ["first", "second", "third"]),
        (" first \r\n\r\n second \n  ", ["first", "second"]),
    ],
)
def test_split_caption_file_variants_handles_all_newline_formats(caption, expected):
    assert split_caption_file_variants(caption, "/dataset/sample.txt") == expected


def test_split_caption_file_variants_rejects_empty_caption_file():
    with pytest.raises(ValueError, match="no non-empty caption variants"):
        split_caption_file_variants(" \r\n\n\t\r", "/dataset/empty.txt")


def test_multi_caption_cache_writes_metadata_and_replaces_stale_variants(tmp_path):
    item = _item(tmp_path)
    old_embeds = [torch.full((index + 1, 2, 3), float(index), dtype=torch.float32) for index in range(3)]
    save_text_encoder_output_cache_krea2(item, old_embeds, ["old 0", "old 1", "old 2"])

    embeds = [torch.full((index + 1, 2, 3), float(index + 10), dtype=torch.float32) for index in range(2)]
    captions = ["new 0", "new 1"]
    save_text_encoder_output_cache_krea2(item, embeds, captions)

    tensors = load_file(item.text_encoder_output_cache_path)
    assert set(tensors) == {
        "varlen_krea2_vl_embed_caption_0000_float32",
        "varlen_krea2_vl_embed_caption_0001_float32",
    }
    assert torch.equal(tensors["varlen_krea2_vl_embed_caption_0001_float32"], embeds[1])

    with safetensors_utils.MemoryEfficientSafeOpen(item.text_encoder_output_cache_path) as cache_file:
        metadata = cache_file.metadata()
    assert metadata["architecture"] == ARCHITECTURE_KREA2_FULL
    assert metadata["caption1"] == "new 0"
    assert metadata["captions"] == '["new 0", "new 1"]'
    assert metadata["caption_count"] == "2"
    assert metadata["format_version"] == "1.1.0"

    item.caption_variants = captions
    assert is_krea2_text_encoder_cache_current(item)
    item.caption_variants = ["new 0", "changed"]
    assert not is_krea2_text_encoder_cache_current(item)


def test_multi_caption_encoding_flattens_variants_in_prompt_sized_batches(tmp_path, monkeypatch):
    first = _item(tmp_path, "first")
    first.caption_variants = ["first 0", "first 1"]
    second = _item(tmp_path, "second")
    second.caption_variants = ["second 0", "second 1", "second 2"]

    calls = []

    def fake_get_prompt_embeds(_, prompts):
        calls.append(prompts)
        values = torch.tensor([float(len(calls) * 10 + index) for index in range(len(prompts))])
        return values.reshape(-1, 1, 1, 1), torch.ones(len(prompts), 1, dtype=torch.bool)

    monkeypatch.setattr(krea2_cache_text_encoder_outputs.krea2_utils, "get_krea2_prompt_embeds", fake_get_prompt_embeds)
    krea2_cache_text_encoder_outputs.encode_and_save_batch(object(), [first, second], prompt_batch_size=2, multi_caption=True)

    assert calls == [["first 0", "first 1"], ["second 0", "second 1"], ["second 2"]]
    first_tensors = load_file(first.text_encoder_output_cache_path)
    second_tensors = load_file(second.text_encoder_output_cache_path)
    assert set(first_tensors) == {
        "varlen_krea2_vl_embed_caption_0000_float32",
        "varlen_krea2_vl_embed_caption_0001_float32",
    }
    assert set(second_tensors) == {
        "varlen_krea2_vl_embed_caption_0000_float32",
        "varlen_krea2_vl_embed_caption_0001_float32",
        "varlen_krea2_vl_embed_caption_0002_float32",
    }


def test_default_caption_caching_keeps_multiline_caption_as_one_prompt(tmp_path, monkeypatch):
    item = _item(tmp_path)
    item.caption = "first line\nsecond line"
    calls = []

    def fake_get_prompt_embeds(_, prompts):
        calls.append(prompts)
        return torch.ones(len(prompts), 1, 1, 1), torch.ones(len(prompts), 1, dtype=torch.bool)

    monkeypatch.setattr(krea2_cache_text_encoder_outputs.krea2_utils, "get_krea2_prompt_embeds", fake_get_prompt_embeds)
    krea2_cache_text_encoder_outputs.encode_and_save_batch(object(), [item], prompt_batch_size=4)

    assert calls == [["first line\nsecond line"]]
    tensors = load_file(item.text_encoder_output_cache_path)
    assert set(tensors) == {"varlen_krea2_vl_embed_float32"}
    with safetensors_utils.MemoryEfficientSafeOpen(item.text_encoder_output_cache_path) as cache_file:
        assert "captions" not in cache_file.metadata()


def test_bucket_selects_one_krea_caption_deterministically_and_ignores_global_rng(tmp_path):
    item = _item(tmp_path)
    embeds = [torch.full((1, 2, 3), float(index), dtype=torch.float32) for index in range(5)]
    save_text_encoder_output_cache_krea2(item, embeds, [f"caption {index}" for index in range(len(embeds))])

    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234)
    manager.set_current_epoch(7)
    expected_index = manager._select_krea2_caption_index(5, 1234, 7, 0, 0, item.item_key)

    random.seed(1)
    first = manager[0]["krea2_vl_embed"][0]
    random.seed(999999)
    second = manager[0]["krea2_vl_embed"][0]

    assert torch.equal(first, embeds[expected_index])
    assert torch.equal(second, embeds[expected_index])


def test_bucket_loads_legacy_single_caption_krea_cache(tmp_path):
    item = _item(tmp_path, "legacy")
    legacy_embed = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    safetensors_utils.mem_eff_save_file(
        {"varlen_krea2_vl_embed_float32": legacy_embed},
        item.text_encoder_output_cache_path,
        metadata={"architecture": ARCHITECTURE_KREA2_FULL, "caption1": "legacy", "format_version": "1.0.1"},
    )

    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234)
    assert torch.equal(manager[0]["krea2_vl_embed"][0], legacy_embed)

    item.caption_variants = ["legacy"]
    assert is_krea2_text_encoder_cache_current(item)
