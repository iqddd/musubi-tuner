"""CPU-only regression tests for Krea 2 dataset-level caption dropout."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from voluptuous import MultipleInvalid

from musubi_tuner import krea2_train_network
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_WAN
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_krea2
from musubi_tuner.dataset.config_utils import BaseDatasetParams, BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ImageDataset, ItemInfo
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


def _blueprint(config):
    return BlueprintGenerator(ConfigSanitizer()).generate(config, Namespace(), architecture=ARCHITECTURE_KREA2)


def test_caption_dropout_config_default_and_general_fallback():
    assert BaseDatasetParams().caption_dropout_rate == 0.0

    blueprint = _blueprint(
        {
            "general": {"caption_dropout_rate": 0.1},
            "datasets": [{"image_directory": "/dataset/first"}, {"image_directory": "/dataset/second"}],
        }
    )
    assert [dataset.params.caption_dropout_rate for dataset in blueprint.dataset_group.datasets] == [0.1, 0.1]


def test_caption_dropout_config_per_dataset_override():
    blueprint = _blueprint(
        {
            "general": {"caption_dropout_rate": 0.1},
            "datasets": [
                {"image_directory": "/dataset/first", "caption_dropout_rate": 0.25},
                {"image_directory": "/dataset/second"},
            ],
        }
    )
    assert [dataset.params.caption_dropout_rate for dataset in blueprint.dataset_group.datasets] == [0.25, 0.1]


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_caption_dropout_config_rejects_out_of_range_values(rate):
    with pytest.raises(MultipleInvalid):
        _blueprint({"general": {"caption_dropout_rate": rate}, "datasets": [{"image_directory": "/dataset"}]})


def test_non_krea_dataset_rejects_enabled_caption_dropout():
    with pytest.raises(ValueError, match="supported only for Krea 2"):
        BaseDataset(architecture=ARCHITECTURE_WAN, caption_dropout_rate=0.1)


def _make_item(tmp_path, embeds=None):
    item = ItemInfo("sample", "", (16, 16), (16, 16))
    item.latent_cache_path = str(tmp_path / "sample_0016x0016_kr2.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "sample_kr2_te.safetensors")
    save_file({"latents_1x1x1_float32": torch.zeros(1, 1, 1, 1)}, item.latent_cache_path)

    embeds = embeds or [torch.full((1, 2, 3), 1.0)]
    save_text_encoder_output_cache_krea2(item, embeds, [f"caption {index}" for index in range(len(embeds))])
    return item


def test_image_dataset_propagates_rate_to_metadata_and_batch_manager(tmp_path):
    _make_item(tmp_path)
    dataset = ImageDataset(
        resolution=(16, 16),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        image_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=ARCHITECTURE_KREA2,
        caption_dropout_rate=0.3,
    )
    dataset.prepare_for_training()

    assert dataset.get_metadata()["caption_dropout_rate"] == 0.3
    assert dataset.batch_manager.caption_dropout_rate == 0.3


def test_bucket_uses_in_memory_empty_embedding_when_dropout_fires(tmp_path, monkeypatch):
    item = _make_item(tmp_path, [torch.full((1, 2, 3), 1.0), torch.full((1, 2, 3), 2.0)])
    empty = torch.zeros(2, 2, 3)
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234, caption_dropout_rate=0.5)
    manager.set_caption_dropout_embedding(empty)
    monkeypatch.setattr("musubi_tuner.dataset.bucket.random.random", lambda: 0.1)

    assert torch.equal(manager[0]["krea2_vl_embed"][0], empty)


def test_bucket_keeps_deterministic_multi_caption_selection_when_dropout_does_not_fire(tmp_path, monkeypatch):
    embeds = [torch.full((1, 2, 3), float(index)) for index in range(3)]
    item = _make_item(tmp_path, embeds)
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234, caption_dropout_rate=0.5)
    manager.set_caption_dropout_embedding(torch.zeros(1, 2, 3))
    monkeypatch.setattr("musubi_tuner.dataset.bucket.random.random", lambda: 0.9)
    expected_index = manager._select_krea2_caption_index(3, 1234, 0, 0, 0, item.item_key)

    assert torch.equal(manager[0]["krea2_vl_embed"][0], embeds[expected_index])


def test_bucket_rate_zero_needs_no_empty_embedding(tmp_path):
    embed = torch.full((1, 2, 3), 7.0)
    item = _make_item(tmp_path, [embed])
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234)

    assert torch.equal(manager[0]["krea2_vl_embed"][0], embed)


def test_bucket_rate_one_always_uses_empty_embedding(tmp_path, monkeypatch):
    item = _make_item(tmp_path)
    empty = torch.zeros(1, 2, 3)
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_dropout_rate=1.0)
    manager.set_caption_dropout_embedding(empty)
    monkeypatch.setattr("musubi_tuner.dataset.bucket.random.random", lambda: 0.999999)

    assert torch.equal(manager[0]["krea2_vl_embed"][0], empty)


def test_bucket_enabled_dropout_requires_prepared_embedding(tmp_path):
    item = _make_item(tmp_path)
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, caption_selection_seed=1234, caption_dropout_rate=0.5)

    with pytest.raises(RuntimeError, match="was not prepared"):
        manager[0]


class _FakeDataset:
    def __init__(self, rate):
        self.caption_dropout_rate = rate
        self.embedding = None

    def set_caption_dropout_embedding(self, embedding):
        self.embedding = embedding


class _FakeVae:
    def requires_grad_(self, value):
        return self

    def eval(self):
        return self


def _patch_prompt_encoder(monkeypatch):
    loaded = []
    encoded = []

    def fake_load(path, dtype, device):
        loaded.append((path, dtype, device))
        return object()

    def fake_encode(encoder, prompts):
        encoded.extend(prompts)
        hiddens = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 2, 4)
        mask = torch.tensor([[True, False, True]])
        return hiddens, mask

    monkeypatch.setattr(krea2_train_network.krea2_utils, "load_krea2_text_encoder", fake_load)
    monkeypatch.setattr(krea2_train_network.krea2_utils, "get_krea2_prompt_embeds", fake_encode)
    monkeypatch.setattr(krea2_train_network, "clean_memory_on_device", lambda device: None)
    return loaded, encoded


def test_trainer_prepares_trimmed_shared_cpu_empty_embedding(monkeypatch):
    loaded, encoded = _patch_prompt_encoder(monkeypatch)
    dataset = _FakeDataset(0.2)
    group = SimpleNamespace(datasets=[dataset])
    args = SimpleNamespace(text_encoder="encoder.safetensors", sample_prompts=None, vae=None)

    sample_parameters, vae = Krea2NetworkTrainer()._prepare_sampling(
        args, SimpleNamespace(device=torch.device("cpu")), torch.float16, group
    )

    assert len(loaded) == 1
    assert encoded == [""]
    assert sample_parameters is None
    assert vae is None
    assert dataset.embedding.device.type == "cpu"
    assert dataset.embedding.shape == (2, 2, 4)
    assert dataset.embedding.is_contiguous()
    assert dataset.embedding.is_shared()


def test_trainer_reuses_one_encoder_load_for_dropout_and_sample_prompts(monkeypatch):
    loaded, encoded = _patch_prompt_encoder(monkeypatch)
    prompts = [{"prompt": "positive", "negative_prompt": "negative"}]
    monkeypatch.setattr(krea2_train_network, "load_prompts", lambda path: prompts)

    trainer = Krea2NetworkTrainer()
    vae = _FakeVae()
    monkeypatch.setattr(trainer, "load_vae", lambda args, vae_dtype, vae_path: vae)
    dataset = _FakeDataset(0.2)
    args = SimpleNamespace(text_encoder="encoder.safetensors", sample_prompts="prompts.txt", vae="vae.safetensors")

    sample_parameters, prepared_vae = trainer._prepare_sampling(
        args, SimpleNamespace(device=torch.device("cpu")), torch.float16, SimpleNamespace(datasets=[dataset])
    )

    assert len(loaded) == 1
    assert encoded == ["", "positive", "negative"]
    assert prepared_vae is vae
    assert len(sample_parameters) == 1
    assert "krea2_vl_embed" in sample_parameters[0]
    assert "negative_krea2_vl_embed" in sample_parameters[0]


def test_trainer_requires_text_encoder_when_dropout_is_enabled():
    dataset = _FakeDataset(0.2)
    args = SimpleNamespace(text_encoder=None, sample_prompts=None, vae=None)

    with pytest.raises(ValueError, match="--text_encoder is required"):
        Krea2NetworkTrainer()._prepare_sampling(
            args, SimpleNamespace(device=torch.device("cpu")), torch.float16, SimpleNamespace(datasets=[dataset])
        )
