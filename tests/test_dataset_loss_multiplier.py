"""CPU-only regression tests for dataset-level LoRA loss multipliers."""

from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file
from voluptuous import MultipleInvalid

from musubi_tuner.dataset.architectures import ARCHITECTURE_WAN
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.config_utils import BaseDatasetParams, BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ImageDataset, ItemInfo, VideoDataset
from musubi_tuner.training.trainer_base import NetworkTrainer


def _blueprint(config):
    return BlueprintGenerator(ConfigSanitizer()).generate(config, Namespace(), architecture=ARCHITECTURE_WAN)


def _write_cache_pair(tmp_path, item_key, latent_filename):
    latent_path = tmp_path / latent_filename
    text_encoder_path = tmp_path / f"{item_key}_{ARCHITECTURE_WAN}_te.safetensors"
    save_file({"latents_1x1x1_float32": torch.zeros(1, 1, 1, 1)}, latent_path)
    save_file({"text_embed_float32": torch.zeros(1, 2, 3)}, text_encoder_path)
    return latent_path, text_encoder_path


def test_loss_multiplier_config_default_general_and_dataset_override():
    assert BaseDatasetParams().loss_multiplier == 1.0

    blueprint = _blueprint(
        {
            "general": {"loss_multiplier": 1.5},
            "datasets": [
                {"image_directory": "/dataset/first", "loss_multiplier": 0.0},
                {"image_directory": "/dataset/second"},
            ],
        }
    )

    assert [dataset.params.loss_multiplier for dataset in blueprint.dataset_group.datasets] == [0.0, 1.5]


@pytest.mark.parametrize("multiplier", [-0.01, float("nan"), float("inf"), float("-inf")])
def test_loss_multiplier_config_rejects_invalid_values(multiplier):
    with pytest.raises(MultipleInvalid, match="loss_multiplier must be a finite number"):
        _blueprint({"general": {"loss_multiplier": multiplier}, "datasets": [{"image_directory": "/dataset"}]})


def test_loss_multiplier_is_in_metadata_and_validated_at_dataset_boundary():
    dataset = BaseDataset(architecture=ARCHITECTURE_WAN, loss_multiplier=2.5)
    assert dataset.get_metadata()["loss_multiplier"] == 2.5

    with pytest.raises(ValueError, match="loss_multiplier must be a finite number"):
        BaseDataset(architecture=ARCHITECTURE_WAN, loss_multiplier=float("inf"))


def test_image_and_video_datasets_propagate_loss_multiplier_to_batch_manager(tmp_path):
    image_dir = tmp_path / "image"
    video_dir = tmp_path / "video"
    image_dir.mkdir()
    video_dir.mkdir()

    _write_cache_pair(image_dir, "image", f"image_0016x0016_{ARCHITECTURE_WAN}.safetensors")
    image_dataset = ImageDataset(
        resolution=(16, 16),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        image_directory=str(image_dir),
        cache_directory=str(image_dir),
        architecture=ARCHITECTURE_WAN,
        loss_multiplier=1.25,
    )
    image_dataset.prepare_for_training()

    _write_cache_pair(video_dir, "video", f"video_00000-001_0016x0016_{ARCHITECTURE_WAN}.safetensors")
    video_dataset = VideoDataset(
        resolution=(16, 16),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        target_frames=[1],
        video_directory=str(video_dir),
        cache_directory=str(video_dir),
        architecture=ARCHITECTURE_WAN,
        loss_multiplier=0.75,
    )
    video_dataset.prepare_for_training()

    assert image_dataset.batch_manager.loss_multiplier == 1.25
    assert video_dataset.batch_manager.loss_multiplier == 0.75


def test_bucket_batch_carries_one_dataset_loss_multiplier(tmp_path):
    latent_path, text_encoder_path = _write_cache_pair(
        tmp_path, "sample", f"sample_0016x0016_{ARCHITECTURE_WAN}.safetensors"
    )
    item = ItemInfo("sample", "", (16, 16), (16, 16), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_encoder_path)
    manager = BucketBatchManager({(16, 16): [item]}, batch_size=1, loss_multiplier=3.0)

    assert manager[0]["loss_multiplier"] == 3.0


def test_trainer_scales_complete_loss_once_and_reports_multiplier():
    primary_loss = torch.tensor(2.0, requires_grad=True)
    auxiliary_loss = torch.tensor(3.0, requires_grad=True)

    scaled_loss, multiplier = NetworkTrainer.apply_dataset_loss_multiplier(
        primary_loss + auxiliary_loss, {"loss_multiplier": 4.0}
    )
    scaled_loss.backward()

    assert scaled_loss.item() == 20.0
    assert multiplier == 4.0
    assert primary_loss.grad.item() == 4.0
    assert auxiliary_loss.grad.item() == 4.0


def test_trainer_default_multiplier_preserves_loss_and_gradient():
    loss = torch.tensor(7.0, requires_grad=True)

    scaled_loss, multiplier = NetworkTrainer.apply_dataset_loss_multiplier(loss, {})
    scaled_loss.backward()

    assert scaled_loss.item() == 7.0
    assert multiplier == 1.0
    assert loss.grad.item() == 1.0


def test_trainer_multiplier_preserves_bfloat16_loss_and_gradient_dtype():
    loss = torch.tensor(2.0, dtype=torch.bfloat16, requires_grad=True)

    scaled_loss, _ = NetworkTrainer.apply_dataset_loss_multiplier(loss, {"loss_multiplier": 1.5})
    scaled_loss.backward()

    assert scaled_loss.dtype == torch.bfloat16
    assert loss.grad.dtype == torch.bfloat16
    assert scaled_loss.item() == 3.0
