from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ImageDataset
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2.mixed_token_batch import process_mixed_token_batch
from musubi_tuner.krea2.token_bucketing import Krea2TokenBucketBatchManager, cache_token_info
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
from musubi_tuner.training.trainer_base import NetworkTrainer


def test_cross_resolution_groups_carry_every_item_once_and_rebuild_deterministically(tmp_path):
    buckets = {(1024, 768): [], (768, 1024): []}
    for i in range(8):
        width = 514 if i < 5 else 512  # 257 or 256 image tokens; two groups of 5 and 3
        latent_path = tmp_path / f"latent_{i}.safetensors"
        text_path = tmp_path / f"text_{i}.safetensors"
        save_file({f"latents_1x2x{width}_float32": torch.zeros(2, 1, 2, width)}, latent_path)
        save_file({"varlen_krea2_vl_embed_float32": torch.zeros(i + 1, 2, 32)}, text_path)
        resolution = list(buckets)[i % 2]
        buckets[resolution].append(SimpleNamespace(
            item_key=str(i), bucket_size=resolution, latent_cache_path=str(latent_path),
            text_encoder_output_cache_path=str(text_path)
        ))
    manager = Krea2TokenBucketBatchManager(
        buckets, 4, multiple=1, drop_alpha_tokens=False, caption_selection_seed=123,
        num_timestep_buckets=3
    )
    assert manager.group_counts == {0: 3, 1: 5}
    assert sorted(item.item_key for batch in manager.batch_items for item in batch) == [str(i) for i in range(8)]
    assert len(manager.batch_items) == 2 and all(len(batch) == 4 for batch in manager.batch_items)
    assert len(manager.timestep_pool) == 2 and all(len(row) == 4 for row in manager.timestep_pool)
    batch = manager[0]
    assert len(batch["latents"]) == 4 and len(batch["krea2_vl_embed"]) == 4
    assert all(alpha is None for alpha in batch["alpha_mask"])

    manager.set_current_epoch(2)
    manager.shuffle()
    first = [[item.item_key for item in batch] for batch in manager.batch_items]
    manager.shuffle()
    assert first == [[item.item_key for item in batch] for batch in manager.batch_items]

    wider = Krea2TokenBucketBatchManager(buckets, 4, multiple=2, drop_alpha_tokens=False,
                                          caption_selection_seed=123)
    assert wider.group_counts == {0: 8}  # 256 and 257 share a 512-wide group


def test_alpha_keep_count_uses_training_snap16_and_remainder_assert(tmp_path):
    latent_path = tmp_path / "one.safetensors"
    alpha = torch.ones(32, 32)
    alpha[15, 15] = 0
    alpha[:16, 16:] = 1 / 255
    save_file({"latents_1x4x4_float32": torch.zeros(2, 1, 4, 4), "alpha_mask": alpha}, latent_path)
    assert cache_token_info(str(latent_path), True) == ((4, 4), 3)
    assert cache_token_info(str(latent_path), False) == ((4, 4), 4)
    item = SimpleNamespace(item_key="one", bucket_size=(32, 32), latent_cache_path=str(latent_path))
    with pytest.raises(AssertionError, match="leave 1 unbatched items"):
        Krea2TokenBucketBatchManager({(32, 32): [item]}, 2, multiple=1, drop_alpha_tokens=True)


def test_multi_caption_and_dropout_use_selected_text_after_grouping(tmp_path):
    items = []
    for i in range(2):
        latent_path = tmp_path / f"{i}.safetensors"
        text_path = tmp_path / f"{i}_te.safetensors"
        save_file({"latents_1x4x4_float32": torch.zeros(2, 1, 4, 4)}, latent_path)
        save_file({"varlen_krea2_vl_embed_caption_0000_float32": torch.zeros(3, 2, 32),
                   "varlen_krea2_vl_embed_caption_0001_float32": torch.ones(5, 2, 32)}, text_path)
        items.append(SimpleNamespace(item_key=str(i), bucket_size=(32, 32),
                                     latent_cache_path=str(latent_path), text_encoder_output_cache_path=str(text_path)))
    manager = Krea2TokenBucketBatchManager({(32, 32): items}, 2, multiple=1,
                                           drop_alpha_tokens=False, caption_selection_seed=7)
    manager.set_current_epoch(1)
    manager.shuffle()
    text = manager[0]["krea2_vl_embed"]
    assert len(text) == 2 and all(x.shape[0] in (3, 5) for x in text)

    manager.caption_dropout_rate = 1.0
    manager.set_caption_dropout_embedding(torch.full((2, 2, 32), 7.0))
    dropped = manager[0]["krea2_vl_embed"]
    assert all(x.shape == (2, 2, 32) and torch.all(x == 7) for x in dropped)


def test_replacing_partial_resolution_batches_updates_concat_dataset_length(tmp_path, monkeypatch):
    buckets = {}
    for i, resolution in enumerate(((32, 32), (32, 48))):
        path = tmp_path / f"{i}.safetensors"
        save_file({"latents_1x4x4_float32": torch.zeros(2, 1, 4, 4)}, path)
        buckets[resolution] = [SimpleNamespace(item_key=str(i), bucket_size=resolution,
                                               latent_cache_path=str(path))]
    dataset = object.__new__(ImageDataset)
    dataset.architecture = ARCHITECTURE_KREA2
    dataset.batch_size = 2
    dataset.seed = 11
    dataset.caption_dropout_rate = 0.0
    dataset.loss_multiplier = 1.0
    dataset.num_train_items = 2
    dataset.batch_manager = BucketBatchManager(buckets, 2)
    group = DatasetGroup([dataset])
    assert len(group) == 2
    monkeypatch.setattr(NetworkTrainer, "_build_dataset", lambda self, args: (group, None, None))
    args = SimpleNamespace(image_token_bucketing=True, image_token_bucket_multiple=1,
                           alpha_masked_token_drop=False, num_timestep_buckets=None, dry_bucketing=False)
    rebuilt, _, _ = Krea2NetworkTrainer()._build_dataset(args)
    assert len(rebuilt) == 1
    assert rebuilt.cumulative_sizes == [1]


@pytest.mark.parametrize("drop_alpha", [False, True])
@pytest.mark.parametrize("all_zero_alpha", [False, True])
def test_mixed_geometry_forward_matches_serial_loss_and_gradients(drop_alpha, all_zero_alpha):
    torch.manual_seed(51)
    config = SingleMMDiTConfig(features=32, tdim=16, txtdim=32, heads=2, kvheads=2,
                               multiplier=1, layers=1, patch=2, channels=2,
                               txtlayers=2, txtheads=2, txtkvheads=2)
    model = SingleStreamDiT(config, attn_mode="torch").float().train()
    trainer = Krea2NetworkTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(alpha_masked_token_drop=drop_alpha, gradient_checkpointing=False,
                           weighting_scheme="none", timestep_sampling="krea2_shift", sigmoid_scale=1.0,
                           min_timestep=None, max_timestep=None, preserve_distribution_shape=False)
    latents = [torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 6)]
    noise = [torch.randn_like(x) for x in latents]
    alpha = torch.ones(32, 32)
    alpha[15, 15] = 0
    alpha[16:, :16] = 1 / 255
    if all_zero_alpha:
        alpha.zero_()
    batch = {"alpha_mask": [alpha, None], "krea2_vl_embed": [torch.randn(3, 2, 32), torch.randn(5, 2, 32)],
             "timesteps": [0.3, 0.6]}

    model.zero_grad(set_to_none=True)
    mixed_loss, _ = process_mixed_token_batch(
        trainer, args, accelerator, model, batch, latents, noise, None, torch.float32, torch.float32
    )
    mixed_loss.backward()
    mixed_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    model.zero_grad(set_to_none=True)
    serial_losses = []
    for i in range(2):
        single = {key: [values[i]] for key, values in batch.items()}
        loss, _ = process_mixed_token_batch(
            trainer, args, accelerator, model, single, [latents[i]], [noise[i]],
            None, torch.float32, torch.float32
        )
        serial_losses.append(loss)
    serial_loss = torch.stack(serial_losses).mean()
    serial_loss.backward()
    torch.testing.assert_close(mixed_loss, serial_loss, rtol=2e-5, atol=2e-5)
    assert mixed_grads.keys() == {name for name, param in model.named_parameters() if param.grad is not None}
    for name, param in model.named_parameters():
        if name in mixed_grads:
            torch.testing.assert_close(mixed_grads[name], param.grad, rtol=5e-5, atol=5e-5)


def test_uniform_geometry_matches_existing_krea2_loss():
    torch.manual_seed(93)
    config = SingleMMDiTConfig(features=32, tdim=16, txtdim=32, heads=2, kvheads=2,
                               multiplier=1, layers=1, patch=2, channels=2,
                               txtlayers=2, txtheads=2, txtkvheads=2)
    model = SingleStreamDiT(config, attn_mode="torch").float().train()
    trainer = Krea2NetworkTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(alpha_masked_token_drop=False, gradient_checkpointing=False,
                           weighting_scheme="none", timestep_sampling="uniform",
                           min_timestep=None, max_timestep=None, preserve_distribution_shape=False)
    latents = torch.randn(2, 2, 1, 4, 4)
    noise = torch.randn_like(latents)
    text = [torch.randn(3, 2, 32), torch.randn(5, 2, 32)]
    times = [0.2, 0.8]
    old_batch = {"latents": latents, "krea2_vl_embed": text, "timesteps": times}
    noisy, timestep = trainer.get_noisy_model_input_and_timesteps(
        args, noise, latents, times, None, torch.device("cpu"), torch.float32
    )
    old_output = trainer.call_dit(args, accelerator, model, latents, old_batch, noise,
                                  noisy, timestep, torch.float32)
    old_loss, _ = trainer.compute_loss(args, old_output, timestep, None,
                                       torch.float32, torch.float32, 0, old_batch)
    new_batch = {"alpha_mask": [None, None], "krea2_vl_embed": text, "timesteps": times}
    new_loss, _ = process_mixed_token_batch(
        trainer, args, accelerator, model, new_batch, list(latents), list(noise),
        None, torch.float32, torch.float32
    )
    torch.testing.assert_close(new_loss, old_loss, rtol=2e-5, atol=2e-5)
