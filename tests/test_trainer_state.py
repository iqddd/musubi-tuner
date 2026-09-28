import os
from types import SimpleNamespace

import pytest

from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils import train_utils


def training_progress(**overrides):
    values = {
        "global_step": 270,
        "completed_epochs": 10,
        "step_in_epoch": 0,
        "num_batches_per_epoch": 54,
        "gradient_accumulation_steps": 2,
    }
    values.update(overrides)
    return train_utils.TrainingProgress(**values)


def test_training_progress_round_trip_and_resume_position(tmp_path):
    state = training_progress()
    train_utils.save_training_progress(tmp_path, state)
    loaded = train_utils.load_training_progress(tmp_path)

    assert loaded == state
    assert loaded.resolve_resume_position(
        num_batches_per_epoch=54,
        gradient_accumulation_steps=2,
        max_train_steps=486,
    ) == (10, 0)


def test_training_progress_resumes_inside_epoch_and_normalizes_epoch_end():
    partial = training_progress(global_step=274, step_in_epoch=8)
    assert partial.resolve_resume_position(
        num_batches_per_epoch=54,
        gradient_accumulation_steps=2,
        max_train_steps=486,
    ) == (10, 8)

    epoch_end = training_progress(global_step=297, step_in_epoch=54)
    assert epoch_end.resolve_resume_position(
        num_batches_per_epoch=54,
        gradient_accumulation_steps=2,
        max_train_steps=486,
    ) == (11, 0)


@pytest.mark.parametrize(
    "state,kwargs,error",
    [
        (training_progress(), {"num_batches_per_epoch": 55}, "current dataloader"),
        (training_progress(), {"gradient_accumulation_steps": 1}, "current value"),
        (training_progress(global_step=269), {}, "inconsistent"),
        (training_progress(), {"max_train_steps": 269}, "exceeds max_train_steps"),
    ],
)
def test_training_progress_rejects_incompatible_resume(state, kwargs, error):
    current = {
        "num_batches_per_epoch": 54,
        "gradient_accumulation_steps": 2,
        "max_train_steps": 486,
    }
    current.update(kwargs)
    with pytest.raises(ValueError, match=error):
        state.resolve_resume_position(**current)


def test_network_trainer_requires_and_loads_position_before_accelerate_state(tmp_path):
    trainer = NetworkTrainer()
    args = SimpleNamespace(resume=str(tmp_path), resume_from_huggingface=False)
    accelerator = SimpleNamespace(load_state=lambda path: loaded_paths.append(path))
    loaded_paths = []

    with pytest.raises(FileNotFoundError, match="training_progress.json"):
        trainer.resume_from_local_or_hf_if_specified(accelerator, args)
    assert loaded_paths == []

    train_utils.save_training_progress(tmp_path, training_progress())
    assert trainer.resume_from_local_or_hf_if_specified(accelerator, args)
    assert trainer._resume_training_progress == training_progress()
    assert loaded_paths == [str(tmp_path)]


def test_epoch_checkpoint_writes_training_progress(tmp_path):
    state = training_progress()
    args = SimpleNamespace(
        output_name="example",
        output_dir=str(tmp_path),
        save_state_to_huggingface=False,
        save_last_n_epochs_state=None,
        save_last_n_epochs=None,
        save_every_n_epochs=1,
    )

    class Accelerator:
        @staticmethod
        def save_state(path):
            os.makedirs(path, exist_ok=True)

    train_utils.save_and_remove_state_on_epoch_end(args, Accelerator(), 10, state)
    state_dir = tmp_path / "example-000010-state"
    assert train_utils.load_training_progress(state_dir) == state
