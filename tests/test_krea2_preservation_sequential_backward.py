"""Sequential preservation must retain combined-loss optimizer semantics."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_krea2_log_bias import _cpu_log_bias_attention
from test_krea2_output_preservation import Trainer, fixtures, model_and_network
from test_krea2_shared_kv import _cpu_plan_attention

from musubi_tuner.krea2 import krea2_mmdit as mmdit
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
from musubi_tuner.training.trainer_base import NetworkTrainer


class FixedTrainer(Krea2NetworkTrainer):
    get_noisy_model_input_and_timesteps = Trainer.get_noisy_model_input_and_timesteps

    def __init__(self):
        super().__init__()
        self.calls = 0


@pytest.mark.parametrize("mode", ["native", "sharedkv", "logbias"])
@pytest.mark.parametrize("bf16", [False, True])
def test_accumulated_gradients_and_adamw_updates_match_combined_loss(monkeypatch, mode, bf16):
    monkeypatch.setattr(mmdit, "shared_kv_flash_attention", _cpu_plan_attention)
    monkeypatch.setattr(mmdit, "log_bias_flash_attention", _cpu_log_bias_attention)
    if mode != "native":

        def cpu_attention(qkv, attn_params):
            q, k, v = [x.transpose(1, 2) for x in qkv]
            mask = None
            if attn_params.seqlens is not None:
                mask = torch.arange(k.shape[2])[None, :] < attn_params.seqlens[:, None]
                mask = mask[:, None, None, :]
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)

        monkeypatch.setattr(mmdit, "common_attention", cpu_attention)
    batch, latents, noise = fixtures(mixed=True)
    batch["loss_multiplier"] = 2.5
    args = SimpleNamespace(
        alpha_masked_output_preservation=True,
        alpha_masked_attention_mode=mode,
        alpha_masked_attention_gamma=3.0,
        weighting_scheme="none",
    )

    def run(sequential):
        model, network = model_and_network()
        if mode != "native":
            model.attn_mode = "flash"
        model.enable_gradient_checkpointing()
        trainer = FixedTrainer()
        optimizer = torch.optim.AdamW(network.parameters(), lr=2e-4)
        events, losses, gradients, rng_states = [], [], [], []
        backward_calls = 0

        def backward(loss):
            nonlocal backward_calls
            backward_calls += 1
            (loss / 2).backward()  # two microbatches per optimizer update
            events.append("backward")

        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            unwrap_model=lambda x: x,
            backward=backward,
            autocast=(lambda: torch.autocast("cpu", dtype=torch.bfloat16)) if bf16 else nullcontext,
        )
        model.register_forward_pre_hook(lambda *unused: events.append("student" if torch.is_grad_enabled() else "teacher"))
        for microstep in range(4):
            torch.manual_seed(400 + microstep)
            method = (
                trainer.process_batch_and_backward
                if sequential
                else (lambda *a: NetworkTrainer.process_batch_and_backward(trainer, *a))
            )
            # Exercise an enclosing autocast scope as well as the model's own scope.
            with accelerator.autocast():
                loss, metrics = method(
                    args,
                    accelerator,
                    model,
                    network,
                    batch,
                    latents,
                    noise,
                    None,
                    torch.bfloat16 if bf16 else torch.float32,
                    torch.float32,
                    None,
                    microstep // 2,
                )
            assert not loss.requires_grad
            torch.testing.assert_close(loss, 2.5 * (metrics["loss_target"] + metrics["loss_preservation"]))
            losses.append(loss)
            rng_states.append(torch.get_rng_state().clone())
            if microstep % 2:
                gradients.append([p.grad.clone() for p in network.parameters()])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        expected = (
            ["teacher", "student", "backward", "student", "backward"]
            if sequential
            else ["teacher", "student", "student", "backward"]
        )
        assert events == expected * 4
        assert backward_calls == (8 if sequential else 4)
        moments = [(v["exp_avg"].clone(), v["exp_avg_sq"].clone(), v["step"].clone()) for v in optimizer.state.values()]
        return losses, gradients, [p.detach().clone() for p in network.parameters()], moments, rng_states

    combined = run(False)
    sequential = run(True)
    torch.testing.assert_close(sequential[:4], combined[:4], rtol=2e-4, atol=2e-7)
    assert all(torch.equal(a, b) for a, b in zip(combined[4], sequential[4]))


@pytest.mark.parametrize("kind", ["zero", "one", "missing"])
def test_empty_branch_runs_only_one_backward(kind):
    trainer = FixedTrainer()
    model, network = model_and_network()
    batch, latents, noise = fixtures(alpha_kind=kind)
    calls = []

    def backward(loss):
        calls.append(loss.detach())
        loss.backward()

    accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda x: x, autocast=nullcontext, backward=backward)
    args = SimpleNamespace(alpha_masked_output_preservation=True, weighting_scheme="none")
    loss, _ = trainer.process_batch_and_backward(
        args,
        accelerator,
        model,
        network,
        batch,
        latents,
        noise,
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    assert len(calls) == 1
    assert torch.isfinite(loss) and not loss.requires_grad


def test_default_backward_hook_scales_once():
    trainer = NetworkTrainer()
    parameter = torch.tensor(3.0, requires_grad=True)
    trainer.process_batch = lambda *unused: (parameter.square(), {"example": 1.0})
    accelerator = SimpleNamespace(backward=lambda loss: (loss / 2).backward())
    loss, metrics = trainer.process_batch_and_backward(
        None,
        accelerator,
        None,
        None,
        {"loss_multiplier": 2.5},
        None,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    assert loss.item() == 22.5 and not loss.requires_grad
    assert parameter.grad.item() == 7.5
    assert metrics == {"example": 1.0}
