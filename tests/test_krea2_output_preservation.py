from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import musubi_tuner.krea2.krea2_mmdit as mmdit
from musubi_tuner.krea2.alpha_token_mask import align_alpha_mask_to_token_grid
from musubi_tuner.krea2.output_preservation import (
    base_model_teacher, branch_loss, prepare_branch, process_preservation_batch,
    validate_output_preservation_args,
)
from musubi_tuner.networks.lora_krea2 import create_arch_network
from test_krea2_log_bias import _cpu_log_bias_attention
from test_krea2_shared_kv import _cpu_plan_attention, _tiny_model


def model_and_network():
    torch.manual_seed(3)
    model = _tiny_model("torch")
    model.requires_grad_(False)
    network = create_arch_network(0.7, 2, 2, None, [], model)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    for lora in network.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.02)
    return model, network


def fixtures(mixed=False, alpha_kind="mixed"):
    latents = [torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 6 if mixed else 4)]
    noise = [torch.randn_like(x) for x in latents]
    masks = []
    for x in latents:
        mask = torch.ones(x.shape[-2] * 8, x.shape[-1] * 8)
        mask[:16, :16] = 0
        mask[16:, :16] = 64 / 255
        mask[0, 16] = 0  # a mixed 0/255 boundary cell must survive
        masks.append(mask)
    if alpha_kind == "zero":
        masks = [torch.zeros_like(x) for x in masks]
    if alpha_kind in ("one", "missing"):
        masks = [torch.ones_like(x) for x in masks]
    if not mixed:
        latents, noise, masks = torch.stack(latents), torch.stack(noise), torch.stack(masks)
    batch = dict(alpha_mask=None if alpha_kind == "missing" else masks,
                 krea2_vl_embed=[torch.randn(2, 2, 32), torch.randn(3, 2, 32)])
    return batch, latents, noise


class Trainer:
    def __init__(self):
        self.calls = 0

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, preset, *unused):
        self.calls += 1
        return latents * 0.25 + noise * 0.75, torch.full((latents.shape[0],), 750.)


def accelerator():
    return SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext, unwrap_model=lambda x: x)


def args(mode="native"):
    return SimpleNamespace(alpha_masked_attention_mode=mode, alpha_masked_attention_gamma=2.0,
                           weighting_scheme="none")


def test_mean_boundaries_and_complementary_loss_cells():
    alpha = torch.ones(1, 32, 32)
    alpha[:, :16, :16] = 0
    alpha[:, 16:24, :16] = 0
    aligned = align_alpha_mask_to_token_grid(alpha, (4, 4), 2)
    assert aligned[0, ::16, ::16].tolist() == [[0, 1], [0.5, 1]]
    assert torch.equal(aligned + (1 - aligned), torch.ones_like(aligned))


def test_teacher_restores_exact_state_even_on_exception_and_consumes_no_rng():
    model, network = model_and_network()
    network.unet_loras[0].multiplier = 0.3
    network.unet_loras[0].dropout = 0.5
    before = [(x.multiplier, x.training) for x in network.unet_loras]
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="intentional"):
        with base_model_teacher(network):
            assert not torch.is_grad_enabled()
            assert all(x.multiplier == 0 and not x.training for x in network.unet_loras)
            model.first(torch.ones(1, 3, 8))
            raise RuntimeError("intentional")
    assert before == [(x.multiplier, x.training) for x in network.unet_loras]
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["native", "sharedkv", "logbias"])
@pytest.mark.parametrize("mixed", [False, True])
def test_three_passes_reuse_plan_restore_lora_and_checkpoint_gradients(monkeypatch, mode, mixed):
    monkeypatch.setattr(mmdit, "shared_kv_flash_attention", _cpu_plan_attention)
    monkeypatch.setattr(mmdit, "log_bias_flash_attention", _cpu_log_bias_attention)
    model, network = model_and_network()
    model.enable_gradient_checkpointing()
    if mode != "native":
        model.attn_mode = "flash"

        def cpu_common(qkv, attn_params):
            q, k, v = [x.transpose(1, 2) for x in qkv]
            mask = None
            if attn_params.seqlens is not None:
                mask = torch.arange(k.shape[2])[None, :] < attn_params.seqlens[:, None]
                mask = mask[:, None, None, :]
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return out.transpose(1, 2).flatten(2)

        monkeypatch.setattr(mmdit, "common_attention", cpu_common)
    batch, latents, noise = fixtures(mixed)
    passes, block_plans = [], []

    def record_forward(module, unused, kw):
        passes.append((kw["packed_alpha_plan"], torch.is_grad_enabled(), network.unet_loras[0].multiplier))

    handle = model.register_forward_pre_hook(record_forward, with_kwargs=True)
    hooks = [block.register_forward_pre_hook(lambda module, inputs: block_plans.append((inputs[4], inputs[5])))
             for block in model.blocks]
    trainer = Trainer()
    state = torch.get_rng_state().clone()
    loss, metrics = process_preservation_batch(trainer, args(mode), accelerator(), model, network,
                                               batch, latents, noise, None, torch.float32, torch.float32)
    after_forward = torch.get_rng_state().clone()
    loss.backward()
    assert torch.equal(after_forward, torch.get_rng_state())
    if mode == "native" or mode == "logbias":
        assert torch.equal(state, after_forward)
    assert len(passes) == 3
    assert passes[0][0] is passes[2][0]
    assert passes[0][1:] == (False, 0)
    assert passes[1][1:] == passes[2][1:] == (True, 0.7)
    assert trainer.calls == (2 if mixed else 1)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert metrics["loss_preservation"] > 0
    torch.testing.assert_close(loss.detach(), metrics["loss_target"] + metrics["loss_preservation"])
    assert len(block_plans) == 5 * len(model.blocks)  # teacher, 2 students, 2 recomputations
    if mode != "native":
        slot = 0 if mode == "sharedkv" else 1
        expected = {id(passes[0][0].shared_kv if slot == 0 else passes[0][0].log_bias),
                    id(passes[1][0].shared_kv if slot == 0 else passes[1][0].log_bias)}
        assert {id(pair[slot]) for pair in block_plans} == expected
    handle.remove()
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("kind,passes", [("zero", 2), ("one", 1), ("missing", 1)])
def test_empty_branches_and_missing_alpha(kind, passes):
    model, network = model_and_network()
    batch, latents, noise = fixtures(alpha_kind=kind)
    seen = []
    hook = model.register_forward_pre_hook(lambda *unused: seen.append(1))
    loss, metrics = process_preservation_batch(Trainer(), args(), accelerator(), model, network,
                                               batch, latents, noise, None, torch.float32, torch.float32)
    loss.backward()
    assert len(seen) == passes
    assert metrics["loss_target" if kind == "zero" else "loss_preservation"] == 0
    hook.remove()


def test_formula_full_denominator_and_gradient_sum_including_empty_row():
    model, _ = model_and_network()
    images = [torch.randn(4, 8), torch.randn(6, 8)]
    positions = [torch.randn(x.shape[0], 3) for x in images]
    p = [torch.tensor([0., 0.25, 0.5, 1.]), torch.zeros(6)]
    context, text_mask = torch.randn(2, 3, 2, 32), torch.ones(2, 3, dtype=torch.bool)
    branch = prepare_branch(model, images, positions, p, context, text_mask, torch.ones(2), "native", None)
    pred = torch.randn(2, 3, 8, requires_grad=True)
    loss = branch_loss(pred, images, branch, [2., 3.], torch.float32)
    expected = ((pred[0] - images[0][1:]).square() * p[0][1:, None] * 2).sum() / (4 * 8) / 2
    torch.testing.assert_close(loss, expected)
    grad, = torch.autograd.grad(loss, pred, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, pred)
    torch.testing.assert_close(grad, expected_grad)
    assert not grad[1].any()
    selected_pos = branch.kwargs["packed_alpha_plan"].positions
    torch.testing.assert_close(selected_pos[0, :3], positions[0][1:])


def test_shared_complementary_decisions_and_text_endpoints():
    model, _ = model_and_network()
    images, positions = [torch.randn(5, 8)], [torch.randn(5, 3)]
    p = torch.tensor([0., 0.25, 0.25, 0.75, 1.])
    context, text_mask = torch.randn(1, 2, 2, 32), torch.ones(1, 2, dtype=torch.bool)
    plans = []
    for u in (0., 0.5, 0.9):
        assert torch.equal((u < p) | (u >= p), torch.ones(5, dtype=torch.bool))
        assert not ((u < p) & (u >= p)).any()
        for weights, keep in ((p, u < p), (1 - p, u >= p)):
            branch = prepare_branch(model, images, positions, [weights], context, text_mask,
                                    torch.ones(1), "sharedkv", None, [keep])
            plan = branch.kwargs["packed_alpha_plan"].shared_kv
            plans.append(plan)
            count = int(plan.cu_seqlens_k[1])
            assert count == int((keep & (weights > 0)).sum()) + 2
    assert len({tuple(plan.order.shape) for plan in plans}) == 1


@pytest.mark.parametrize("overrides,error", [
    ({"alpha_masked_token_drop": False}, "token_drop"),
    ({"blocks_to_swap": 1}, "blocks_to_swap"),
    ({"network_module": "custom"}, "standard"),
])
def test_cli_preservation_restrictions(overrides, error):
    values = dict(alpha_masked_output_preservation=True, alpha_masked_token_drop=True,
                  blocks_to_swap=0, network_module="networks.lora_krea2")
    values.update(overrides)
    with pytest.raises(ValueError, match=error):
        validate_output_preservation_args(SimpleNamespace(**values))


def test_preservation_cli_default():
    from argparse import ArgumentParser
    from musubi_tuner.krea2_train_network import krea2_setup_parser
    parser = krea2_setup_parser(ArgumentParser())
    assert parser.parse_args([]).alpha_masked_output_preservation is False


def test_bf16_autocast_teacher_does_not_detach_student_weights():
    model, network = model_and_network()
    batch, latents, noise = fixtures()
    acc = accelerator()
    acc.autocast = lambda: torch.autocast("cpu", dtype=torch.bfloat16)
    # Explicit outer autocast exercises the cache-clear safeguard too.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = process_preservation_batch(Trainer(), args(), acc, model, network,
                                             batch, latents, noise, None, torch.bfloat16, torch.float32)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    assert all(x.lora_up.weight.grad.abs().sum() > 0 for x in network.unet_loras)


def test_opaque_preservation_matches_existing_loss_and_gradients():
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
    model, network = model_and_network()
    batch, latents, noise = fixtures(alpha_kind="one")
    batch["latents"] = latents
    opts = args()
    opts.alpha_masked_token_drop = True
    opts.gradient_checkpointing = False
    loss, _ = process_preservation_batch(Trainer(), opts, accelerator(), model, network,
                                         batch, latents, noise, None, torch.float32, torch.float32)
    loss.backward()
    grads = [p.grad.clone() for p in network.parameters()]
    network.zero_grad(set_to_none=True)
    output = Krea2NetworkTrainer.call_dit(None, opts, accelerator(), model, latents, batch, noise,
                                         0.25 * latents + 0.75 * noise, torch.full((2,), 750.), torch.float32)
    expected = (output.pred - output.target).square().mean()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for grad, parameter in zip(grads, network.parameters()):
        torch.testing.assert_close(grad, parameter.grad, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_combined_backward_equals_sum_of_branch_gradients(checkpointing):
    model, network = model_and_network()
    if checkpointing:
        model.enable_gradient_checkpointing()
    batch, latents, noise = fixtures(mixed=True)
    import musubi_tuner.krea2.output_preservation as preservation_module
    from unittest.mock import patch
    parts = []
    real_loss = branch_loss

    def record(*a, **kw):
        loss = real_loss(*a, **kw)
        parts.append(loss)
        return loss

    with patch.object(preservation_module, "branch_loss", record):
        loss, _ = process_preservation_batch(Trainer(), args(), accelerator(), model, network,
                                             batch, latents, noise, None, torch.float32, torch.float32)
    parameters = tuple(network.parameters())
    grad_a = torch.autograd.grad(parts[0], parameters, retain_graph=True)
    grad_b = torch.autograd.grad(parts[1], parameters, retain_graph=True)
    loss.backward()
    for a, b, parameter in zip(grad_a, grad_b, parameters):
        torch.testing.assert_close(parameter.grad, a + b, atol=2e-7, rtol=2e-5)
