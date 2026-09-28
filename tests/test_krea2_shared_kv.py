from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import musubi_tuner.krea2.krea2_mmdit as mmdit
from musubi_tuner.krea2.alpha_token_mask import (
    align_alpha_mask_to_token_grid,
    make_alpha_token_keep_mask,
    make_alpha_token_probabilities,
    make_shared_kv_keep_mask,
)
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2.shared_kv import make_shared_kv_plan
from musubi_tuner.krea2_train_network import (
    Krea2NetworkTrainer,
    validate_alpha_masked_attention_args,
)


def _cpu_plan_attention(q, k, v, plan):
    """Small differentiable reference for plan semantics; not a production backend."""
    batch_size, capacity, _, _ = q.shape
    order = plan.order.reshape(batch_size, capacity)
    rows = []
    for index in range(batch_size):
        q_length = int(plan.cu_seqlens_q[2 * index + 1] - index * capacity)
        k_length = int(plan.cu_seqlens_k[2 * index + 1] - index * capacity)
        key_indices = order[index, :k_length] - index * capacity
        qi = q[index, :q_length].transpose(0, 1).unsqueeze(0)
        ki = k[index, key_indices].transpose(0, 1).unsqueeze(0)
        vi = v[index, key_indices].transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(qi, ki, vi).squeeze(0).transpose(0, 1)
        rows.append(F.pad(output, (0, 0, 0, 0, 0, capacity - q_length)))
    return torch.stack(rows)


def test_token_mean_and_shared_threshold_endpoints_are_nested():
    values = torch.tensor([0, 1 / 255, 64 / 255, 1, 0.25, 0.25, 0.75, 1], dtype=torch.float32)
    alpha = values.reshape(1, 2, 4, 1, 1).expand(-1, -1, -1, 16, 16)
    alpha = alpha.permute(0, 1, 3, 2, 4).reshape(1, 32, 64)
    aligned = align_alpha_mask_to_token_grid(alpha, (4, 8), patch=2)
    probabilities = make_alpha_token_probabilities(aligned, (4, 8), patch=2, device=torch.device("cpu"))
    torch.testing.assert_close(probabilities, values[None])
    assert make_alpha_token_keep_mask(aligned, (4, 8), 2, torch.device("cpu")).tolist() == [
        [False, True, True, True, True, True, True, True]
    ]

    probabilities = probabilities.expand(2, -1)
    retained = make_shared_kv_keep_mask(probabilities, torch.tensor([0.1, 0.5]))
    assert retained[0].tolist() == [False, False, True, True, True, True, True, True]
    assert retained[1].tolist() == [False, False, False, True, False, False, True, True]
    assert torch.all(retained[1] <= retained[0])
    assert retained[:, 0].tolist() == [False, False]
    assert retained[:, 3].tolist() == [True, True]
    assert retained[0, 4] == retained[0, 5]  # equal-alpha region shares one decision


def test_shared_kv_plan_has_fixed_storage_and_expected_gradients():
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    keep_a = torch.tensor([[True, False, True, True], [False, True, True, False]])
    keep_b = valid.clone()
    plan_a = make_shared_kv_plan(valid, keep_a)
    plan_b = make_shared_kv_plan(valid, keep_b)
    assert plan_a.order.shape == plan_b.order.shape == (8,)
    assert plan_a.cu_seqlens_q.shape == plan_b.cu_seqlens_q.shape == (5,)
    assert plan_a.cu_seqlens_k.shape == plan_b.cu_seqlens_k.shape == (5,)
    assert plan_a.order.reshape(2, 4)[0].tolist() == [0, 2, 3, 1]
    assert plan_b.order.tolist() == list(range(8))
    assert torch.equal(plan_b.cu_seqlens_q, plan_b.cu_seqlens_k)

    torch.manual_seed(9)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 4, 2, 4, requires_grad=True)
    v = torch.randn(1, 4, 2, 4, requires_grad=True)
    plan = make_shared_kv_plan(torch.ones(1, 4, dtype=torch.bool), keep_a[:1])
    output = _cpu_plan_attention(q, k, v, plan)
    changed_k = k.detach().clone()
    changed_v = v.detach().clone()
    changed_k[0, 1].add_(1000)
    changed_v[0, 1].sub_(1000)
    torch.testing.assert_close(output.detach(), _cpu_plan_attention(q.detach(), changed_k, changed_v, plan))
    output.square().sum().backward()
    assert torch.count_nonzero(k.grad[0, 1]) == 0
    assert torch.count_nonzero(v.grad[0, 1]) == 0
    assert torch.count_nonzero(q.grad[0, 1]) > 0  # its Q and own output still participate


def _tiny_model(attn_mode="flash"):
    config = SingleMMDiTConfig(
        features=32, tdim=16, txtdim=32, heads=2, kvheads=2, multiplier=1,
        layers=2, patch=2, channels=2, txtlayers=2, txtheads=2, txtkvheads=2,
    )
    return SingleStreamDiT(config, attn_mode=attn_mode).float().train()


def test_one_plan_is_reused_by_all_blocks_and_checkpoint_recomputation(monkeypatch):
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    build_count = 0
    observed_plans = []
    real_make_plan = mmdit.make_shared_kv_plan

    def record_make_plan(valid_q, keep_kv):
        nonlocal build_count
        build_count += 1
        return real_make_plan(valid_q, keep_kv)

    def text_attention(qkv, **_kwargs):
        q = qkv[0]
        return q.reshape(q.shape[0], q.shape[1], -1)

    def shared_attention(q, _k, _v, plan):
        observed_plans.append(plan)
        return q

    monkeypatch.setattr(mmdit, "make_shared_kv_plan", record_make_plan)
    monkeypatch.setattr(mmdit, "common_attention", text_attention)
    monkeypatch.setattr(mmdit, "shared_kv_flash_attention", shared_attention)

    torch.manual_seed(17)
    img = torch.randn(2, 4, 8, requires_grad=True)
    context = torch.randn(2, 3, 2, 32, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    image_mask = torch.tensor([[True, False, True, True], [True, True, True, True]])
    probabilities = torch.tensor([[1.0, 0.0, 0.25, 1.0], [1.0, 0.75, 0.25, 1.0]])
    pos = torch.zeros(2, 7, 3)
    output = model(
        img=img, context=context, t=torch.tensor([0.2, 0.7]), pos=pos, mask=mask,
        image_mask=image_mask, image_kv_probabilities=probabilities,
        shared_kv_uniforms=torch.tensor([0.5, 0.5]),
    )
    output.square().sum().backward()
    assert build_count == 1
    assert len(observed_plans) == 2 * len(model.blocks)
    assert len({id(plan) for plan in observed_plans}) == 1
    plan = observed_plans[0]
    retained_counts = plan.cu_seqlens_k[1::2] - plan.cu_seqlens_k[:-1:2]
    assert retained_counts.tolist() == [4, 6]  # retained image K/V plus every valid text token
    assert torch.count_nonzero(output[0, 1]) == 0  # physical alpha=0 scatter remains zero


def test_trainer_sharedkv_prepares_mean_probabilities_and_one_uniform():
    class Model:
        config = SimpleNamespace(patch=2)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["img"]

    model = Model()
    args = SimpleNamespace(
        alpha_masked_token_drop=True, alpha_masked_attention_mode="sharedkv", gradient_checkpointing=False
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    latents = torch.zeros(1, 2, 1, 4, 4)
    alpha = torch.ones(1, 32, 32)
    alpha[:, :16, :16] = 0
    alpha[:, :16, 16:] = 0.25
    batch = {"latents": latents, "alpha_mask": alpha, "krea2_vl_embed": [torch.zeros(2, 2, 32)]}
    Krea2NetworkTrainer.call_dit(
        None, args, accelerator, model, latents, batch, torch.ones_like(latents),
        latents, torch.tensor([500.0]), torch.float32,
    )
    assert model.kwargs["image_mask"].tolist() == [[False, True, True, True]]
    torch.testing.assert_close(
        model.kwargs["image_kv_probabilities"], torch.tensor([[0.0, 0.25, 1.0, 1.0]])
    )
    uniforms = model.kwargs["shared_kv_uniforms"]
    assert uniforms.shape == (1,) and 0 <= uniforms.item() < 1

    opaque_model = Model()
    opaque_batch = {"latents": latents, "krea2_vl_embed": [torch.zeros(2, 2, 32)]}
    Krea2NetworkTrainer.call_dit(
        None, args, accelerator, opaque_model, latents, opaque_batch, torch.ones_like(latents),
        latents, torch.tensor([500.0]), torch.float32,
    )
    assert opaque_model.kwargs["image_mask"].all()
    assert torch.all(opaque_model.kwargs["image_kv_probabilities"] == 1)


def test_sharedkv_cli_requirements_and_native_default():
    base = {
        "alpha_masked_attention_mode": "sharedkv",
        "alpha_masked_token_drop": True,
        "flash_attn": True,
        "sdpa": False,
        "split_attn": False,
    }
    validate_alpha_masked_attention_args(SimpleNamespace(**base))
    for changes, message in (
        ({"alpha_masked_token_drop": False}, "alpha_masked_token_drop"),
        ({"flash_attn": False}, "FlashAttention"),
        ({"sdpa": True}, "FlashAttention"),
        ({"split_attn": True}, "split_attn"),
    ):
        values = base | changes
        with pytest.raises(ValueError, match=message):
            validate_alpha_masked_attention_args(SimpleNamespace(**values))
    validate_alpha_masked_attention_args(SimpleNamespace(alpha_masked_attention_mode="native"))
    validate_alpha_masked_attention_args(SimpleNamespace())
