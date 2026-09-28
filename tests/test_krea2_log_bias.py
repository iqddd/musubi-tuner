from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import musubi_tuner.krea2.krea2_mmdit as mmdit
import musubi_tuner.krea2.log_bias as log_bias_module
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2.log_bias import make_log_bias_plan
from musubi_tuner.krea2.mixed_token_batch import process_mixed_token_batch
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, validate_alpha_masked_attention_args


def _cpu_log_bias_attention(q, k, v, plan):
    """Differentiable reference for fixed per-key log weights."""
    batch_size, capacity, _, head_dim = q.shape
    rows = []
    for index in range(batch_size):
        length = int(plan.valid_tokens[index].sum())
        qi = q[index, :length].transpose(0, 1)
        ki = k[index, :length].transpose(0, 1)
        vi = v[index, :length].transpose(0, 1)
        scores = torch.einsum("hqd,hkd->hqk", qi, ki) * (head_dim**-0.5)
        scores = scores + plan.log_key_weights[index, :length][None, None]
        output = torch.einsum("hqk,hkd->hqd", scores.softmax(dim=-1), vi).transpose(0, 1)
        rows.append(F.pad(output, (0, 0, 0, 0, 0, capacity - length)))
    return torch.stack(rows)


def test_log_bias_plan_formula_endpoints_and_monotonic_suppression():
    valid = torch.tensor([[True, True, True, False]])
    probabilities = torch.tensor([[0.25, 1.0, 1.0, 0.0]])
    plan = make_log_bias_plan(valid, probabilities, gamma=2.0)
    torch.testing.assert_close(
        plan.log_key_weights,
        torch.tensor([[2 * torch.log(torch.tensor(0.25)), 0.0, 0.0, 0.0]]),
    )
    assert plan.cu_seqlens.tolist() == [0, 3, 4]

    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]], [[99.0, 99.0]]]])
    weak = _cpu_log_bias_attention(q, k, v, make_log_bias_plan(valid, probabilities, gamma=1.0))
    strong = _cpu_log_bias_attention(q, k, v, plan)
    assert 0 < strong[0, 0, 0, 0] < weak[0, 0, 0, 0]
    assert torch.count_nonzero(strong[:, 3]) == 0

    opaque = torch.ones_like(probabilities)
    opaque[:, 3] = 0
    native_plan = make_log_bias_plan(valid, opaque, gamma=3.0)
    native = _cpu_log_bias_attention(q, k, v, native_plan)
    torch.testing.assert_close(native[0, :3, 0, 0], torch.full((3,), 1 / 3))


def test_log_bias_keeps_positive_token_q_and_outgoing_gradients():
    torch.manual_seed(4)
    valid = torch.tensor([[True, True, True, False]])
    probabilities = torch.tensor([[1 / 255, 0.25, 1.0, 0.0]])
    plan = make_log_bias_plan(valid, probabilities, gamma=1.0)
    q = torch.randn(1, 4, 2, 4, requires_grad=True)
    k = torch.randn(1, 4, 2, 4, requires_grad=True)
    v = torch.randn(1, 4, 2, 4, requires_grad=True)
    output = _cpu_log_bias_attention(q, k, v, plan)
    output.square().sum().backward()
    assert torch.count_nonzero(q.grad[0, 0]) > 0
    assert torch.count_nonzero(k.grad[0, 0]) > 0
    assert torch.count_nonzero(v.grad[0, 0]) > 0
    assert torch.count_nonzero(q.grad[0, 3]) == 0
    assert torch.count_nonzero(k.grad[0, 3]) == 0
    assert torch.count_nonzero(v.grad[0, 3]) == 0


def _tiny_model():
    config = SingleMMDiTConfig(
        features=32,
        tdim=16,
        txtdim=32,
        heads=2,
        kvheads=2,
        multiplier=1,
        layers=2,
        patch=2,
        channels=2,
        txtlayers=2,
        txtheads=2,
        txtkvheads=2,
    )
    return SingleStreamDiT(config, attn_mode="flash").float().train()


def test_one_log_bias_plan_is_reused_by_blocks_and_checkpointing(monkeypatch):
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    build_count = 0
    observed_plans = []
    real_make_plan = mmdit.make_log_bias_plan

    def record_make_plan(valid_tokens, probabilities, gamma):
        nonlocal build_count
        build_count += 1
        return real_make_plan(valid_tokens, probabilities, gamma)

    def text_attention(qkv, **_kwargs):
        q = qkv[0]
        return q.reshape(q.shape[0], q.shape[1], -1)

    def log_bias_attention(q, _k, _v, plan):
        observed_plans.append(plan)
        return q

    monkeypatch.setattr(mmdit, "make_log_bias_plan", record_make_plan)
    monkeypatch.setattr(mmdit, "common_attention", text_attention)
    monkeypatch.setattr(mmdit, "log_bias_flash_attention", log_bias_attention)

    torch.manual_seed(17)
    img = torch.randn(2, 4, 8, requires_grad=True)
    context = torch.randn(2, 3, 2, 32, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    image_mask = torch.tensor([[True, False, True, True], [True, True, True, True]])
    probabilities = torch.tensor([[1.0, 0.0, 0.25, 1.0], [1.0, 0.75, 0.25, 1.0]])
    output = model(
        img=img,
        context=context,
        t=torch.tensor([0.2, 0.7]),
        pos=torch.zeros(2, 7, 3),
        mask=mask,
        image_mask=image_mask,
        image_kv_probabilities=probabilities,
        log_bias_gamma=3.0,
    )
    output.square().sum().backward()
    assert build_count == 1
    assert len(observed_plans) == 2 * len(model.blocks)
    assert len({id(plan) for plan in observed_plans}) == 1
    plan = observed_plans[0]
    torch.testing.assert_close(plan.log_key_weights[0, :5], torch.tensor([0.0, 3 * torch.log(torch.tensor(0.25)), 0.0, 0.0, 0.0]))
    assert torch.count_nonzero(plan.log_key_weights[0, 5:]) == 0
    assert torch.count_nonzero(output[0, 1]) == 0


def test_trainer_logbias_prepares_probabilities_without_rng():
    class Model:
        config = SimpleNamespace(patch=2)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["img"]

    args = SimpleNamespace(
        alpha_masked_token_drop=True,
        alpha_masked_attention_mode="logbias",
        alpha_masked_attention_gamma=3.0,
        gradient_checkpointing=False,
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    latents = torch.zeros(1, 2, 1, 4, 4)
    alpha = torch.ones(1, 32, 32)
    alpha[:, :16, :16] = 0
    alpha[:, :16, 16:] = 0.25
    batch = {"latents": latents, "alpha_mask": alpha, "krea2_vl_embed": [torch.zeros(2, 2, 32)]}
    model = Model()
    rng_state = torch.get_rng_state().clone()
    Krea2NetworkTrainer.call_dit(
        None,
        args,
        accelerator,
        model,
        latents,
        batch,
        torch.ones_like(latents),
        latents,
        torch.tensor([500.0]),
        torch.float32,
    )
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert model.kwargs["image_mask"].tolist() == [[False, True, True, True]]
    torch.testing.assert_close(
        model.kwargs["image_kv_probabilities"], torch.tensor([[0.0, 0.25, 1.0, 1.0]])
    )
    assert model.kwargs["log_bias_gamma"] == 3.0
    assert "shared_kv_uniforms" not in model.kwargs

    opaque_model = Model()
    opaque_batch = {"latents": latents, "krea2_vl_embed": [torch.zeros(2, 2, 32)]}
    Krea2NetworkTrainer.call_dit(
        None,
        args,
        accelerator,
        opaque_model,
        latents,
        opaque_batch,
        torch.ones_like(latents),
        latents,
        torch.tensor([500.0]),
        torch.float32,
    )
    assert opaque_model.kwargs["image_mask"].all()
    assert torch.all(opaque_model.kwargs["image_kv_probabilities"] == 1)


def test_mixed_geometry_logbias_packs_selected_probabilities():
    class Model:
        config = SimpleNamespace(patch=2)

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.zeros_like(kwargs["img"])

    model = Model()
    trainer = Krea2NetworkTrainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(
        alpha_masked_token_drop=True,
        alpha_masked_attention_mode="logbias",
        alpha_masked_attention_gamma=3.0,
        gradient_checkpointing=False,
        weighting_scheme="none",
        timestep_sampling="uniform",
        sigmoid_scale=1.0,
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
    )
    latents = [torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 6)]
    noise = [torch.ones_like(latent) for latent in latents]
    alpha = torch.ones(32, 32)
    alpha[:16, :16] = 0
    alpha[:16, 16:] = 0.25
    batch = {
        "alpha_mask": [alpha, None],
        "krea2_vl_embed": [torch.zeros(3, 2, 32), torch.zeros(2, 2, 32)],
        "timesteps": [0.3, 0.6],
    }
    loss, _ = process_mixed_token_batch(
        trainer, args, accelerator, model, batch, latents, noise, None, torch.float32, torch.float32
    )
    assert torch.isfinite(loss)
    assert model.kwargs["image_mask"].sum(dim=1).tolist() == [3, 6]
    assert model.kwargs["image_kv_probabilities"][0, :3].tolist() == [0.25, 1.0, 1.0]
    assert torch.all(model.kwargs["image_kv_probabilities"][1, :6] == 1)
    assert model.kwargs["log_bias_gamma"] == 3.0
    assert "shared_kv_uniforms" not in model.kwargs


def test_logbias_cli_requirements_and_optional_dependency(monkeypatch):
    base = {
        "alpha_masked_attention_mode": "logbias",
        "alpha_masked_attention_gamma": 3.0,
        "alpha_masked_token_drop": True,
        "flash_attn": True,
        "sdpa": False,
        "split_attn": False,
    }
    validate_alpha_masked_attention_args(SimpleNamespace(**base))
    for gamma in (None, 0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="gamma"):
            validate_alpha_masked_attention_args(SimpleNamespace(**(base | {"alpha_masked_attention_gamma": gamma})))
    for changes, message in (
        ({"alpha_masked_token_drop": False}, "alpha_masked_token_drop"),
        ({"flash_attn": False}, "FlashAttention"),
        ({"sdpa": True}, "FlashAttention"),
        ({"split_attn": True}, "split_attn"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_alpha_masked_attention_args(SimpleNamespace(**(base | changes)))
    for mode in ("native", "sharedkv"):
        with pytest.raises(ValueError, match="only"):
            validate_alpha_masked_attention_args(SimpleNamespace(**(base | {"alpha_masked_attention_mode": mode})))

    monkeypatch.setattr(log_bias_module, "_alpha_attention", None)
    monkeypatch.setattr(log_bias_module, "_FA2_ALPHA_IMPORT_ERROR", ImportError("missing test wheel"))
    monkeypatch.setattr(log_bias_module, "_FA2_ALPHA_IMPORT_ATTEMPTED", True)
    with pytest.raises(ImportError, match="compatible.*fa2-alpha"):
        validate_alpha_masked_attention_args(SimpleNamespace(**base))


def test_native_and_sharedkv_do_not_import_optional_logbias_backend(monkeypatch):
    monkeypatch.setattr(log_bias_module, "_alpha_attention", None)
    monkeypatch.setattr(log_bias_module, "_FA2_ALPHA_IMPORT_ERROR", None)
    monkeypatch.setattr(log_bias_module, "_FA2_ALPHA_IMPORT_ATTEMPTED", False)

    def unexpected_import(_name):
        raise AssertionError("native/sharedkv must not import fa2-alpha")

    monkeypatch.setattr(log_bias_module, "import_module", unexpected_import)
    validate_alpha_masked_attention_args(SimpleNamespace(alpha_masked_attention_mode="native"))
    validate_alpha_masked_attention_args(
        SimpleNamespace(
            alpha_masked_attention_mode="sharedkv",
            alpha_masked_token_drop=True,
            flash_attn=True,
            sdpa=False,
            split_attn=False,
        )
    )
    assert not log_bias_module._FA2_ALPHA_IMPORT_ATTEMPTED
