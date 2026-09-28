"""CPU checks for the opt-in GPU probe's numerical and checkpoint controls."""

import pytest
import torch
import torch.utils.checkpoint
from krea2_preservation_gpu_probe import compare
from krea2_preservation_gradient_diagnostic import bypass_block_checkpoint


def test_zero_gradient_cosine_is_undefined():
    both = compare(torch.zeros(8), torch.zeros(8))
    assert both["both_zero"] and not both["cosine_defined"]
    assert both["cosine"] is None and both["relative_l2"] == 0
    one = compare(torch.ones(8), torch.zeros(8))
    assert not one["both_zero"] and one["cosine"] is None
    assert one["reference_norm"] == 0 and one["tested_norm"] > 0


def test_metric_uses_double_reductions():
    vector = torch.tensor([1e20, 1e-20, -1e20])
    result = compare(vector, vector)
    assert result["relative_l2"] == 0
    assert result["cosine"] == pytest.approx(1)
    assert result["reference_norm"] == float(vector.double().norm())


def test_selective_checkpoint_recomputes_only_other_blocks_and_restores_on_error():
    blocks = [torch.nn.Linear(4, 4) for _ in range(3)]
    calls = [0, 0, 0]
    handles = []
    for index, block in enumerate(blocks):

        def count(module, args, index=index):
            calls[index] += 1

        handles.append(block.register_forward_pre_hook(count))
    original = torch.utils.checkpoint.checkpoint
    with pytest.raises(RuntimeError, match="intentional"), bypass_block_checkpoint(blocks[1]):
        value = torch.randn(2, 4, requires_grad=True)
        for block in blocks:
            value = torch.utils.checkpoint.checkpoint(block, value, use_reentrant=False)
        value.sum().backward()
        assert calls == [2, 1, 2]
        assert all(p.grad is not None for block in blocks for p in block.parameters())
        raise RuntimeError("intentional")
    assert torch.utils.checkpoint.checkpoint is original
    for handle in handles:
        handle.remove()


def test_boundary_trace_observes_without_changing_outputs_or_gradients():
    from krea2_preservation_localize import boundary_trace, compare_traces
    from test_krea2_output_preservation import Trainer, accelerator, args, fixtures, model_and_network

    from musubi_tuner.krea2.output_preservation import process_preservation_batch

    model, network = model_and_network()
    model.train().enable_gradient_checkpointing()
    batch, latents, noise = fixtures(mixed=True)

    def run():
        network.zero_grad(set_to_none=True)
        loss, _ = process_preservation_batch(
            Trainer(), args(), accelerator(), model, network, batch, latents, noise, None, torch.float32, torch.float32
        )
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in network.named_parameters()}

    expected, expected_grads = run()
    with boundary_trace(model, list(model.blocks)) as trace:
        actual, actual_grads = run()
    assert torch.equal(actual, expected)
    assert all(torch.equal(actual_grads[n], v) for n, v in expected_grads.items())
    assert "cotangent" not in trace["txtfusion/0"]  # teacher
    assert "cotangent" in trace["txtfusion/1"]
    assert "cotangent" in trace["block_0/0"]
    assert "cotangent" in trace["block_0/1"]
    assert all(v["output"]["relative_l2"] == 0 for v in compare_traces(trace, trace).values())


def test_deterministic_attention_covers_shared_kv_alias(monkeypatch):
    from krea2_preservation_gpu_probe import deterministic_flash_attention

    from musubi_tuner.krea2 import shared_kv
    from musubi_tuner.modules import attention

    def fake_attention(*args, **kwargs):
        return kwargs["deterministic"]

    monkeypatch.setattr(attention, "flash_attn_func", fake_attention)
    monkeypatch.setattr(attention, "flash_attn_varlen_func", fake_attention)
    monkeypatch.setattr(shared_kv, "flash_attn_varlen_func", fake_attention)
    deterministic_flash_attention()
    assert attention.flash_attn_func()
    assert attention.flash_attn_varlen_func()
    assert shared_kv.flash_attn_varlen_func()
