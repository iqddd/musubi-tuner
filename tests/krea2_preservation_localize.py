"""Boundary traces and matched-input block VJPs for the opt-in GPU diagnostic."""

import gc
from contextlib import contextmanager
from unittest.mock import patch

import torch
from krea2_preservation_gpu_probe import compare


@contextmanager
def boundary_trace(model, blocks):
    """Observe outside compiled/checkpointed functions, without changing their graph."""
    records = {}
    counts = {}
    checkpoint = torch.utils.checkpoint.checkpoint
    selected = {id(blocks[i]): f"block_{i}" for i in (0, len(blocks) - 1)}

    def capture(name, args, value):
        index = counts.get(name, 0)
        counts[name] = index + 1
        record = {"output": value.detach().cpu()}
        records[f"{name}/{index}"] = record
        if value.requires_grad:

            def gradient(grad):
                record["cotangent"] = grad.detach().cpu()

            value.register_hook(gradient)
        # Only block inputs are needed for replay. Plans are small immutable
        # objects; keep their device tensors, while offloading large activations.
        if name.startswith("block_"):
            record["args"] = tuple(a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args)
        return record

    def traced_checkpoint(function, *args, **kwargs):
        original = getattr(function, "_orig_mod", function)
        value = checkpoint(function, *args, **kwargs)
        if id(original) in selected:
            capture(selected[id(original)], args, value)
        return value

    def trace_text(module, args, value):
        capture("txtfusion", args, value)

    handle = model.txtfusion.register_forward_hook(trace_text)
    try:
        with patch.object(torch.utils.checkpoint, "checkpoint", traced_checkpoint):
            yield records
    finally:
        handle.remove()


def compare_traces(tested, reference):
    result = {}
    if tested.keys() != reference.keys():
        raise AssertionError("Boundary call counts differ")
    for name, record in reference.items():
        result[name] = {key: compare(tested[name][key], record[key]) for key in ("output", "cotangent") if key in record}
    return result


def block_vjp(block, network, index, record, *, compiled, backend="inductor"):
    """Use exactly the same eager inputs/cotangent for each independent replay."""
    network.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    args = [a.cuda() if isinstance(a, torch.Tensor) else a for a in record["args"]]
    args[0] = args[0].detach().requires_grad_(True)
    args[1] = args[1].detach().requires_grad_(True)
    parameters = {name: p for name, p in network.named_parameters() if name.startswith(f"lora_unet_blocks_{index}_")}
    assert parameters, f"No LoRA parameters for block {index}"
    function = torch.compile(block, backend=backend) if compiled else block
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = function(*args)
    names = ["input", "modulation", *parameters]
    gradients = torch.autograd.grad(prediction, [args[0], args[1], *parameters.values()], record["cotangent"].cuda())
    return {"output": prediction.detach().cpu(), **{n: g.detach().cpu() for n, g in zip(names, gradients)}}


def localize(model, network, blocks, run, report, save, *, backend="inductor"):
    with boundary_trace(model, blocks) as eager_trace:
        eager = run("trace_eager")
    del eager
    with boundary_trace(model, blocks) as compiled_trace:
        compiled = run("trace_compiled", compiled=True)
    del compiled
    report["boundary_comparisons"] = compare_traces(compiled_trace, eager_trace)
    del compiled_trace
    save()
    report["block_vjp"] = {}
    for index in (0, len(blocks) - 1):
        for branch in (0, 1):
            name = f"block_{index}/{branch}"
            print("VJP", name, flush=True)
            record = eager_trace[name]
            eager = block_vjp(blocks[index], network, index, record, compiled=False)
            compiled = block_vjp(blocks[index], network, index, record, compiled=True, backend=backend)
            report["block_vjp"][name] = {key: compare(compiled[key], eager[key]) for key in eager}
            del eager, compiled
            save()
