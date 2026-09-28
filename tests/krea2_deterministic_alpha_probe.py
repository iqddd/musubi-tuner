"""Opt-in validation of a separately built deterministic fa2-alpha extension.

Apply patches/fa2_alpha_deterministic.patch to the archived extension.cpp; build
with name=fa2_alpha_deterministic_cuda and a distinct FLASH_NAMESPACE. The
installed fa2-alpha package is only redirected inside this diagnostic process.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from krea2_preservation_gpu_probe import compare
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def install_deterministic_alpha(path):
    import fa2_alpha

    spec = importlib.util.spec_from_file_location("fa2_alpha_deterministic_cuda", path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)

    def backward(*args):
        return extension.backward(*args, True)

    fa2_alpha._extension = SimpleNamespace(forward=extension.forward, backward=backward)
    return fa2_alpha.alpha_attention


def main(cli):
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(420)
    operator = install_deterministic_alpha(cli.extension)
    results = []
    for length, heads, kv_heads, splits in (
        (128, 8, 8, [0, 128, 128, 255, 256]),
        (257, 48, 12, [0, 233, 257, 514, 514]),
        (512, 48, 12, [0, 1, 512, 1024, 1024]),
    ):
        q = torch.randn(2, length, heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k, v = [torch.randn(2, length, kv_heads, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        bias = torch.rand(2, length, device="cuda").clamp_min(1e-6).log() * 3
        cu = torch.tensor(splits, device="cuda", dtype=torch.int32)
        cotangent = torch.randn_like(q)
        ref_inputs = [x.detach().float().requires_grad_(True) for x in (q, k, v)]
        flat = [x.flatten(0, 1) for x in ref_inputs]
        parts = []
        with sdpa_kernel(SDPBackend.MATH):
            for start, end in zip(splits[:-1], splits[1:]):
                if start == end:
                    continue
                qs, ks, vs = [x[start:end].transpose(0, 1)[None] for x in flat]
                result = F.scaled_dot_product_attention(
                    qs, ks, vs, attn_mask=bias.flatten()[None, None, None, start:end], enable_gqa=True
                )
                parts.append(result[0].transpose(0, 1))
        reference = torch.cat(parts).reshape(q.shape)
        reference_grads = torch.autograd.grad(reference, ref_inputs, cotangent.float())

        def run(function, q=q, k=k, v=v, bias=bias, cu=cu, cotangent=cotangent):
            prediction = function(q, k, v, bias, None, cu)
            gradients = torch.autograd.grad(prediction, (q, k, v), cotangent)
            return prediction.detach(), gradients

        prediction, gradients = run(operator)
        repeat, repeat_grads = run(operator)
        row = {
            "length": length,
            "heads": heads,
            "kv_heads": kv_heads,
            "forward": compare(prediction, reference),
            "backward": [compare(a, b) for a, b in zip(gradients, reference_grads)],
            "repeat": [compare(a, b) for a, b in zip(repeat_grads, gradients)],
        }
        assert torch.equal(prediction, repeat)
        assert all(torch.equal(a, b) for a, b in zip(gradients, repeat_grads))
        assert row["forward"]["relative_l2"] <= 5e-3
        assert all(x["relative_l2"] <= 1e-2 and x["cosine"] >= 0.999 for x in row["backward"])
        if length == 257:

            def checkpointed(*args):
                return checkpoint(operator, *args, use_reentrant=False)

            compiled, compiled_grads = run(torch.compile(checkpointed, fullgraph=True))
            row["compiled_checkpoint"] = [compare(a, b) for a, b in zip(compiled_grads, gradients)]
            assert torch.equal(compiled, prediction)
            assert all(torch.equal(a, b) for a, b in zip(compiled_grads, gradients))
        results.append(row)
        (output / "report.json").write_text(json.dumps(results, indent=2))
        print(length, "PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
