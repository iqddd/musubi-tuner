"""Opt-in, no-optimizer isolation of preservation eager/compile gradient differences."""

import argparse
import gc
import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from krea2_preservation_gpu_probe import FixedInputs, compare, deterministic_flash_attention
from safetensors.torch import load_file

import musubi_tuner.krea2.output_preservation as preservation
from musubi_tuner.krea2.krea2_utils import load_krea2_dit
from musubi_tuner.krea2.mixed_token_batch import process_mixed_token_batch
from musubi_tuner.networks import lora_krea2


@contextmanager
def bypass_block_checkpoint(block):
    """Keep all other blocks checkpointed, including during recomputation."""
    checkpoint = torch.utils.checkpoint.checkpoint

    def selective(function, *args, **kwargs):
        if function is block or getattr(function, "_orig_mod", None) is block:
            kwargs.pop("use_reentrant", None)
            return function(*args, **kwargs)
        return checkpoint(function, *args, **kwargs)

    with patch.object(torch.utils.checkpoint, "checkpoint", selective):
        yield


def difference(left, right):
    names = sorted(left["gradients"].keys() & right["gradients"].keys())
    if not names:
        raise ValueError("No common gradients to compare")
    if len(left["predictions"]) != len(right["predictions"]):
        raise ValueError("Different prediction counts")
    a = torch.cat([left["gradients"][n].flatten() for n in names])
    b = torch.cat([right["gradients"][n].flatten() for n in names])
    by_group = {}
    for group in ("lora_up", "lora_down", "blocks", "txtfusion", "tmlp", "first"):
        matching = [n for n in names if (n.startswith("lora_unet_blocks_") if group == "blocks" else group in n)]
        if matching:
            by_group[group] = compare(
                torch.cat([left["gradients"][n].flatten() for n in matching]),
                torch.cat([right["gradients"][n].flatten() for n in matching]),
            )
    return {
        "gradient": compare(a, b),
        "reference_gradient_norm": float(b.double().norm()),
        "tested_gradient_norm": float(a.double().norm()),
        "groups": by_group,
        "missing_or_extra": sorted(left["gradients"].keys() ^ right["gradients"].keys()),
        "predictions": [compare(a, b) for a, b in zip(left["predictions"], right["predictions"])],
        "output_cotangents": [
            compare(a, b) if a is not None and b is not None else None
            for a, b in zip(left["output_cotangents"], right["output_cotangents"])
        ],
        "losses": [left["loss"], right["loss"]],
        "terms": [left["terms"], right["terms"]],
    }


def main(cli):
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "runs": {},
        "comparisons": {},
        "errors": {},
        "optimizer": None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json.tmp").write_text(json.dumps(report, indent=2))
        (output / "report.json.tmp").replace(output / "report.json")

    try:
        torch.set_num_threads(4)
        torch._inductor.config.compile_threads = 4
        torch._inductor.config.emulate_precision_casts = cli.emulate_precision_casts
        report["emulate_precision_casts"] = cli.emulate_precision_casts
        report["compile_backend"] = cli.compile_backend
        torch.manual_seed(420)
        # Large pinned-memory offload probes can take time to release their CUDA
        # context after termination. Do not overlap real-model allocations.
        deadline = time.monotonic() + 300
        while torch.cuda.mem_get_info()[0] < 28 * 1024**3:
            if time.monotonic() >= deadline:
                raise RuntimeError("GPU memory did not become available within 5 minutes")
            print("Waiting for at least 28 GiB free VRAM before loading", flush=True)
            time.sleep(10)
        report["deterministic_fa"] = cli.deterministic_fa
        if cli.deterministic_fa:
            deterministic_flash_attention()
        model = load_krea2_dit(cli.dit, device="cuda", dtype=torch.bfloat16, fp8_scaled=True, attn_mode="flash")
        model.requires_grad_(False)
        network = lora_krea2.create_arch_network(1.0, 32, 32, None, [], model)
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
        network.to("cuda").train().requires_grad_(True)
        for module in network.unet_loras:
            torch.nn.init.normal_(module.lora_up.weight, std=0.005)
        model.train().enable_gradient_checkpointing()
        blocks = list(model.blocks)
        compiled_blocks = None
        entries = json.loads(Path(cli.manifest).read_text())[:2]
        latents, masks, texts = [], [], []
        for entry in entries:
            sd, te = load_file(entry["latent"]), load_file(entry["text"])
            latents.append(next(v for k, v in sd.items() if k.startswith("latents_")).float().cuda())
            masks.append(sd["alpha_mask"].cuda())
            texts.append(next(v for k, v in sorted(te.items()) if k.startswith("varlen_krea2_vl_embed")).cuda())
        noise = [torch.randn_like(x) for x in latents]
        batch = {"alpha_mask": masks, "krea2_vl_embed": texts}
        report["inputs"] = entries
        report["latent_shapes"] = [list(x.shape) for x in latents]
        report["text_shapes"] = [list(x.shape) for x in texts]
        report["lora_initialization"] = "rank32, up normal std0.005 initially; separate zero-up control at end"
        acc = SimpleNamespace(
            device=torch.device("cuda"), autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16), unwrap_model=lambda x: x
        )
        real_loss = preservation.branch_loss
        observations = []
        output_cotangents = []
        cotangent_override = None

        def observe(module, inputs, kwargs, output):
            mask = kwargs["packed_alpha_plan"].image_mask if "packed_alpha_plan" in kwargs else kwargs["image_mask"]
            observations.append(output[mask].detach().float().cpu())
            index = len(output_cotangents)
            output_cotangents.append(None)
            if output.requires_grad:

                def capture_cotangent(grad):
                    if cotangent_override is not None:
                        grad = grad.clone()
                        grad[mask] = cotangent_override.to(device=grad.device, dtype=grad.dtype)
                    output_cotangents[index] = grad[mask].detach().float().cpu()
                    return grad

                output.register_hook(capture_cotangent)

        handle = model.register_forward_hook(observe, with_kwargs=True)

        def run(label, compiled=False, objective="total", matched_cotangent=None):
            nonlocal compiled_blocks, cotangent_override
            cotangent_override = matched_cotangent
            print("START", label, flush=True)
            before = time.monotonic()
            network.zero_grad(set_to_none=True)
            observations.clear()
            output_cotangents.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(420)
            model.gradient_checkpointing = True
            if compiled and compiled_blocks is None:
                compiled_blocks = [torch.compile(block, backend=cli.compile_backend) for block in blocks]
            model.blocks = torch.nn.ModuleList(compiled_blocks if compiled else blocks)
            parts = []

            def capture(*a, **kw):
                loss = real_loss(*a, **kw)
                parts.append(loss)
                return loss

            with patch.object(preservation, "branch_loss", capture):
                opts = SimpleNamespace(
                    alpha_masked_attention_mode="native",
                    alpha_masked_token_drop=True,
                    gradient_checkpointing=True,
                    alpha_masked_attention_gamma=None,
                    weighting_scheme="none",
                )
                if objective == "original":
                    loss, terms = process_mixed_token_batch(
                        FixedInputs(), opts, acc, model, batch, latents, noise, None, torch.bfloat16, torch.float32
                    )
                else:
                    loss, terms = preservation.process_preservation_batch(
                        FixedInputs(), opts, acc, model, network, batch, latents, noise, None, torch.bfloat16, torch.float32
                    )
            selected_loss = loss if objective in ("total", "original") else parts[0 if objective == "target" else 1]
            selected_loss.backward()
            grads = {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in network.named_parameters()
                if parameter.grad is not None
            }
            assert len(grads) == len(list(network.parameters())), "Missing LoRA gradients"
            assert all(torch.isfinite(v).all() for v in [loss, *grads.values(), *observations])
            result = {
                "loss": float(selected_loss.detach()),
                "terms": {k: float(v) for k, v in terms.items()},
                "gradients": grads,
                "predictions": list(observations),
                "output_cotangents": list(output_cotangents),
            }
            report["runs"][label] = {
                "seconds": time.monotonic() - before,
                "loss": result["loss"],
                "terms": result["terms"],
                "finite": True,
                "checkpoint": True,
                "compiled": compiled,
                "objective": objective,
                "gradient_norm": float(sum(v.double().square().sum() for v in grads.values()).sqrt()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            save()
            print("DONE", label, report["runs"][label], flush=True)
            return result

        def record(label, a, b):
            report["comparisons"][label] = difference(a, b)
            save()
            print("COMPARE", label, report["comparisons"][label]["gradient"], flush=True)

        if not cli.localize_only:
            eager = run("nonzero_eager_total")
            repeat = run("nonzero_eager_repeat")
            record("eager_repeat", repeat, eager)
            del repeat
            compiled = run("nonzero_compiled_total", compiled=True)
            record("compiled_vs_eager", compiled, eager)
            repeat = run("nonzero_compiled_repeat", compiled=True)
            record("compiled_repeat", repeat, compiled)
            del repeat, compiled
            del eager
            for objective in ("target", "preservation"):
                a = run("nonzero_eager_" + objective, objective=objective)
                b = run("nonzero_compiled_" + objective, compiled=True, objective=objective)
                record(objective + "_compiled_vs_eager", b, a)
                if objective == "target":
                    ordinary = run("nonzero_eager_original", objective="original")
                    target_only = dict(a, predictions=[a["predictions"][1]], output_cotangents=[a["output_cotangents"][1]])
                    record("original_vs_preservation_target", ordinary, target_only)
                    ordinary_compiled = run("nonzero_compiled_original", compiled=True, objective="original")
                    record("original_compiled_vs_eager", ordinary_compiled, ordinary)
                    matched = run(
                        "nonzero_eager_target_matched_cotangent",
                        objective="target",
                        matched_cotangent=ordinary["output_cotangents"][0],
                    )
                    matched_target = dict(
                        matched, predictions=[matched["predictions"][1]], output_cotangents=[matched["output_cotangents"][1]]
                    )
                    record("original_vs_matched_target", matched_target, ordinary)
                    del ordinary, ordinary_compiled, matched, matched_target
                del a, b
            nonzero_up = [module.lora_up.weight.detach().cpu().clone() for module in network.unet_loras]
            for module in network.unet_loras:
                torch.nn.init.zeros_(module.lora_up.weight)
            a = run("zero_up_eager_total")
            b = run("zero_up_compiled_total", compiled=True)
            record("zero_up_compiled_vs_eager", b, a)
            del a, b
            # Optional controls run last, and never discard the main branch results.
            with torch.no_grad():
                for module, weight in zip(network.unet_loras, nonzero_up):
                    module.lora_up.weight.copy_(weight)
            del nonzero_up

        @contextmanager
        def optional_control(label):
            try:
                yield
            except Exception:  # noqa: BLE001 - optional diagnostic must retain main results
                report["errors"][label] = traceback.format_exc()
                print("OPTIONAL CONTROL FAILED", label, report["errors"][label], flush=True)
            finally:
                network.zero_grad(set_to_none=True)
                observations.clear()
                output_cotangents.clear()
                gc.collect()
                torch.cuda.empty_cache()
                save()

        if cli.localize or cli.localize_only:
            from krea2_preservation_localize import localize

            with optional_control("localization"):
                localize(model, network, blocks, run, report, save, backend=cli.compile_backend)
        if cli.checkpoint_controls:
            # Dequantized weights for even one uncheckpointed block can exceed
            # the full-input VRAM budget. Use identical cropped inputs for this
            # baseline and every bypass, retaining checkpointing everywhere else.
            latents = [x[..., :16, :16].contiguous() for x in latents[:1]]
            noise = [x[..., :16, :16].contiguous() for x in noise[:1]]
            alpha = torch.full((128, 128), 0.5, device="cuda")
            alpha[:32] = 0
            alpha[-32:] = 1
            batch = {"alpha_mask": [alpha], "krea2_vl_embed": [texts[0][:32].contiguous()]}
            report["checkpoint_control"] = "128x128 crop, 32 text tokens, alpha 0/.5/1; bypass one block"
            with optional_control("checkpoint_baseline"):
                baseline = run("selective_all_checkpoint")
                for index in (0, len(blocks) // 2, len(blocks) - 1):
                    label = f"selective_bypass_block_{index}"
                    try:
                        with bypass_block_checkpoint(blocks[index]):
                            result = run(label)
                        record(label, result, baseline)
                        del result
                    except Exception:  # noqa: BLE001 - optional diagnostic must retain main results
                        report["errors"][label] = traceback.format_exc()
                        print("OPTIONAL CONTROL FAILED", label, report["errors"][label], flush=True)
                    finally:
                        network.zero_grad(set_to_none=True)
                        observations.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
                        save()
                del baseline
        handle.remove()
        report["status"] = "complete_with_control_errors" if report["errors"] else "complete"
    except Exception:
        report["status"] = "failed"
        report["errors"]["fatal"] = traceback.format_exc()
        raise
    finally:
        save()
        lines = [
            "# Preservation gradient isolation",
            "",
            "Status: " + report["status"],
            "",
            "No optimizer or training. Native attention, full RAW scaled FP8/BF16, rank32 LoRA.",
            "Optional checkpoint controls use the same 128x128 crop and bypass one block at a time.",
            "",
            "| Comparison | Gradient relative L2 | Cosine |",
            "|---|---:|---:|",
        ]
        for name, value in report["comparisons"].items():
            g = value["gradient"]
            cosine = "undefined (zero norm)" if g["cosine"] is None else f"{g['cosine']:.6g}"
            lines.append(f"| {name} | {g['relative_l2']:.6g} | {cosine} |")
            if name.startswith("zero_up"):
                for group in ("lora_up", "lora_down"):
                    metric = value["groups"][group]
                    lines.append(f"| {name}/{group} | {metric['relative_l2']:.6g} | {metric['cosine']} |")
        if report["errors"]:
            lines += ["", "```", str(report["errors"]), "```"]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic-fa", action="store_true")
    parser.add_argument("--checkpoint-controls", action="store_true")
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--localize-only", action="store_true")
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--compile-backend", choices=("inductor", "aot_eager"), default="inductor")
    main(parser.parse_args())
