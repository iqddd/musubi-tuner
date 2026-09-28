"""Opt-in, no-optimizer isolation of preservation eager/compile gradient differences."""

import argparse
import json
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from krea2_preservation_gpu_probe import FixedInputs, compare
import musubi_tuner.krea2.output_preservation as preservation
from musubi_tuner.krea2.krea2_utils import load_krea2_dit
from musubi_tuner.krea2.mixed_token_batch import process_mixed_token_batch
from musubi_tuner.networks import lora_krea2


def difference(left, right):
    names = sorted(left["gradients"].keys() & right["gradients"].keys())
    a = torch.cat([left["gradients"][n].flatten() for n in names])
    b = torch.cat([right["gradients"][n].flatten() for n in names])
    by_group = {}
    for group in ("lora_up", "lora_down", "blocks", "txtfusion", "tmlp", "first"):
        matching = [n for n in names if group in n]
        if matching:
            by_group[group] = compare(torch.cat([left["gradients"][n].flatten() for n in matching]),
                                      torch.cat([right["gradients"][n].flatten() for n in matching]))
    return dict(gradient=compare(a, b), reference_gradient_norm=float(b.norm()),
                tested_gradient_norm=float(a.norm()), groups=by_group,
                missing_or_extra=sorted(left["gradients"].keys() ^ right["gradients"].keys()),
                predictions=[compare(a, b) for a, b in zip(left["predictions"], right["predictions"])],
                losses=[left["loss"], right["loss"]], terms=[left["terms"], right["terms"]])


def main(cli):
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", runs={}, comparisons={}, errors={}, optimizer=None)
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))

    try:
        torch.set_num_threads(4)
        torch._inductor.config.compile_threads = 4
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
            import musubi_tuner.modules.attention as attention_module
            attention_module.flash_attn_func = partial(attention_module.flash_attn_func, deterministic=True)
            attention_module.flash_attn_varlen_func = partial(attention_module.flash_attn_varlen_func, deterministic=True)
        model = load_krea2_dit(cli.dit, device="cuda", dtype=torch.bfloat16, fp8_scaled=True, attn_mode="flash")
        model.requires_grad_(False)
        network = lora_krea2.create_arch_network(1., 32, 32, None, [], model)
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
        batch = dict(alpha_mask=masks, krea2_vl_embed=texts)
        report["inputs"] = entries
        report["lora_initialization"] = "rank32, up normal std0.005 initially; separate zero-up control at end"
        acc = SimpleNamespace(device=torch.device("cuda"),
                              autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16),
                              unwrap_model=lambda x: x)
        real_loss = preservation.branch_loss
        observations = []

        def observe(module, inputs, kwargs, output):
            mask = kwargs["packed_alpha_plan"].image_mask if "packed_alpha_plan" in kwargs else kwargs["image_mask"]
            observations.append(output[mask].detach().float().cpu())

        handle = model.register_forward_hook(observe, with_kwargs=True)

        def run(label, compiled=False, checkpoint=True, objective="total"):
            nonlocal compiled_blocks
            print("START", label, flush=True)
            before = time.monotonic()
            network.zero_grad(set_to_none=True)
            observations.clear()
            torch.manual_seed(420)
            model.gradient_checkpointing = checkpoint
            if compiled and compiled_blocks is None:
                compiled_blocks = [torch.compile(block) for block in blocks]
            model.blocks = torch.nn.ModuleList(compiled_blocks if compiled else blocks)
            parts = []

            def capture(*a, **kw):
                loss = real_loss(*a, **kw)
                parts.append(loss)
                return loss

            with patch.object(preservation, "branch_loss", capture):
                opts = SimpleNamespace(alpha_masked_attention_mode="native", alpha_masked_token_drop=True,
                                       gradient_checkpointing=checkpoint,
                                       alpha_masked_attention_gamma=None, weighting_scheme="none")
                if objective == "original":
                    loss, terms = process_mixed_token_batch(FixedInputs(), opts, acc, model, batch,
                                                            latents, noise, None, torch.bfloat16, torch.float32)
                else:
                    loss, terms = preservation.process_preservation_batch(
                        FixedInputs(), opts, acc, model, network, batch, latents, noise, None,
                        torch.bfloat16, torch.float32)
            selected_loss = loss if objective in ("total", "original") else parts[0 if objective == "target" else 1]
            selected_loss.backward()
            grads = {name: parameter.grad.detach().float().cpu().clone()
                     for name, parameter in network.named_parameters() if parameter.grad is not None}
            assert len(grads) == len(list(network.parameters())), "Missing LoRA gradients"
            assert all(torch.isfinite(v).all() for v in [loss, *grads.values(), *observations])
            result = dict(loss=float(selected_loss.detach()), terms={k: float(v) for k, v in terms.items()},
                          gradients=grads, predictions=list(observations))
            report["runs"][label] = dict(seconds=time.monotonic()-before, loss=result["loss"],
                terms=result["terms"], finite=True, checkpoint=checkpoint, compiled=compiled, objective=objective,
                gradient_norm=float(sum(v.square().sum() for v in grads.values()).sqrt()))
            save()
            print("DONE", label, report["runs"][label], flush=True)
            return result

        def record(label, a, b):
            report["comparisons"][label] = difference(a, b)
            save()
            print("COMPARE", label, report["comparisons"][label]["gradient"], flush=True)

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
        original_inputs = batch, latents, noise
        # A dedicated small-input pair changes neither weights nor arithmetic,
        # and avoids >120 GiB of saved activations in the full-length no-CP case.
        latents = [x[..., :16, :16].contiguous() for x in latents[:1]]
        noise = [x[..., :16, :16].contiguous() for x in noise[:1]]
        alpha = torch.full((128, 128), 0.5, device="cuda")
        alpha[:32] = 0
        alpha[-32:] = 1
        batch = dict(alpha_mask=[alpha], krea2_vl_embed=[texts[0][:32].contiguous()])
        report["checkpoint_control"] = "one 128x128 image latent crop; 32 text tokens; synthetic alpha 0/.5/1"
        with_checkpoint = run("small_eager_checkpoint")
        no_checkpoint = run("small_eager_no_checkpoint", checkpoint=False)
        record("small_eager_checkpoint_vs_disabled", with_checkpoint, no_checkpoint)
        del no_checkpoint, with_checkpoint
        batch, latents, noise = original_inputs
        for objective in ("target", "preservation"):
            a = run("nonzero_eager_" + objective, objective=objective)
            b = run("nonzero_compiled_" + objective, compiled=True, objective=objective)
            record(objective + "_compiled_vs_eager", b, a)
            if objective == "target":
                ordinary = run("nonzero_eager_original", objective="original")
                target_only = dict(a, predictions=[a["predictions"][1]])
                record("original_vs_preservation_target", ordinary, target_only)
                del ordinary
            del a, b
        for module in network.unet_loras:
            torch.nn.init.zeros_(module.lora_up.weight)
        a = run("zero_up_eager_total")
        b = run("zero_up_compiled_total", compiled=True)
        record("zero_up_compiled_vs_eager", b, a)
        del a, b
        handle.remove()
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"
        report["errors"]["fatal"] = traceback.format_exc()
        raise
    finally:
        save()
        lines = ["# Preservation gradient isolation", "", "Status: " + report["status"], "",
                 "No optimizer or training. Native attention, full RAW scaled FP8/BF16, rank32 LoRA.",
                 "Checkpoint control uses a separate 128x128 crop / 32 text-token pair; other tests use full inputs.", "",
                 "| Comparison | Gradient relative L2 | Cosine |", "|---|---:|---:|"]
        for name, value in report["comparisons"].items():
            g = value["gradient"]
            lines.append(f"| {name} | {g['relative_l2']:.6g} | {g['cosine']:.6g} |")
        if report["errors"]:
            lines += ["", "```", str(report["errors"]), "```"]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deterministic-fa", action="store_true")
    main(parser.parse_args())
