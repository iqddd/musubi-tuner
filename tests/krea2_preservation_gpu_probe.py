"""Explicit opt-in real-weight probe (not collected by pytest); never trains.

Example: python tests/krea2_preservation_gpu_probe.py --dit ... --manifest ... --output ...
The manifest is a JSON list of {latent, text} cache paths. Outputs are diagnostics only.
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch._dynamo.utils import counters

from musubi_tuner.krea2.krea2_utils import load_krea2_dit
from musubi_tuner.krea2.log_bias import require_fa2_alpha
from musubi_tuner.krea2.output_preservation import process_preservation_batch
from musubi_tuner.networks import lora_krea2


def compare(a, b):
    # Large concatenated LoRA vectors need FP64 reductions: FP32 cosine on
    # hundreds of millions of elements can even report a value above one.
    a, b = a.double().flatten(), b.double().flatten()
    delta = a - b
    norm_a, norm_b = a.norm(), b.norm()
    return dict(max_abs=float(delta.abs().max()), mean_abs=float(delta.abs().mean()),
                reference_norm=float(norm_b), tested_norm=float(norm_a),
                both_zero=bool(norm_a == 0 and norm_b == 0),
                relative_l2=float(delta.norm() / norm_b.clamp_min(1e-30)),
                cosine=float(torch.dot(a, b) / (norm_a * norm_b).clamp_min(1e-300)))


class FixedInputs:
    def get_noisy_model_input_and_timesteps(self, args, noise, latents, preset, *unused):
        return latents * 0.25 + noise * 0.75, torch.full((latents.shape[0],), 750., device=latents.device)


def main(cli):
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(status="running", model=cli.dit, checkpointing=True, fp8_scaled=True,
                  dtype="BF16 autocast / FP32 trainable rank32 LoRA", optimizer=None, modes={})

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))

    try:
        torch.set_num_threads(4)
        torch._inductor.config.compile_threads = 4
        torch.manual_seed(420)
        require_fa2_alpha()
        print("Loading RAW scaled-FP8 model", flush=True)
        model = load_krea2_dit(cli.dit, device="cuda", dtype=torch.bfloat16,
                               fp8_scaled=True, attn_mode="flash")
        model.requires_grad_(False)
        network = lora_krea2.create_arch_network(1., 32, 32, None, [], model)
        network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
        network.to("cuda").train().requires_grad_(True)
        # Nonzero adapters make teacher restoration and preservation gradients observable.
        for module in network.unet_loras:
            torch.nn.init.normal_(module.lora_up.weight, std=0.005)
        model.train().enable_gradient_checkpointing()
        entries = json.loads(Path(cli.manifest).read_text())[:2]
        latents, masks, texts = [], [], []
        for entry in entries:
            latent_sd, text_sd = load_file(entry["latent"]), load_file(entry["text"])
            latent = next(v for k, v in latent_sd.items() if k.startswith("latents_")).float().cuda()
            latents.append(latent)
            alpha = latent_sd.get("alpha_mask")
            masks.append(None if alpha is None else alpha.cuda())
            texts.append(next(v for k, v in sorted(text_sd.items()) if k.startswith("varlen_krea2_vl_embed")).cuda())
        noise = [torch.randn_like(x) for x in latents]
        batch = dict(alpha_mask=masks, krea2_vl_embed=texts)
        report["inputs"] = entries
        report["latent_shapes"] = [list(x.shape) for x in latents]
        report["lora"] = "seed420 nonzero random up std0.005; no optimizer/updates"
        accelerator = SimpleNamespace(device=torch.device("cuda"),
            autocast=lambda: torch.autocast("cuda", dtype=torch.bfloat16), unwrap_model=lambda x: x)
        original_blocks = list(model.blocks)
        snapshots, pass_info = [], []

        def observe(module, unused, kwargs, value):
            plan = kwargs["packed_alpha_plan"]
            snapshots.append(value[plan.image_mask].detach().float().cpu())
            pass_info.append(dict(plan_id=id(plan), grad=torch.is_grad_enabled(),
                                  multiplier=network.unet_loras[0].multiplier))

        handle = model.register_forward_hook(observe, with_kwargs=True)

        def run(mode, seed):
            network.zero_grad(set_to_none=True)
            snapshots.clear()
            pass_info.clear()
            torch.manual_seed(seed)
            args = SimpleNamespace(alpha_masked_attention_mode=mode,
                                   alpha_masked_attention_gamma=3., weighting_scheme="none")
            loss, terms = process_preservation_batch(FixedInputs(), args, accelerator, model, network,
                                                      batch, latents, noise, None, torch.bfloat16, torch.float32)
            rng_before = torch.cuda.get_rng_state().clone()
            loss.backward()
            assert torch.equal(rng_before, torch.cuda.get_rng_state()), "checkpoint changed RNG"
            assert len(pass_info) == 3 and pass_info[0]["plan_id"] == pass_info[2]["plan_id"]
            assert pass_info[0]["multiplier"] == 0 and not pass_info[0]["grad"]
            assert all(p["multiplier"] == 1 and p["grad"] for p in pass_info[1:])
            grads = {n: p.grad.detach().float().cpu() for n, p in network.named_parameters() if p.grad is not None}
            assert grads and all(torch.isfinite(x).all() for x in [loss, *snapshots, *grads.values()])
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            return dict(loss=float(loss.detach()), terms={k: float(v) for k, v in terms.items()},
                        predictions=list(snapshots), gradients=grads,
                        graphs=counters["stats"]["unique_graphs"])

        for mode in ("native", "sharedkv", "logbias"):
            model.blocks = torch.nn.ModuleList(original_blocks)
            torch._dynamo.reset()
            counters.clear()
            print(f"{mode}: eager", flush=True)
            eager = run(mode, 420)
            model.blocks = torch.nn.ModuleList([torch.compile(block) for block in original_blocks])
            print(f"{mode}: compiled", flush=True)
            compiled = run(mode, 420)
            assert eager["gradients"].keys() == compiled["gradients"].keys()
            names = list(eager["gradients"])
            per_parameter = {name: compare(compiled["gradients"][name], eager["gradients"][name]) for name in names}
            g = compare(torch.cat([compiled["gradients"][n].flatten() for n in names]),
                        torch.cat([eager["gradients"][n].flatten() for n in names]))
            metrics = dict(predictions=[compare(a, b) for a, b in zip(compiled["predictions"], eager["predictions"])],
                loss_relative_difference=abs(compiled["loss"]-eager["loss"])/max(abs(eager["loss"]), 1e-30),
                eager_terms=eager["terms"], compiled_terms=compiled["terms"], gradients=g,
                worst_parameter=max(per_parameter, key=lambda n: per_parameter[n]["relative_l2"]),
                per_parameter=per_parameter, missing_or_extra_gradients=[])
            del eager, compiled
            graphs = []
            for seed in (421, 422, 423):
                print(f"{mode}: graph stability seed={seed}", flush=True)
                result = run(mode, seed)
                graphs.append(result["graphs"])
                del result
            metrics["graphs_after_warmup"] = graphs
            metrics["graph_stable"] = len(set(graphs)) == 1
            metrics["numeric_thresholds_pass"] = (
                all(x["relative_l2"] <= 5e-3 for x in metrics["predictions"])
                and metrics["loss_relative_difference"] <= 5e-3
                and g["relative_l2"] <= 1e-2 and g["cosine"] >= 0.999
            )
            report["modes"][mode] = metrics
            save()
            print(mode, {k: v for k, v in metrics.items() if k != "per_parameter"}, flush=True)
        handle.remove()
        report["status"] = "passed" if all(x["numeric_thresholds_pass"] and x["graph_stable"]
                                              for x in report["modes"].values()) else "needs_review"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()
        lines = ["# Krea2 output preservation GPU probe", "", f"Status: {report['status']}",
                 "", "Real RAW scaled-FP8, BF16, rank32 LoRA, checkpointing, default compile. No training.", ""]
        for mode, result in report["modes"].items():
            lines.append(f"- {mode}: numeric pass={result['numeric_thresholds_pass']}; "
                         f"gradient rel-L2={result['gradients']['relative_l2']:.6g}; "
                         f"cosine={result['gradients']['cosine']:.6g}; graphs={result['graphs_after_warmup']}")
        if "error" in report:
            lines += ["", "```", report["error"], "```"]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
