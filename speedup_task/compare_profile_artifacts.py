import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

# Fixed baseline ensemble for nondeterminism checks.
# Source: 8 reference runs in `speedup_task/nondet_n5_m12_adamw/run1..run8`,
# which were used to build `speedup_task/nondet_n5_m12_adamw/analysis_summary.json`.
ENSEMBLE_BASELINE_RUNS = [f"run{i}" for i in range(1, 9)]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_stop_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("**/stop_step_*"))
    if not candidates:
        raise FileNotFoundError(f"stop_step_* directory not found under: {run_dir}")
    return candidates[-1]


def find_lora_checkpoint(stop_dir: Path) -> Path | None:
    candidates = sorted(stop_dir.glob("*.safetensors"))
    if not candidates:
        return None
    return candidates[0]


def summarize_matrix_group(state_dict: dict[str, torch.Tensor], keys: list[str]) -> dict:
    if not keys:
        return {
            "tensor_count": 0,
            "numel_total": 0,
            "abs_mean_weighted": 0.0,
            "abs_max": 0.0,
            "zero_ratio": 0.0,
        }

    numel_total = 0
    abs_sum = 0.0
    abs_max = 0.0
    zero_count = 0

    for key in keys:
        tensor = state_dict[key].detach().to(torch.float32).cpu()
        abs_tensor = tensor.abs()
        numel = int(tensor.numel())
        numel_total += numel
        abs_sum += float(abs_tensor.sum().item())
        abs_max = max(abs_max, float(abs_tensor.max().item()))
        zero_count += int((tensor == 0).sum().item())

    return {
        "tensor_count": len(keys),
        "numel_total": numel_total,
        "abs_mean_weighted": abs_sum / numel_total if numel_total else 0.0,
        "abs_max": abs_max,
        "zero_ratio": zero_count / numel_total if numel_total else 0.0,
    }


def build_single_lora_effect_probe(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    probe_dim: int = 16,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Functional proxy for LoRA effect:
      y = (B @ A) * scale @ probe
    where probe is deterministic per-prefix random matrix.
    """
    down = state_dict[f"{prefix}.lora_down.weight"]
    up = state_dict[f"{prefix}.lora_up.weight"]
    alpha_key = f"{prefix}.alpha"

    down2 = down.to(device=device, dtype=torch.float32).reshape(down.shape[0], -1)  # [r, in_flat]
    up2 = up.to(device=device, dtype=torch.float32)
    if up2.ndim == 4:
        if up2.shape[2:] == (1, 1):
            up2 = up2[:, :, 0, 0]
        else:
            up2 = up2.reshape(up2.shape[0], up2.shape[1])
    else:
        up2 = up2.reshape(up2.shape[0], up2.shape[1])
    rank = down2.shape[0]

    # Keep alpha in functional effect computation (scale = alpha / rank),
    # but we do not report a standalone alpha-diff metric because it is scalar and typically constant.
    if alpha_key in state_dict:
        alpha_val = float(state_dict[alpha_key].reshape(-1)[0].item())
        scale = alpha_val / float(rank)
    else:
        scale = 1.0

    # Deterministic probe to make comparisons repeatable.
    seed = int.from_bytes(hashlib.sha256(prefix.encode("utf-8")).digest()[:8], byteorder="little")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    probe = torch.randn(down2.shape[1], probe_dim, generator=g, dtype=torch.float32, device=device)
    projected = torch.matmul(up2, torch.matmul(down2, probe)) * scale  # [out, probe_dim]
    return projected.cpu()


def build_lora_effect_probe_state(
    state_dict: dict[str, torch.Tensor],
    probe_dim: int = 16,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    down_suffix = ".lora_down.weight"
    probe_state: dict[str, torch.Tensor] = {}
    prefixes = sorted({k[: -len(down_suffix)] for k in state_dict if k.endswith(down_suffix)})
    for prefix in prefixes:
        up_key = f"{prefix}.lora_up.weight"
        if up_key not in state_dict:
            continue
        probe_state[f"{prefix}.lora_effect_probe"] = build_single_lora_effect_probe(
            state_dict, prefix, probe_dim=probe_dim, device=device
        )
    return probe_state


def compare_matrix_group(
    baseline_sd: dict[str, torch.Tensor],
    candidate_sd: dict[str, torch.Tensor],
    suffix: str,
    top_k: int,
) -> dict:
    baseline_keys = {k for k in baseline_sd.keys() if k.endswith(suffix)}
    candidate_keys = {k for k in candidate_sd.keys() if k.endswith(suffix)}
    shared_keys = sorted(baseline_keys & candidate_keys)
    only_baseline = sorted(baseline_keys - candidate_keys)
    only_candidate = sorted(candidate_keys - baseline_keys)

    diff_numel_total = 0
    changed_numel_total = 0
    equal_tensors = 0
    incompatible = []
    max_abs = 0.0
    abs_sum = 0.0
    per_tensor = []
    baseline_sq_sum = 0.0
    candidate_sq_sum = 0.0
    diff_sq_sum = 0.0
    dot_sum = 0.0

    for key in shared_keys:
        b = baseline_sd[key]
        c = candidate_sd[key]
        if b.shape != c.shape or b.dtype != c.dtype:
            incompatible.append(
                {
                    "key": key,
                    "baseline_shape": tuple(b.shape),
                    "candidate_shape": tuple(c.shape),
                    "baseline_dtype": str(b.dtype),
                    "candidate_dtype": str(c.dtype),
                }
            )
            continue

        bd = b.detach().to(torch.float32).cpu()
        cd = c.detach().to(torch.float32).cpu()
        diff_signed = bd - cd
        diff = diff_signed.abs()
        numel = int(diff.numel())

        baseline_sq_sum += float((bd * bd).sum().item())
        candidate_sq_sum += float((cd * cd).sum().item())
        diff_sq_sum += float((diff_signed * diff_signed).sum().item())
        dot_sum += float((bd * cd).sum().item())

        if torch.equal(b, c):
            equal_tensors += 1
            per_tensor.append({"key": key, "max_abs": 0.0, "mean_abs": 0.0, "changed_ratio": 0.0})
            diff_numel_total += numel
            continue

        changed_numel = int((diff != 0).sum().item())
        tensor_max_abs = float(diff.max().item()) if numel else 0.0
        tensor_mean_abs = float(diff.mean().item()) if numel else 0.0

        diff_numel_total += numel
        changed_numel_total += changed_numel
        max_abs = max(max_abs, tensor_max_abs)
        abs_sum += float(diff.sum().item())
        per_tensor.append(
            {
                "key": key,
                "max_abs": tensor_max_abs,
                "mean_abs": tensor_mean_abs,
                "changed_ratio": (changed_numel / numel) if numel else 0.0,
            }
        )

    per_tensor_sorted = sorted(per_tensor, key=lambda x: x["max_abs"], reverse=True)
    eps = 1e-12
    baseline_l2 = baseline_sq_sum**0.5
    candidate_l2 = candidate_sq_sum**0.5
    diff_l2 = diff_sq_sum**0.5
    baseline_summary = summarize_matrix_group(baseline_sd, sorted(baseline_keys))
    candidate_summary = summarize_matrix_group(candidate_sd, sorted(candidate_keys))

    return {
        "suffix": suffix,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "shared_tensor_count": len(shared_keys),
        "equal_tensor_count": equal_tensors,
        "incompatible_count": len(incompatible),
        "only_in_baseline_count": len(only_baseline),
        "only_in_candidate_count": len(only_candidate),
        "changed_numel_total": changed_numel_total,
        "diff_numel_total": diff_numel_total,
        "changed_ratio_total": (changed_numel_total / diff_numel_total) if diff_numel_total else 0.0,
        "max_abs_diff": max_abs,
        "mean_abs_diff_weighted": (abs_sum / diff_numel_total) if diff_numel_total else 0.0,
        "baseline_l2": baseline_l2,
        "candidate_l2": candidate_l2,
        "diff_l2": diff_l2,
        "relative_l2_to_baseline": (diff_l2 / (baseline_l2 + eps)) if baseline_l2 else 0.0,
        "cosine_similarity": max(-1.0, min(1.0, dot_sum / (baseline_l2 * candidate_l2 + eps))),
        "relative_mean_abs_to_baseline_abs_mean": (
            (abs_sum / diff_numel_total) / (baseline_summary["abs_mean_weighted"] + eps)
            if diff_numel_total
            else 0.0
        ),
        "top_diff_tensors": per_tensor_sorted[:top_k],
    }


def print_group_report(report: dict):
    print(f"\n[group: {report['suffix']}]")
    print(
        "shared="
        f"{report['shared_tensor_count']} "
        f"equal={report['equal_tensor_count']} "
        f"incompatible={report['incompatible_count']} "
        f"only_baseline={report['only_in_baseline_count']} "
        f"only_candidate={report['only_in_candidate_count']}"
    )
    print(
        "diff: "
        f"changed_numel={report['changed_numel_total']} "
        f"total_numel={report['diff_numel_total']} "
        f"changed_ratio={report['changed_ratio_total']:.6f} "
        f"mean_abs={report['mean_abs_diff_weighted']:.10f}"
    )
    print(
        "relative: "
        f"rel_mean_abs={report['relative_mean_abs_to_baseline_abs_mean']:.10f} "
        f"rel_l2={report['relative_l2_to_baseline']:.10f} "
        f"cosine={report['cosine_similarity']:.10f}"
    )
    b = report["baseline_summary"]
    c = report["candidate_summary"]
    print(
        "baseline_abs: "
        f"mean={b['abs_mean_weighted']:.10f} "
        f"max={b['abs_max']:.10f} "
        f"zero_ratio={b['zero_ratio']:.6f}"
    )
    print(
        "candidate_abs: "
        f"mean={c['abs_mean_weighted']:.10f} "
        f"max={c['abs_max']:.10f} "
        f"zero_ratio={c['zero_ratio']:.6f}"
    )
    print("top_diff_tensors:")
    for row in report["top_diff_tensors"]:
        print(
            "  "
            f"{row['key']} "
            f"mean_abs={row['mean_abs']:.10f} "
            f"changed_ratio={row['changed_ratio']:.6f}"
        )


def resolve_effect_device(mode: str) -> str:
    if mode == "auto":
        # In some environments torch.cuda.is_available() can emit noisy warnings
        # even though comparison can proceed on CPU fallback.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cudaGetDeviceCount.*")
            warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
            return "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "cuda":
        # Honor explicit user intent; avoid pre-flight probing that may be flaky
        # under constrained runtimes. Any real CUDA issue will surface on tensor ops.
        return "cuda"
    return mode


def compute_upper_envelope_assessment(value: float, p99: float | None, max_v: float | None) -> dict[str, float | None]:
    eps = 1e-12
    ratio_to_p99 = (value / (p99 + eps)) if p99 is not None else None
    ratio_to_max = (value / (max_v + eps)) if max_v is not None else None
    outside_p99_pct = ((value - p99) / (p99 + eps) * 100.0) if p99 is not None and value > p99 else 0.0
    outside_max_pct = ((value - max_v) / (max_v + eps) * 100.0) if max_v is not None and value > max_v else 0.0
    return {
        "ratio_to_p99": ratio_to_p99,
        "ratio_to_max": ratio_to_max,
        "outside_p99_pct": outside_p99_pct,
        "outside_max_pct": outside_max_pct,
    }


def compute_lower_envelope_assessment(value: float, min_v: float | None) -> dict[str, float | None]:
    eps = 1e-12
    ratio_min_over_value = (min_v / (value + eps)) if min_v is not None else None
    outside_min_pct = ((min_v - value) / (min_v + eps) * 100.0) if min_v is not None and value < min_v else 0.0
    return {
        "ratio_min_over_value": ratio_min_over_value,
        "outside_min_pct": outside_min_pct,
    }


def print_upper_assessment_line(name: str, value: float, summary_metric: dict[str, Any]):
    assessed = compute_upper_envelope_assessment(
        value=value,
        p99=summary_metric.get("p99"),
        max_v=summary_metric.get("max"),
    )
    ratio_to_p99 = assessed["ratio_to_p99"]
    ratio_to_max = assessed["ratio_to_max"]
    ratio_to_p99_text = f"{ratio_to_p99:.6f}" if ratio_to_p99 is not None else "n/a"
    ratio_to_max_text = f"{ratio_to_max:.6f}" if ratio_to_max is not None else "n/a"
    print(
        f"{name}: value={value:.10f} "
        f"ratio_to_p99={ratio_to_p99_text} "
        f"ratio_to_max={ratio_to_max_text} "
        f"outside_p99_pct={assessed['outside_p99_pct']:.4f}% "
        f"outside_max_pct={assessed['outside_max_pct']:.4f}%"
    )


def print_lower_assessment_line(name: str, value: float, summary_metric: dict[str, Any]):
    assessed = compute_lower_envelope_assessment(value=value, min_v=summary_metric.get("min"))
    ratio_min_over_value = assessed["ratio_min_over_value"]
    ratio_min_text = f"{ratio_min_over_value:.6f}" if ratio_min_over_value is not None else "n/a"
    print(
        f"{name}: value={value:.10f} "
        f"ratio_min_over_value={ratio_min_text} "
        f"outside_min_pct={assessed['outside_min_pct']:.4f}%"
    )


def print_nondet_numeric_assessment(
    nondet_summary: dict[str, Any],
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
    group_reports: dict[str, dict[str, Any]],
):
    print("\n[nondet_numeric_assessment]")
    print(f"summary_root={nondet_summary.get('root')}")
    print(f"summary_run_count={nondet_summary.get('run_count')}")

    loss_abs_diff = abs(float(baseline_result.get("current_loss", 0.0)) - float(candidate_result.get("current_loss", 0.0)))
    moving_loss_abs_diff = abs(
        float(baseline_result.get("moving_average_loss", 0.0)) - float(candidate_result.get("moving_average_loss", 0.0))
    )
    loss_summary = nondet_summary.get("pair_loss_abs_diff")
    moving_summary = nondet_summary.get("pair_moving_loss_abs_diff")
    if loss_summary:
        print_upper_assessment_line("current_loss_abs_diff", loss_abs_diff, loss_summary)
    if moving_summary:
        print_upper_assessment_line("moving_average_loss_abs_diff", moving_loss_abs_diff, moving_summary)

    metrics_summary = nondet_summary.get("metrics_all_pairs", {})
    mapping = [
        (".lora_down.weight", "relative_l2_to_baseline", "rel_l2", "upper"),
        (".lora_down.weight", "relative_mean_abs_to_baseline_abs_mean", "rel_mean_abs", "upper"),
        (".lora_down.weight", "cosine_similarity", "cos", "lower"),
        (".lora_up.weight", "relative_l2_to_baseline", "rel_l2", "upper"),
        (".lora_up.weight", "relative_mean_abs_to_baseline_abs_mean", "rel_mean_abs", "upper"),
        (".lora_up.weight", "cosine_similarity", "cos", "lower"),
        (".lora_effect_probe", "relative_l2_to_baseline", "rel_l2", "upper"),
        (".lora_effect_probe", "relative_mean_abs_to_baseline_abs_mean", "rel_mean_abs", "upper"),
        (".lora_effect_probe", "cosine_similarity", "cos", "lower"),
    ]
    for suffix, report_key, summary_key, direction in mapping:
        report = group_reports.get(suffix)
        group_summary = metrics_summary.get(suffix, {})
        metric_summary = group_summary.get(summary_key)
        if report is None or metric_summary is None:
            continue
        metric_value = float(report.get(report_key, 0.0))
        label = f"{suffix}::{summary_key}"
        if direction == "upper":
            print_upper_assessment_line(label, metric_value, metric_summary)
        else:
            print_lower_assessment_line(label, metric_value, metric_summary)


def quantile_linear(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    w = pos - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)


def print_ensemble_metric_assessment(
    metric_name: str,
    values: list[float],
    nondet_metric: dict[str, Any],
):
    if not values:
        return
    p50 = quantile_linear(values, 0.5)
    p75 = quantile_linear(values, 0.75)
    p90 = quantile_linear(values, 0.9)
    print(f"{metric_name}: n={len(values)}")
    print_upper_assessment_line("  p50", p50, nondet_metric)
    print_upper_assessment_line("  p75", p75, nondet_metric)
    print_upper_assessment_line("  p90", p90, nondet_metric)


def print_nondet_ensemble_assessment(
    nondet_summary: dict[str, Any],
    baseline_root: Path,
    candidate_stop_dir: Path,
    candidate_result: dict[str, Any],
    candidate_sd: dict[str, torch.Tensor],
    effect_probe_dim: int,
    effect_device: str,
):
    print("\n[nondet_ensemble_assessment]")
    print(f"baseline_root={baseline_root}")
    print(f"candidate_stop_dir={candidate_stop_dir}")

    run_dirs = [baseline_root / run_name for run_name in ENSEMBLE_BASELINE_RUNS]
    missing = [str(p) for p in run_dirs if not p.exists()]
    if missing:
        print("missing_baseline_runs:")
        for path in missing:
            print(f"  {path}")
        print("skip ensemble assessment.")
        return

    candidate_effect_probe = build_lora_effect_probe_state(
        candidate_sd, probe_dim=effect_probe_dim, device=effect_device
    )
    metrics = {
        "current_loss_abs_diff": [],
        "moving_average_loss_abs_diff": [],
        ".lora_down.weight::rel_l2": [],
        ".lora_up.weight::rel_l2": [],
        ".lora_effect_probe::rel_l2": [],
    }

    for run_dir in run_dirs:
        stop_dir = find_stop_dir(run_dir)
        run_result = load_json(stop_dir / "run_result.json")
        run_ckpt = find_lora_checkpoint(stop_dir)
        if run_ckpt is None:
            continue
        run_sd = load_file(str(run_ckpt))
        down_report = compare_matrix_group(run_sd, candidate_sd, ".lora_down.weight", top_k=1)
        up_report = compare_matrix_group(run_sd, candidate_sd, ".lora_up.weight", top_k=1)
        run_effect_probe = build_lora_effect_probe_state(run_sd, probe_dim=effect_probe_dim, device=effect_device)
        effect_report = compare_matrix_group(run_effect_probe, candidate_effect_probe, ".lora_effect_probe", top_k=1)

        metrics["current_loss_abs_diff"].append(
            abs(float(run_result.get("current_loss", 0.0)) - float(candidate_result.get("current_loss", 0.0)))
        )
        metrics["moving_average_loss_abs_diff"].append(
            abs(float(run_result.get("moving_average_loss", 0.0)) - float(candidate_result.get("moving_average_loss", 0.0)))
        )
        metrics[".lora_down.weight::rel_l2"].append(float(down_report.get("relative_l2_to_baseline", 0.0)))
        metrics[".lora_up.weight::rel_l2"].append(float(up_report.get("relative_l2_to_baseline", 0.0)))
        metrics[".lora_effect_probe::rel_l2"].append(float(effect_report.get("relative_l2_to_baseline", 0.0)))

    print(f"baseline_runs_used={','.join(ENSEMBLE_BASELINE_RUNS)}")
    print_ensemble_metric_assessment(
        "current_loss_abs_diff",
        metrics["current_loss_abs_diff"],
        nondet_summary.get("pair_loss_abs_diff", {}),
    )
    print_ensemble_metric_assessment(
        "moving_average_loss_abs_diff",
        metrics["moving_average_loss_abs_diff"],
        nondet_summary.get("pair_moving_loss_abs_diff", {}),
    )
    metrics_summary = nondet_summary.get("metrics_all_pairs", {})
    print_ensemble_metric_assessment(
        ".lora_down.weight::rel_l2",
        metrics[".lora_down.weight::rel_l2"],
        metrics_summary.get(".lora_down.weight", {}).get("rel_l2", {}),
    )
    print_ensemble_metric_assessment(
        ".lora_up.weight::rel_l2",
        metrics[".lora_up.weight::rel_l2"],
        metrics_summary.get(".lora_up.weight", {}).get("rel_l2", {}),
    )
    print_ensemble_metric_assessment(
        ".lora_effect_probe::rel_l2",
        metrics[".lora_effect_probe::rel_l2"],
        metrics_summary.get(".lora_effect_probe", {}).get("rel_l2", {}),
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two FLUX.2 profiling artifact runs.")
    parser.add_argument("--baseline", default=None, help="Optional baseline run dir for pairwise diagnostics")
    parser.add_argument("--candidate", required=True, help="Candidate run dir")
    parser.add_argument("--top_k", type=int, default=5, help="How many tensors to print in top diff lists")
    parser.add_argument(
        "--effect_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for BA probe computation (default: auto)",
    )
    parser.add_argument(
        "--effect_probe_dim",
        type=int,
        default=16,
        help="Probe dimension for BA functional comparison (default: 16)",
    )
    parser.add_argument(
        "--nondet_summary",
        type=str,
        default=None,
        help="Optional path to analysis_summary.json with CUDA nondeterminism envelope.",
    )
    args = parser.parse_args()

    candidate_stop_dir = find_stop_dir(Path(args.candidate))

    candidate_result = load_json(candidate_stop_dir / "run_result.json")
    print(f"candidate_stop_dir={candidate_stop_dir}")

    candidate_lora_ckpt = find_lora_checkpoint(candidate_stop_dir)
    if candidate_lora_ckpt is None:
        print("\n[lora_matrix_check]")
        print("LoRA checkpoint (*.safetensors) not found in candidate stop dir, skip.")
        return
    candidate_sd = load_file(str(candidate_lora_ckpt))
    effect_device = resolve_effect_device(args.effect_device)
    print(f"effect_probe: device={effect_device}, probe_dim={args.effect_probe_dim}")

    baseline_result = None
    down_report = None
    up_report = None
    effect_report = None

    if args.baseline:
        baseline_stop_dir = find_stop_dir(Path(args.baseline))
        baseline_result = load_json(baseline_stop_dir / "run_result.json")
        baseline_manifest = load_json(baseline_stop_dir / "artifact_manifest.json")
        candidate_manifest = load_json(candidate_stop_dir / "artifact_manifest.json")

        print(f"baseline_stop_dir={baseline_stop_dir}")
        print(f"candidate_stop_dir={candidate_stop_dir}")

        keys_to_compare = ["global_step", "epoch", "current_loss", "moving_average_loss", "optimizer_class", "seed"]
        print("\n[run_result]")
        for key in keys_to_compare:
            print(f"{key}: baseline={baseline_result.get(key)} candidate={candidate_result.get(key)}")

        if "optimizer_state_summary" in baseline_result and "optimizer_state_summary" in candidate_result:
            b_sum = baseline_result["optimizer_state_summary"]
            c_sum = candidate_result["optimizer_state_summary"]
            print("\n[optimizer_state_summary]")
            for key in ["num_state_entries", "num_state_tensors", "num_state_numel"]:
                print(f"{key}: baseline={b_sum.get(key)} candidate={c_sum.get(key)}")

        print("\n[artifact_manifest]")
        baseline_paths = set(baseline_manifest.keys())
        candidate_paths = set(candidate_manifest.keys())
        only_baseline = sorted(baseline_paths - candidate_paths)
        only_candidate = sorted(candidate_paths - baseline_paths)
        shared_paths = sorted(baseline_paths & candidate_paths)

        if only_baseline:
            print("only_in_baseline:")
            for path in only_baseline:
                print(f"  {path}")
        if only_candidate:
            print("only_in_candidate:")
            for path in only_candidate:
                print(f"  {path}")

        changed = []
        for path in shared_paths:
            if baseline_manifest[path]["sha256"] != candidate_manifest[path]["sha256"]:
                changed.append(path)

        if changed:
            print("sha256_changed:")
            for path in changed:
                print(f"  {path}")
        else:
            print("sha256_changed: none")

        baseline_lora_ckpt = find_lora_checkpoint(baseline_stop_dir)
        if baseline_lora_ckpt is None:
            print("\n[lora_matrix_check]")
            print("LoRA checkpoint (*.safetensors) not found in baseline stop dir, skip pairwise.")
        else:
            baseline_sd = load_file(str(baseline_lora_ckpt))

            print("\n[lora_matrix_check]")
            print(f"baseline_lora_ckpt={baseline_lora_ckpt}")
            print(f"candidate_lora_ckpt={candidate_lora_ckpt}")

            down_report = compare_matrix_group(baseline_sd, candidate_sd, ".lora_down.weight", args.top_k)
            up_report = compare_matrix_group(baseline_sd, candidate_sd, ".lora_up.weight", args.top_k)
            baseline_effect_probe = build_lora_effect_probe_state(
                baseline_sd, probe_dim=args.effect_probe_dim, device=effect_device
            )
            candidate_effect_probe = build_lora_effect_probe_state(
                candidate_sd, probe_dim=args.effect_probe_dim, device=effect_device
            )
            effect_report = compare_matrix_group(
                baseline_effect_probe, candidate_effect_probe, ".lora_effect_probe", args.top_k
            )

            print_group_report(down_report)
            print_group_report(up_report)
            print_group_report(effect_report)
    else:
        print("\n[pairwise_comparison]")
        print("skip: --baseline not provided (ensemble-only mode).")

    if not args.nondet_summary:
        return

    nondet_summary = load_json(Path(args.nondet_summary))
    if baseline_result is not None and down_report is not None and up_report is not None and effect_report is not None:
        print_nondet_numeric_assessment(
            nondet_summary=nondet_summary,
            baseline_result=baseline_result,
            candidate_result=candidate_result,
            group_reports={
                ".lora_down.weight": down_report,
                ".lora_up.weight": up_report,
                ".lora_effect_probe": effect_report,
            },
        )

    baseline_root = Path(str(nondet_summary.get("root"))) if nondet_summary.get("root") else Path(args.candidate).parent
    print_nondet_ensemble_assessment(
        nondet_summary=nondet_summary,
        baseline_root=baseline_root,
        candidate_stop_dir=candidate_stop_dir,
        candidate_result=candidate_result,
        candidate_sd=candidate_sd,
        effect_probe_dim=args.effect_probe_dim,
        effect_device=effect_device,
    )


if __name__ == "__main__":
    main()
