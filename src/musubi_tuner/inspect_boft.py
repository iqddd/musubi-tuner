import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
from safetensors.torch import load_file


KLEIN_9B_EXPECTED_GROUP_COUNTS = {
    "double_img_attn_proj": 8,
    "double_img_attn_qkv": 8,
    "double_img_mlp_0": 8,
    "double_img_mlp_2": 8,
    "double_txt_attn_proj": 8,
    "double_txt_attn_qkv": 8,
    "double_txt_mlp_0": 8,
    "double_txt_mlp_2": 8,
    "single_linear1": 24,
    "single_linear2": 24,
}

KLEIN_9B_SINGLE_ONLY_GROUP_COUNTS = {
    "single_linear1": 24,
    "single_linear2": 24,
}

KLEIN_9B_DOUBLE_ONLY_GROUP_COUNTS = {
    "double_img_attn_proj": 8,
    "double_img_attn_qkv": 8,
    "double_img_mlp_0": 8,
    "double_img_mlp_2": 8,
    "double_txt_attn_proj": 8,
    "double_txt_attn_qkv": 8,
    "double_txt_mlp_0": 8,
    "double_txt_mlp_2": 8,
}

ARCHITECTURE_GROUP_PROFILES = {
    "klein-9b": KLEIN_9B_EXPECTED_GROUP_COUNTS,
    "klein-9b-single-only": KLEIN_9B_SINGLE_ONLY_GROUP_COUNTS,
    "klein-9b-double-only": KLEIN_9B_DOUBLE_ONLY_GROUP_COUNTS,
}


@dataclass
class ModuleSummary:
    name: str
    group: str
    q_norm: float
    representation: str
    oft_shape: tuple[int, ...]
    block_size: int
    saved_alpha: Optional[float]
    rescale_params: int
    rescale_mean: Optional[float]
    rescale_min: Optional[float]
    rescale_max: Optional[float]
    rescale_delta_abs_mean: Optional[float]
    rescale_delta_abs_max: Optional[float]
    packed_params: int
    stored_params: int
    export_params: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BOFT weights and report effective Q norms.")
    parser.add_argument("weights", type=str, help="Path to BOFT weights (.safetensors, .pt, .bin, etc.)")
    parser.add_argument(
        "--architecture",
        type=str,
        default="klein-9b",
        choices=["klein-9b", "klein-9b-single-only", "klein-9b-double-only", "none"],
        help="Optional architecture-specific grouping/validation. Default is klein-9b.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top/bottom modules to print. Default is 10.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu")


def infer_block_size_from_packed_size(packed_size: int) -> int:
    discriminant = 1 + 8 * packed_size
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        raise ValueError(f"Cannot infer block_size from packed size {packed_size}")
    block_size = (1 + root) // 2
    if block_size * (block_size - 1) // 2 != packed_size:
        raise ValueError(f"Packed size {packed_size} does not match any triangular block size")
    return block_size


def export_oft_blocks_from_packed(packed_oft_blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    upper_rows, upper_cols = torch.triu_indices(block_size, block_size, offset=1, device=packed_oft_blocks.device)
    export_blocks = packed_oft_blocks.new_zeros(*packed_oft_blocks.shape[:-1], block_size, block_size)
    export_blocks[..., upper_rows, upper_cols] = packed_oft_blocks
    return export_blocks


def skew_oft_blocks(oft_blocks: torch.Tensor) -> tuple[torch.Tensor, str, int]:
    if oft_blocks.ndim == 4:
        return oft_blocks - oft_blocks.transpose(-1, -2), "full", int(oft_blocks.shape[-1])
    if oft_blocks.ndim == 3:
        block_size = infer_block_size_from_packed_size(int(oft_blocks.shape[-1]))
        upper_rows, upper_cols = torch.triu_indices(block_size, block_size, offset=1, device=oft_blocks.device)
        skew = oft_blocks.new_zeros(*oft_blocks.shape[:-1], block_size, block_size)
        skew[..., upper_rows, upper_cols] = oft_blocks
        skew[..., upper_cols, upper_rows] = -oft_blocks
        return skew, "packed", block_size
    raise ValueError(f"Unsupported oft_blocks rank {oft_blocks.ndim}, expected 3 or 4")


def tensor_scalar(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    return float(value.detach().float().item())


def summarize_series(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "min": float(sorted_values[0]),
        "p25": float(statistics.quantiles(sorted_values, n=4, method="inclusive")[0]),
        "median": float(statistics.median(sorted_values)),
        "p75": float(statistics.quantiles(sorted_values, n=4, method="inclusive")[2]),
        "max": float(sorted_values[-1]),
        "mean": float(statistics.fmean(sorted_values)),
    }


def summarize_tensor_values(values: torch.Tensor) -> Optional[dict[str, float]]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return None
    quantiles = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75], device=flat.device))
    return {
        "min": float(flat.min().item()),
        "p25": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p75": float(quantiles[2].item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
    }


def klein9b_group(name: str) -> str:
    if "single_blocks" in name:
        if name.endswith("_linear1"):
            return "single_linear1"
        if name.endswith("_linear2"):
            return "single_linear2"
    if "double_blocks" in name:
        if "img_attn_qkv" in name:
            return "double_img_attn_qkv"
        if "img_attn_proj" in name:
            return "double_img_attn_proj"
        if "txt_attn_qkv" in name:
            return "double_txt_attn_qkv"
        if "txt_attn_proj" in name:
            return "double_txt_attn_proj"
        if "img_mlp_0" in name:
            return "double_img_mlp_0"
        if "img_mlp_2" in name:
            return "double_img_mlp_2"
        if "txt_mlp_0" in name:
            return "double_txt_mlp_0"
        if "txt_mlp_2" in name:
            return "double_txt_mlp_2"
    return "ungrouped"


def classify_group(name: str, architecture: str) -> str:
    if architecture in ARCHITECTURE_GROUP_PROFILES:
        return klein9b_group(name)
    return "all"


def summarize_boft_file(path: str, architecture: str) -> dict[str, Any]:
    state_dict = load_state_dict(path)

    module_summaries: list[ModuleSummary] = []
    alpha_values: list[float] = []
    group_map: dict[str, list[ModuleSummary]] = defaultdict(list)
    global_rescale_values: list[torch.Tensor] = []
    global_rescale_delta_abs_values: list[torch.Tensor] = []

    for key, oft_blocks in state_dict.items():
        if not key.endswith(".oft_blocks"):
            continue

        module_name = key[: -len(".oft_blocks")]
        skew_blocks, representation, block_size = skew_oft_blocks(oft_blocks.to(torch.float32))
        q_norm = float(torch.linalg.vector_norm(skew_blocks).item())
        alpha_key = f"{module_name}.alpha"
        rescale_key = f"{module_name}.rescale"
        saved_alpha = tensor_scalar(state_dict.get(alpha_key))
        if saved_alpha is not None:
            alpha_values.append(saved_alpha)
        rescale_tensor = state_dict.get(rescale_key)
        rescale_params = 0 if rescale_tensor is None else int(rescale_tensor.numel())
        rescale_mean = None
        rescale_min = None
        rescale_max = None
        rescale_delta_abs_mean = None
        rescale_delta_abs_max = None
        if rescale_tensor is not None:
            flat_rescale = rescale_tensor.detach().float().reshape(-1)
            flat_delta_abs = (flat_rescale - 1.0).abs()
            global_rescale_values.append(flat_rescale)
            global_rescale_delta_abs_values.append(flat_delta_abs)
            rescale_mean = float(flat_rescale.mean().item())
            rescale_min = float(flat_rescale.min().item())
            rescale_max = float(flat_rescale.max().item())
            rescale_delta_abs_mean = float(flat_delta_abs.mean().item())
            rescale_delta_abs_max = float(flat_delta_abs.max().item())
        stored_params = int(oft_blocks.numel()) + rescale_params
        packed_params = int(oft_blocks.shape[0] * oft_blocks.shape[1] * (block_size * (block_size - 1) // 2)) + rescale_params
        export_params = int(
            export_oft_blocks_from_packed(oft_blocks, block_size).numel() if oft_blocks.ndim == 3 else oft_blocks.numel()
        )
        export_params += rescale_params

        summary = ModuleSummary(
            name=module_name,
            group=classify_group(module_name, architecture),
            q_norm=q_norm,
            representation=representation,
            oft_shape=tuple(int(dim) for dim in oft_blocks.shape),
            block_size=block_size,
            saved_alpha=saved_alpha,
            rescale_params=rescale_params,
            rescale_mean=rescale_mean,
            rescale_min=rescale_min,
            rescale_max=rescale_max,
            rescale_delta_abs_mean=rescale_delta_abs_mean,
            rescale_delta_abs_max=rescale_delta_abs_max,
            packed_params=packed_params,
            stored_params=stored_params,
            export_params=export_params,
        )
        module_summaries.append(summary)
        group_map[summary.group].append(summary)

    if not module_summaries:
        raise ValueError(f"No BOFT weights found in {path}")

    module_summaries.sort(key=lambda item: item.q_norm, reverse=True)

    global_q_norms = [summary.q_norm for summary in module_summaries]
    unique_saved_alpha = sorted(set(alpha_values))
    global_rescale_stats = None
    global_rescale_delta_abs_stats = None
    if global_rescale_values:
        global_rescale_stats = summarize_tensor_values(torch.cat(global_rescale_values))
        global_rescale_delta_abs_stats = summarize_tensor_values(torch.cat(global_rescale_delta_abs_values))

    groups = {}
    for group_name, modules in sorted(group_map.items()):
        q_norms = [module.q_norm for module in modules]
        group_rescale_stats = None
        group_rescale_delta_abs_stats = None
        group_rescale_values = []
        group_rescale_delta_abs_values = []
        for module in modules:
            module_name = module.name
            rescale_tensor = state_dict.get(f"{module_name}.rescale")
            if rescale_tensor is None:
                continue
            flat_rescale = rescale_tensor.detach().float().reshape(-1)
            group_rescale_values.append(flat_rescale)
            group_rescale_delta_abs_values.append((flat_rescale - 1.0).abs())
        if group_rescale_values:
            group_rescale_stats = summarize_tensor_values(torch.cat(group_rescale_values))
            group_rescale_delta_abs_stats = summarize_tensor_values(torch.cat(group_rescale_delta_abs_values))
        groups[group_name] = {
            "count": len(modules),
            "q_norm": summarize_series(q_norms),
            "rescale": group_rescale_stats,
            "rescale_delta_abs": group_rescale_delta_abs_stats,
            "packed_params": int(sum(module.packed_params for module in modules)),
            "stored_params": int(sum(module.stored_params for module in modules)),
            "export_params": int(sum(module.export_params for module in modules)),
        }

    validation = None
    expected_counts = ARCHITECTURE_GROUP_PROFILES.get(architecture)
    if expected_counts is not None:
        actual_counts = {group_name: len(group_map.get(group_name, [])) for group_name in expected_counts}
        validation = {
            "expected_counts": expected_counts,
            "actual_counts": actual_counts,
            "matches_expected": actual_counts == expected_counts,
        }

    return {
        "path": path,
        "module_count": len(module_summaries),
        "saved_alpha_values": unique_saved_alpha,
        "global_q_norm": summarize_series(global_q_norms),
        "global_rescale": global_rescale_stats,
        "global_rescale_delta_abs": global_rescale_delta_abs_stats,
        "packed_params_total": int(sum(module.packed_params for module in module_summaries)),
        "stored_params_total": int(sum(module.stored_params for module in module_summaries)),
        "export_params_total": int(sum(module.export_params for module in module_summaries)),
        "groups": groups,
        "top_modules": [asdict(module) for module in module_summaries],
        "validation": validation,
    }


def format_number(value: float) -> str:
    return f"{value:.6f}"


def format_params_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def print_text_report(report: dict[str, Any], top_n: int) -> None:
    print(f"Path: {report['path']}")
    print(f"Modules: {report['module_count']}")
    print(f"Saved alpha values: {report['saved_alpha_values']}")
    print(
        "Packed-equivalent params: "
        f"{report['packed_params_total']} ({format_params_millions(report['packed_params_total'])}), "
        "Stored params: "
        f"{report['stored_params_total']} ({format_params_millions(report['stored_params_total'])}), "
        "Export/full params: "
        f"{report['export_params_total']} ({format_params_millions(report['export_params_total'])})"
    )

    global_stats = report["global_q_norm"]
    print("Global q_norm:")
    print(
        "  "
        f"min={format_number(global_stats['min'])} "
        f"p25={format_number(global_stats['p25'])} "
        f"median={format_number(global_stats['median'])} "
        f"p75={format_number(global_stats['p75'])} "
        f"max={format_number(global_stats['max'])} "
        f"mean={format_number(global_stats['mean'])}"
    )

    global_rescale = report.get("global_rescale")
    global_rescale_delta_abs = report.get("global_rescale_delta_abs")
    if global_rescale is not None and global_rescale_delta_abs is not None:
        print("Global rescale:")
        print(
            "  "
            f"min={format_number(global_rescale['min'])} "
            f"p25={format_number(global_rescale['p25'])} "
            f"median={format_number(global_rescale['median'])} "
            f"p75={format_number(global_rescale['p75'])} "
            f"max={format_number(global_rescale['max'])} "
            f"mean={format_number(global_rescale['mean'])}"
        )
        print("Global |rescale - 1|:")
        print(
            "  "
            f"min={format_number(global_rescale_delta_abs['min'])} "
            f"p25={format_number(global_rescale_delta_abs['p25'])} "
            f"median={format_number(global_rescale_delta_abs['median'])} "
            f"p75={format_number(global_rescale_delta_abs['p75'])} "
            f"max={format_number(global_rescale_delta_abs['max'])} "
            f"mean={format_number(global_rescale_delta_abs['mean'])}"
        )

    validation = report.get("validation")
    if validation is not None:
        print(f"Klein-9B group validation: {validation['matches_expected']}")
        if not validation["matches_expected"]:
            for group_name, expected_count in validation["expected_counts"].items():
                actual_count = validation["actual_counts"].get(group_name, 0)
                print(f"  {group_name}: expected={expected_count} actual={actual_count}")

    print("Group stats:")
    for group_name, group_report in report["groups"].items():
        q_norm = group_report["q_norm"]
        line = (
            f"  {group_name}: count={group_report['count']} "
            f"mean={format_number(q_norm['mean'])} "
            f"min={format_number(q_norm['min'])} "
            f"median={format_number(q_norm['median'])} "
            f"max={format_number(q_norm['max'])} "
            f"packed={format_params_millions(group_report['packed_params'])} "
            f"stored={format_params_millions(group_report['stored_params'])} "
            f"export={format_params_millions(group_report['export_params'])}"
        )
        rescale = group_report.get("rescale")
        rescale_delta_abs = group_report.get("rescale_delta_abs")
        if rescale is not None and rescale_delta_abs is not None:
            line += (
                f" rescale_mean={format_number(rescale['mean'])}"
                f" rescale_min={format_number(rescale['min'])}"
                f" rescale_max={format_number(rescale['max'])}"
                f" |d|_mean={format_number(rescale_delta_abs['mean'])}"
                f" |d|_max={format_number(rescale_delta_abs['max'])}"
            )
        print(line)

    top_modules = report["top_modules"][:top_n]
    bottom_modules = list(reversed(report["top_modules"][-top_n:]))

    print(f"Top {len(top_modules)} modules by q_norm:")
    for module in top_modules:
        print(
            "  "
            f"{module['name']}: q_norm={format_number(module['q_norm'])} "
            f"group={module['group']} rep={module['representation']} "
            f"shape={tuple(module['oft_shape'])}"
        )

    print(f"Bottom {len(bottom_modules)} modules by q_norm:")
    for module in bottom_modules:
        print(
            "  "
            f"{module['name']}: q_norm={format_number(module['q_norm'])} "
            f"group={module['group']} rep={module['representation']} "
            f"shape={tuple(module['oft_shape'])}"
        )


def main() -> None:
    args = parse_args()
    report = summarize_boft_file(args.weights, args.architecture)
    if args.json:
        trimmed_report = dict(report)
        top_n = max(args.top_n, 0)
        trimmed_report["top_modules"] = report["top_modules"][:top_n]
        trimmed_report["bottom_modules"] = list(reversed(report["top_modules"][-top_n:]))
        print(json.dumps(trimmed_report, indent=2, sort_keys=True))
        return
    print_text_report(report, args.top_n)


if __name__ == "__main__":
    main()
