#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAST_OPS = {"aten::to", "aten::_to_copy", "aten::copy_"}

ELEMENTWISE_OP_PATTERN = re.compile(
    r"^aten::("
    r"add|sub|mul|div|pow|exp|expm1|log|log1p|sqrt|rsqrt|"
    r"sin|cos|tan|tanh|sigmoid|silu|gelu|relu|neg|abs|clamp|where|"
    r"gt|ge|lt|le|eq|ne|bitwise_|logical_|maximum|minimum|reciprocal|"
    r"floor|ceil|round|frac|fmod|remainder|erf|erfc|digamma|lgamma|"
    r"trunc|sign|copysign|atan|asin|acos|sinh|cosh|asinh|acosh|atanh|"
    r"xlogy|nan_to_num|clamp_min|clamp_max|masked_fill|fill_|zero_|"
    r"addcdiv|addcmul|lerp|softplus"
    r")(_|$)"
)


@dataclass(frozen=True)
class Interval:
    ts: float
    end: float
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H7 phase-1: source-level attribution from torch profiler chrome trace")
    parser.add_argument("--trace_json", type=Path, required=True, help="Path to stepXXXX.json chrome trace")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for csv/md outputs")
    parser.add_argument("--top_n", type=int, default=30, help="Top rows to include in markdown tables")
    return parser.parse_args()


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError("traceEvents is missing or malformed")
    return events


def is_elementwise_op(name: str) -> bool:
    return bool(ELEMENTWISE_OP_PATTERN.match(name))


def classify_op(name: str) -> str | None:
    if name in CAST_OPS:
        return "cast"
    if is_elementwise_op(name):
        return "elementwise"
    return None


def build_stage_intervals(events: list[dict[str, Any]]) -> list[Interval]:
    stages: list[Interval] = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "user_annotation":
            continue
        name = str(ev.get("name", ""))
        if not name.startswith("train/"):
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        stages.append(Interval(ts=ts, end=ts + dur, name=name))
    stages.sort(key=lambda x: x.ts)
    return stages


def stage_of_ts(ts: float, stages: list[Interval]) -> str:
    # train/* markers are sequential in this training step; linear scan is sufficient.
    for stage in stages:
        if stage.ts <= ts <= stage.end:
            return stage.name
    return "outside_train_markers"


def build_python_intervals(events: list[dict[str, Any]]) -> list[Interval]:
    out: list[Interval] = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "python_function":
            continue
        name = str(ev.get("name", ""))
        if ".py(" not in name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        out.append(Interval(ts=ts, end=ts + dur, name=name))
    out.sort(key=lambda x: x.ts)
    return out


def normalize_callsite(name: str) -> str:
    return name.strip()


def is_project_callsite(name: str) -> bool:
    prefixes = (
        "hv_train_network.py(",
        "flux_2_train_network.py(",
        "flux_2/",
        "src/musubi_tuner/",
    )
    return name.startswith(prefixes)


def extract_target_cpu_ops(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "cpu_op":
            continue
        name = str(ev.get("name", ""))
        op_group = classify_op(name)
        if op_group is None:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        ops.append({"ts": ts, "dur_us": dur, "op_name": name, "op_group": op_group})
    ops.sort(key=lambda x: x["ts"])
    return ops


def attribute_callsites(target_ops: list[dict[str, Any]], python_intervals: list[Interval]) -> None:
    # Sweep through time: for each target op, pick the deepest active python frame
    # approximated as the active frame with the largest start timestamp.
    py_idx = 0
    active: list[Interval] = []

    for op in target_ops:
        ts = op["ts"]
        while py_idx < len(python_intervals) and python_intervals[py_idx].ts <= ts:
            active.append(python_intervals[py_idx])
            py_idx += 1

        if active:
            active = [frame for frame in active if frame.end >= ts]

        if active:
            frame = max(active, key=lambda x: x.ts)
            op["callsite"] = normalize_callsite(frame.name)
        else:
            op["callsite"] = "UNKNOWN"


def aggregate_rows(target_ops: list[dict[str, Any]], stages: list[Interval]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_stage_callsite_op: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    by_stage_callsite: dict[tuple[str, str, str], dict[str, Any]] = {}

    for op in target_ops:
        stage = stage_of_ts(op["ts"], stages)
        op_name = op["op_name"]
        op_group = op["op_group"]
        callsite = op["callsite"]
        dur_us = op["dur_us"]

        key1 = (op_group, stage, op_name, callsite)
        row1 = by_stage_callsite_op.setdefault(
            key1,
            {
                "op_group": op_group,
                "stage": stage,
                "op_name": op_name,
                "callsite": callsite,
                "project_callsite": is_project_callsite(callsite),
                "calls": 0,
                "cpu_time_us": 0.0,
            },
        )
        row1["calls"] += 1
        row1["cpu_time_us"] += dur_us

        key2 = (op_group, stage, callsite)
        row2 = by_stage_callsite.setdefault(
            key2,
            {
                "op_group": op_group,
                "stage": stage,
                "callsite": callsite,
                "project_callsite": is_project_callsite(callsite),
                "calls": 0,
                "cpu_time_us": 0.0,
            },
        )
        row2["calls"] += 1
        row2["cpu_time_us"] += dur_us

    rows1 = sorted(by_stage_callsite_op.values(), key=lambda r: r["cpu_time_us"], reverse=True)
    rows2 = sorted(by_stage_callsite.values(), key=lambda r: r["cpu_time_us"], reverse=True)
    return rows1, rows2


def build_group_summary(events: list[dict[str, Any]], target_ops: list[dict[str, Any]], stages: list[Interval]) -> list[dict[str, Any]]:
    total_cpu_op_us = 0.0
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "cpu_op":
            total_cpu_op_us += float(ev.get("dur", 0.0))

    step_wall_us = 0.0
    for stage in stages:
        step_wall_us += stage.end - stage.ts

    grouped = defaultdict(lambda: {"calls": 0, "cpu_time_us": 0.0})
    for op in target_ops:
        g = grouped[op["op_group"]]
        g["calls"] += 1
        g["cpu_time_us"] += op["dur_us"]

    summary: list[dict[str, Any]] = []
    for op_group in ("cast", "elementwise"):
        cpu_time_us = grouped[op_group]["cpu_time_us"]
        calls = grouped[op_group]["calls"]
        summary.append(
            {
                "op_group": op_group,
                "calls": calls,
                "cpu_time_ms": cpu_time_us / 1000.0,
                "share_of_total_cpu_op_pct": (cpu_time_us / total_cpu_op_us * 100.0) if total_cpu_op_us else 0.0,
                "share_vs_step_wall_pct": (cpu_time_us / step_wall_us * 100.0) if step_wall_us else 0.0,
                "avg_us_per_call": (cpu_time_us / calls) if calls else 0.0,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], top_n: int) -> str:
    header = "| " + " | ".join(name for _, name in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows[:top_n]:
        values = []
        for key, _label in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                if "pct" in key:
                    values.append(f"{val:.3f}")
                elif "ms" in key:
                    values.append(f"{val:.3f}")
                else:
                    values.append(f"{val:.3f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    trace_path: Path,
    summary_rows: list[dict[str, Any]],
    op_rows: list[dict[str, Any]],
    callsite_rows: list[dict[str, Any]],
    top_n: int,
) -> None:
    cast_op_rows = [r for r in op_rows if r["op_group"] == "cast"]
    ewise_op_rows = [r for r in op_rows if r["op_group"] == "elementwise"]
    cast_project_rows = [r for r in cast_op_rows if r["project_callsite"]]
    ewise_project_rows = [r for r in ewise_op_rows if r["project_callsite"]]
    cast_callsite_rows = [r for r in callsite_rows if r["op_group"] == "cast"]
    ewise_callsite_rows = [r for r in callsite_rows if r["op_group"] == "elementwise"]

    lines: list[str] = []
    lines.append("# H7 Phase-1 Source Attribution")
    lines.append("")
    lines.append(f"Trace: `{trace_path}`")
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    lines.append(
        to_table(
            summary_rows,
            [
                ("op_group", "Group"),
                ("calls", "Calls"),
                ("cpu_time_ms", "CPU time (ms)"),
                ("avg_us_per_call", "Avg us/call"),
                ("share_of_total_cpu_op_pct", "Share of all cpu_op (%)"),
                ("share_vs_step_wall_pct", "Share vs train/* wall (%)"),
            ],
            top_n=10,
        )
    )
    lines.append("")
    lines.append("## Top Cast Callsites (project-only)")
    lines.append("")
    lines.append(
        to_table(
            cast_project_rows,
            [
                ("stage", "Stage"),
                ("op_name", "Op"),
                ("callsite", "Callsite"),
                ("calls", "Calls"),
                ("cpu_time_us", "CPU time (us)"),
            ],
            top_n=top_n,
        )
    )
    lines.append("")
    lines.append("## Top Elementwise Callsites (project-only)")
    lines.append("")
    lines.append(
        to_table(
            ewise_project_rows,
            [
                ("stage", "Stage"),
                ("op_name", "Op"),
                ("callsite", "Callsite"),
                ("calls", "Calls"),
                ("cpu_time_us", "CPU time (us)"),
            ],
            top_n=top_n,
        )
    )
    lines.append("")
    lines.append("## Top Cast Callsites (all)")
    lines.append("")
    lines.append(
        to_table(
            cast_callsite_rows,
            [
                ("stage", "Stage"),
                ("callsite", "Callsite"),
                ("calls", "Calls"),
                ("cpu_time_us", "CPU time (us)"),
            ],
            top_n=top_n,
        )
    )
    lines.append("")
    lines.append("## Top Elementwise Callsites (all)")
    lines.append("")
    lines.append(
        to_table(
            ewise_callsite_rows,
            [
                ("stage", "Stage"),
                ("callsite", "Callsite"),
                ("calls", "Calls"),
                ("cpu_time_us", "CPU time (us)"),
            ],
            top_n=top_n,
        )
    )
    lines.append("")
    lines.append(
        f"Generated from {len(op_rows)} aggregated (group, stage, op, callsite) rows "
        f"and {len(cast_op_rows) + len(ewise_op_rows)} total rows across cast+elementwise."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = load_trace_events(args.trace_json)
    stages = build_stage_intervals(events)
    python_intervals = build_python_intervals(events)
    target_ops = extract_target_cpu_ops(events)

    attribute_callsites(target_ops, python_intervals)
    op_rows, callsite_rows = aggregate_rows(target_ops, stages)
    summary_rows = build_group_summary(events, target_ops, stages)

    for row in op_rows:
        row["cpu_time_ms"] = row["cpu_time_us"] / 1000.0
        row["avg_us_per_call"] = row["cpu_time_us"] / row["calls"] if row["calls"] else 0.0
    for row in callsite_rows:
        row["cpu_time_ms"] = row["cpu_time_us"] / 1000.0
        row["avg_us_per_call"] = row["cpu_time_us"] / row["calls"] if row["calls"] else 0.0

    summary_csv = args.output_dir / "h7_phase1_group_summary.csv"
    op_csv = args.output_dir / "h7_phase1_op_stage_callsite.csv"
    callsite_csv = args.output_dir / "h7_phase1_stage_callsite.csv"
    report_md = args.output_dir / "h7_phase1_attribution_report.md"

    write_csv(
        summary_csv,
        summary_rows,
        [
            "op_group",
            "calls",
            "cpu_time_ms",
            "avg_us_per_call",
            "share_of_total_cpu_op_pct",
            "share_vs_step_wall_pct",
        ],
    )
    write_csv(
        op_csv,
        op_rows,
        [
            "op_group",
            "stage",
            "op_name",
            "callsite",
            "project_callsite",
            "calls",
            "cpu_time_us",
            "cpu_time_ms",
            "avg_us_per_call",
        ],
    )
    write_csv(
        callsite_csv,
        callsite_rows,
        [
            "op_group",
            "stage",
            "callsite",
            "project_callsite",
            "calls",
            "cpu_time_us",
            "cpu_time_ms",
            "avg_us_per_call",
        ],
    )

    write_markdown_report(report_md, args.trace_json, summary_rows, op_rows, callsite_rows, args.top_n)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {op_csv}")
    print(f"Saved: {callsite_csv}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
