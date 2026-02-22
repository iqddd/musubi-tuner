import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def build_base_train_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    artifacts_dir = run_dir / "artifacts"
    cmd = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        str(args.num_cpu_threads_per_process),
        "--mixed_precision",
        args.mixed_precision,
        args.train_script,
        "--config_file",
        args.config_file,
        "--profile_capture_step",
        str(args.capture_step),
        "--profile_stop_step",
        str(args.stop_step),
        "--profile_artifacts_dir",
        str(artifacts_dir),
        "--profile_disable_sampling",
        "--profile_save_model_on_stop",
        "--profile_save_state_on_stop",
        "--profile_save_optimizer_summary_on_stop",
    ]

    if args.profiler == "torch":
        cmd.append("--profile_with_torch")
    else:
        cmd.append("--profile_with_cuda_profiler_api")
        cmd.append("--profile_emit_nvtx")

    if args.extra_train_args:
        cmd.extend(args.extra_train_args)

    return cmd


def wrap_with_gpu_profiler(args: argparse.Namespace, train_cmd: list[str], run_dir: Path) -> list[str]:
    if args.profiler == "nsys":
        out_prefix = run_dir / "nsys" / "report"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        return [
            "nsys",
            "profile",
            "--capture-range",
            "cudaProfilerApi",
            "--capture-range-end",
            "stop",
            "--trace",
            "cuda,nvtx,cublas,cudnn,osrt",
            "--sample",
            "none",
            "--force-overwrite",
            "true",
            "--output",
            str(out_prefix),
            *train_cmd,
        ]

    if args.profiler == "ncu":
        out_prefix = run_dir / "ncu" / "report"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        return [
            "ncu",
            "--profile-from-start",
            "off",
            "--target-processes",
            "all",
            "--set",
            "full",
            "--force-overwrite",
            "-o",
            str(out_prefix),
            *train_cmd,
        ]

    return train_cmd


def main():
    parser = argparse.ArgumentParser(description="Profiling bench launcher for FLUX.2 training run from config.")
    parser.add_argument("--config_file", required=True, help="Path to train config TOML")
    parser.add_argument(
        "--profiler",
        default="torch",
        choices=["torch", "nsys", "ncu"],
        help="Profiler backend",
    )
    parser.add_argument("--capture_step", type=int, default=5, help="Capture exactly this optimization step")
    parser.add_argument("--stop_step", type=int, default=12, help="Stop run and save artifacts at this step")
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_cpu_threads_per_process", type=int, default=1)
    parser.add_argument("--train_script", default="src/musubi_tuner/flux_2_train_network.py")
    parser.add_argument("--output_root", default="speedup_task/profile_runs", help="Root dir for bench runs")
    parser.add_argument("--dry_run", action="store_true", help="Only print command and write metadata")
    parser.add_argument("extra_train_args", nargs=argparse.REMAINDER, help="Extra args passed to train script")

    args = parser.parse_args()

    if args.capture_step > args.stop_step:
        raise ValueError("--capture_step must be <= --stop_step")

    ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
    run_name = f"{ts}_{args.profiler}_capture{args.capture_step:04d}_stop{args.stop_step:04d}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = build_base_train_cmd(args, run_dir)
    full_cmd = wrap_with_gpu_profiler(args, train_cmd, run_dir)
    if args.profiler in ("nsys", "ncu"):
        full_cmd = ["env", "MUSUBI_PROFILE_EMIT_NVTX=1", *full_cmd]

    command_text = shlex.join(full_cmd)
    (run_dir / "run_command.sh").write_text(command_text + "\n", encoding="utf-8")
    os.chmod(run_dir / "run_command.sh", 0o755)

    metadata = {
        "timestamp": ts,
        "run_name": run_name,
        "profiler": args.profiler,
        "capture_step": args.capture_step,
        "stop_step": args.stop_step,
        "config_file": args.config_file,
        "run_dir": str(run_dir),
        "command": full_cmd,
        "command_shell": command_text,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(command_text)
    print(f"run_dir={run_dir}")

    if args.dry_run:
        return

    completed = subprocess.run(full_cmd, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
