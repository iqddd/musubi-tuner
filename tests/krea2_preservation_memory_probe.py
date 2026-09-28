"""Opt-in short training probe for preservation memory with real optimizer state.

Uses a training TOML and its dataset, performs disposable optimizer updates, and
writes only a JSON memory report. Checkpoint saving, sampling and trackers are
disabled. --combined selects the former two-live-graphs schedule for comparison.
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import toml
import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser, setup_parser_common
from musubi_tuner.networks.lora import LoRANetwork
from musubi_tuner.training.trainer_base import NetworkTrainer


def main(cli):
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=False)
    config = toml.load(cli.config)
    values = {}
    for key, value in config.items():
        values.update(value if isinstance(value, dict) else {key: value})
    parser = krea2_setup_parser(setup_parser_common())
    args = parser.parse_args([], namespace=argparse.Namespace(**values))
    args.max_train_epochs = None
    args.max_train_steps = cli.steps
    args.output_dir = str(output)
    args.output_name = "memory_probe"
    args.log_with = args.logging_dir = None
    args.sample_every_n_steps = args.sample_every_n_epochs = None
    args.sample_at_first = False
    args.sample_prompts = None
    args.save_every_n_epochs = args.save_every_n_steps = None
    args.save_state = args.save_state_on_train_end = False
    args.resume = None
    args.dit_dtype = "bfloat16"
    args.vae_dtype = args.vae_dtype or "bfloat16"
    if not args.alpha_masked_output_preservation:
        raise ValueError("Enable alpha_masked_output_preservation in the input config")
    torch.set_num_threads(1)
    torch._inductor.config.compile_threads = 4
    report = {
        "config": str(cli.config),
        "schedule": "combined" if cli.combined else "sequential",
        "steps_requested": cli.steps,
        "optimizer": args.optimizer_type,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "attention_mode": args.alpha_masked_attention_mode,
        "microbatches": [],
        "updates": [],
        "status": "running",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    class ProbeTrainer(Krea2NetworkTrainer):
        def on_train_start(self, args, accelerator, network, transformer, optimizer):
            super().on_train_start(args, accelerator, network, transformer, optimizer)
            self.probe_optimizer = optimizer

        def process_batch_and_backward(self, *a, **kw):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            row = {"allocated_before_gib": torch.cuda.memory_allocated() / 2**30}
            report["microbatches"].append(row)
            try:
                method = (
                    NetworkTrainer.process_batch_and_backward if cli.combined else Krea2NetworkTrainer.process_batch_and_backward
                )
                loss, metrics = method(self, *a, **kw)
                row["loss"] = float(loss)
                row["metrics"] = {k: float(v) for k, v in metrics.items()}
                assert torch.isfinite(loss), "Non-finite loss"
                return loss, metrics
            finally:
                row["seconds"] = time.monotonic() - started
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                row["allocated_after_gib"] = torch.cuda.memory_allocated() / 2**30
                save()
                print("MEMORY_MICROBATCH", row, flush=True)

        def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
            super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)
            if sync_gradients:
                state_bytes = sum(
                    value.numel() * value.element_size()
                    for state in self.probe_optimizer.state.values()
                    for value in state.values()
                    if torch.is_tensor(value)
                )
                row = {
                    "step": global_step + 1,
                    "optimizer_state_gib": state_bytes / 2**30,
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                }
                report["updates"].append(row)
                save()
                print("MEMORY_UPDATE", row, flush=True)

    try:
        # The normal trainer still handles accumulation, autocast, compile and AdamW.
        # Its final checkpoint call is suppressed; these updates are diagnostic only.
        with patch.object(LoRANetwork, "save_weights", return_value=None):
            ProbeTrainer().train(args)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description=__doc__)
    cli_parser.add_argument("--config", required=True)
    cli_parser.add_argument("--output", required=True)
    cli_parser.add_argument("--steps", type=int, default=3)
    cli_parser.add_argument("--combined", action="store_true")
    main(cli_parser.parse_args())
