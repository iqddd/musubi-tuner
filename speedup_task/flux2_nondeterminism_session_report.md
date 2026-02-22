# Pipeline Artifact: FLUX.2 CUDA Nondeterminism Envelope and Significance Checks

## Scope of this artifact

This document summarizes the profiling/comparison work from this session around:

- establishing a practical CUDA nondeterminism envelope (`run1..run8`, seed=420),
- adding numeric significance output to the comparison script,
- validating how far new runs are from that envelope.

All runs below use:

- FLUX.2 LoRA training,
- `N=5` (`--profile_capture_step`),
- `M=12` (`--profile_stop_step`),
- AdamW with `learning_rate=1e-4`,
- comparison against `speedup_task/nondet_n5_m12_adamw/analysis_summary.json`.

## Baseline envelope (reference)

Reference set:

- `speedup_task/nondet_n5_m12_adamw/run1` ... `run8`
- built into `analysis_summary.json`

Key baseline facts:

- `run_count=8`
- `pair_count=28`
- `rng_unique_count=1` (same RNG hashes across baseline runs)

Key envelope limits (from `analysis_summary.json`):

- `pair_loss_abs_diff`: `p99=1.2884557247161866e-04`, `max=1.3172626495361328e-04`
- `pair_moving_loss_abs_diff`: `p99=1.9604265689849855e-05`, `max=2.078711986541748e-05`
- `.lora_down.weight::rel_l2`: `p99=0.0367337589108163`, `max=0.036777972755612565`
- `.lora_up.weight::rel_l2`: `p99=0.6219326257500009`, `max=0.6282450862501554`
- `.lora_effect_probe::rel_l2`: `p99=0.6242268350175315`, `max=0.6303311898109997`

## Tooling updates made in this session

Updated file:

- `speedup_task/compare_profile_artifacts.py`

Main changes:

- added numeric nondet assessment output:
  - `ratio_to_p99`, `ratio_to_max`, `outside_p99_pct`, `outside_max_pct`,
- added ensemble assessment against fixed baseline runs `run1..run8`:
  - outputs `p50/p75/p90` for each metric,
- made `--baseline` optional (ensemble-only mode works with `--candidate + --nondet_summary`),
- kept `.alpha` in functional effect computation (`scale = alpha / rank`) but removed standalone `.alpha` reporting,
- removed `max_abs` from regular console report,
- `--effect_device cuda` now honors explicit CUDA intent; `auto` suppresses noisy `torch.cuda.is_available()` warnings.

Hardcoded ensemble baseline (with in-code comment):

- `ENSEMBLE_BASELINE_RUNS = [run1..run8]`
- source: `speedup_task/nondet_n5_m12_adamw/run1..run8`

## Commands used in this session

### Ensemble comparison (current preferred mode)

```bash
.venv/bin/python speedup_task/compare_profile_artifacts.py \
  --candidate <run_dir> \
  --nondet_summary speedup_task/nondet_n5_m12_adamw/analysis_summary.json \
  --effect_device cuda
```

### Profiling run launcher

```bash
PATH=.venv/bin:$PATH .venv/bin/python speedup_task/flux2_profile_bench.py \
  --config_file speedup_task/config_nondet_adamw.toml \
  --profiler torch \
  --capture_step 5 \
  --stop_step 12 \
  --output_root <run_output_root>
```

## New runs and outcomes

## 1) Control check: seed=420, fp8 off

Run:

- `seed=420`

Ensemble result (p90 `ratio_to_max`):

- `current_loss_abs_diff`: `0.4618`
- `moving_average_loss_abs_diff`: `0.4658`
- `.lora_down.weight::rel_l2`: `0.9317`
- `.lora_up.weight::rel_l2`: `0.9286`
- `.lora_effect_probe::rel_l2`: `0.9148`

Interpretation:

- fully inside baseline nondet envelope.

## 2) Seed sensitivity check: seed=421, fp8 off

Run:

- `seed=421`

Ensemble result (p90 `ratio_to_max`):

- `current_loss_abs_diff`: `1394.0976`
- `moving_average_loss_abs_diff`: `1556.6883`
- `.lora_down.weight::rel_l2`: `38.4512`
- `.lora_up.weight::rel_l2`: `2.2675`
- `.lora_effect_probe::rel_l2`: `2.2351`

Interpretation:

- massively outside baseline nondet envelope (as expected for seed change on a short run).

## 3) FP8 check: seed=420, fp8_base=true, fp8_scaled=true

Config change used:

- `speedup_task/config_nondet_adamw.toml`
  - `seed = 420`
  - `fp8_base = true`
  - `fp8_scaled = true`

Run:

- `seed=420`

Ensemble result (p90 `ratio_to_max`):

- `current_loss_abs_diff`: `0.6370`
- `moving_average_loss_abs_diff`: `10.3094`
- `.lora_down.weight::rel_l2`: `1.0473`
- `.lora_up.weight::rel_l2`: `1.2861`
- `.lora_effect_probe::rel_l2`: `1.2689`

Interpretation:

- outside baseline nondet envelope,
- much weaker than seed=421 effect,
- still clearly significant on weight/effect metrics.

## Why short runs can amplify seed effects

The run horizon is short (`12` steps, before a full epoch pass).  
In this codebase, seed influences both stochastic training noise and data ordering:

- global seed is set before dataset and dataloader creation:
  - `src/musubi_tuner/hv_train_network.py` (`set_seed(args.seed)`),
- dataloader uses `shuffle=True`:
  - `src/musubi_tuner/hv_train_network.py`,
- dataset-group seed is generated from Python RNG after global seeding:
  - `src/musubi_tuner/dataset/config_utils.py`,
- bucket shuffling uses epoch-based seeded RNG:
  - `src/musubi_tuner/dataset/image_video_dataset.py`,
- training noise/timestep sampling includes random draws:
  - `src/musubi_tuner/hv_train_network.py`.

So `seed=420 -> 421` is not a tiny perturbation; it changes the optimization path from step 1.

## Practical interpretation rule used here

For a metric where higher is worse:

- `ratio_to_max <= 1.0`: inside observed nondet envelope,
- `ratio_to_max > 1.0`: outside envelope by `outside_max_pct`.

For robust decision-making, use ensemble quantiles (`p50/p75/p90`) rather than only worst-case.

## Baseline vs FP8 vs t [0,1000]

The table below uses `p90 ratio_to_max` against the same baseline envelope (`run1..run8`).

| Metric (`p90 ratio_to_max`) | Baseline Ensemble (reference) | FP8 (`fp8_base=true`, `fp8_scaled=true`) | `t in [0,1000]` (`preserve_distribution_shape` and `max_timestep` omitted) |
|---|---:|---:|---:|
| `current_loss_abs_diff` | `<=1` | `0.6370` | `306.2107` |
| `moving_average_loss_abs_diff` | `<=1` | `10.3094` | `153.1994` |
| `.lora_down.weight::rel_l2` | `<=1` | `1.0473` | `0.9616` |
| `.lora_up.weight::rel_l2` | `<=1` | `1.2861` | `0.9532` |
| `.lora_effect_probe::rel_l2` | `<=1` | `1.2689` | `0.9412` |

Interpretation:

- FP8 shows a clear out-of-envelope effect on weight/effect metrics (`rel_l2` > 1), with moderate magnitude.
- `t in [0,1000]` strongly shifts loss-level statistics, but weight/effect metrics remain inside the baseline envelope.
- In this short-run setup (`M=12`), FP8 appears to change LoRA update geometry more directly, while full-range timestep sampling primarily shifts scalar loss behavior.
