# FLUX.2: Step Time Optimization Hypotheses

Date recorded: 2026-02-22

## Hypotheses Status

| ID | Hypothesis | What will be tested | Expected effect | Status |
|---|---|---|---|---|
| H1 | Remove repeated `.to()` calls for rotary embeddings inside attention | Cache `freqs_cos/freqs_sin` on the target device/dtype and avoid per-call transfers | Medium/High | discarded (tested, no measurable speedup) |
| H2 | Reduce redundant casting in `call_dit` | Avoid unnecessary `to(..., dtype=...)` for `latents/noisy_model_input`, rely on autocast | Medium | discarded (tested, no speedup) |
| H3 | Enable `torch.compile` for training runs | Run A/B with `--compile` and tune `compile_mode` | Medium/High | not tested |
| H4 | Disable `gradient_checkpointing` (or find a compromise) | Compare step time with `gradient_checkpointing=true/false` while tracking VRAM | High | discarded (not feasible on 32 GB VRAM setup) |
| H5 | Vectorize timestep sampling with `preserve_distribution_shape=true` | Remove Python loops/per-element filtering while preserving current sampling math | Low/Medium | discarded (tested, no speedup) |
| H6 | Improve DataLoader/CPU->GPU pipeline | Test `num_workers`, `pin_memory`, batch transfer path, and cache-read contribution | Low/Medium (may grow after compute-side speedups) | not tested |
| H7 | Reduce per-step launch overhead | Reduce number of small kernel launches and host overhead (per profiler: `cudaLaunchKernel`, `Command Buffer Full`) | Medium | in progress (phase-3D steady-state positive, needs multi-step confirmation) |
| H8 | Add step-time metrics to profiling artifacts | Write `step_wall_ms/fwd_bwd_ms/data_wait_ms` to `run_result.json` and compare automatically | Infrastructure impact (faster optimization loop) | not tested |

## TODO

- H7 phase-3 (candidate A): reduce cast churn in `RMSNorm.forward` / `apply_rope` and run A/B on identical capture settings.
- H7 phase-3D: repeat A/B on additional capture windows (e.g., step 18/20/22) for the new narrow-compile profile (`dynamic=1`, `force_contiguous=1`, raised dynamo limits) and confirm variance envelope.
- H7 phase-3 (candidate B) with bucket prewarm + persistent cache + guard diagnostics stayed slower; keep this branch deprioritized unless guard failures (`grad_mode`/stride) are explicitly solved.
- For any next candidate: validate against nondeterminism envelope and compare `step_wall`, `Command Buffer Full`, `cudaLaunchKernel`.
- H3 smoke-test: run conservative `--compile` A/B (`compile_mode=default`, then `reduce-overhead`) with OOM guardrails and compare step wall + launch metrics.

## Verification Log

### H2 check (2026-02-22)

Goal: verify whether removing forced dtype cast in `call_dit` improves step performance.

Method:
- Baseline run: current code path.
- Candidate run: temporary local patch for test only (skip forced dtype cast, keep device transfer).
- Both runs used:
- `speedup_task/config_nondet_adamw.toml`
- `--profile_capture_step 5 --profile_stop_step 12 --profile_with_torch`
- profiling output roots:
- `speedup_task/h2_cast_check/baseline/...`
- `speedup_task/h2_cast_check/candidate/...`

Observed profiler deltas (`step00000005.txt`):
- `Self CPU time total`: `9.978s -> 10.028s` (worse, `+0.50%`)
- `Self CUDA time total`: `5.630s -> 5.657s` (worse, `+0.48%`)
- `aten::to` CPU total: `680.914ms -> 683.703ms` (worse)
- `aten::_to_copy` CPU total: `679.199ms -> 681.992ms` (worse)
- `aten::copy_` CPU total: `824.534ms -> 827.932ms` (worse)
- `cudaLaunchKernel` self CPU: `2.781s -> 2.793s` (worse)
- `Command Buffer Full` self CPU: `5.079s -> 5.105s` (worse)

Behavioral sanity:
- `run_result.json` (seed, RNG hashes, loss-level stats) remained consistent.
- Baseline vs candidate stayed inside the existing nondeterminism envelope (using `compare_profile_artifacts.py` with `analysis_summary.json`).

Conclusion:
- H2 is discarded for now as a speedup direction for this setup.

### Optimizer/zero_grad share check with `record_function` (2026-02-22)

Goal: measure optimizer and zero_grad contribution to one captured train step.

Instrumentation added (for this check):
- `train/optimizer_step`
- `train/lr_scheduler_step`
- `train/optimizer_zero_grad`

Run:
- `speedup_task/flux2_profile_bench.py`
- `--config_file speedup_task/config_nondet_adamw.toml`
- `--profiler torch --capture_step 5 --stop_step 12`
- Output root: `speedup_task/record_function_check`

Profile artifact used:
- `speedup_task/record_function_check/20260222195318_torch_capture0005_stop0012/artifacts/20260222195324_capture00000005_stop00000012/torch_profiler/rank00/step00000005.{txt,json}`

Measured values:
- Step wall span from trace events: `5648.581 ms`
- `train/optimizer_step` CPU duration: `3.565 ms` (`0.063%` of step wall)
- `train/optimizer_step` CUDA duration (from profiler table `Optimizer.step#AdamW.step`): `3.975 ms` (`~0.07%` of `Self CUDA time total = 5.631 s`)
- `train/optimizer_zero_grad` CPU duration: `0.207 ms` (`0.0037%` of step wall)
- `train/optimizer_zero_grad` CUDA launches inside marker: `0` `cudaLaunchKernel` calls (for this capture)

Conclusion:
- Optimizer update and zero_grad are both negligible for current step time.
- No evidence that optimizer/zero_grad are first-order bottlenecks in this configuration.

### H4 decision (2026-02-22)

Decision:
- H4 is removed from active optimization directions for this setup.

Reason:
- Disabling `gradient_checkpointing` is not feasible/practical for the current hardware target (`32 GB VRAM`, not `100 GB`).

### H5 check (2026-02-22)

Goal: verify whether vectorizing rejection filtering in `preserve_distribution_shape=true` reduces step time.

Method:
- Baseline run: current rejection loop (`for t_i in t`).
- Candidate run: vectorized filtering (`valid_t = t[(t >= t_min) & (t <= t_max)]`) with the same rejection logic and fallback behavior.
- Both runs used:
- `speedup_task/config_nondet_adamw.toml`
- `--profile_capture_step 5 --profile_stop_step 12 --profile_with_torch`
- profiling output roots:
- `speedup_task/h5_check/baseline/...`
- `speedup_task/h5_check/candidate/...`

Observed profiler deltas (`step00000005.txt`):
- `Self CPU time total`: `9.983s -> 10.009s` (worse, `+0.26%`)
- `Self CUDA time total`: `5.630s -> 5.643s` (worse, `+0.23%`)
- Step wall span from trace events: `5648.454ms -> 5661.554ms` (worse, `+0.23%`)
- `Command Buffer Full` self CPU: `5.083s -> 5.098s` (worse)
- `cudaLaunchKernel` self CPU: `2.785s -> 2.791s` (worse)
- `cudaLaunchKernel` calls: `7752 -> 7748` (effectively unchanged)
- `aten::to` CPU total: `681.945ms -> 683.456ms` (worse)
- `aten::_to_copy` CPU total: `680.215ms -> 681.676ms` (worse)
- `aten::copy_` CPU total: `825.857ms -> 827.835ms` (worse)

Behavioral sanity:
- `run_result.json` (loss-level stats, step/epoch, optimizer summary) remained consistent.
- Baseline vs candidate stayed within the existing nondeterminism envelope using `speedup_task/compare_profile_artifacts.py` with `speedup_task/nondet_n5_m12_adamw/analysis_summary.json`.

Conclusion:
- H5 is discarded for now as a speedup direction for this setup.
- The tested vectorization did not reduce step time and was reverted from code after verification.

### H1 check (2026-02-22)

Goal: verify whether reducing rotary-embedding overhead gives measurable step-time gain in the active FLUX.2 setup.

Method:
- Baseline run: reused from `speedup_task/h5_check/baseline/20260222201440_torch_capture0005_stop0012`.
- Candidate run: tested precomputed/cached rotary embedding path in FLUX.2 forward, then profiled with the same capture settings.
- Candidate artifacts:
- `speedup_task/h1_check/candidate/20260222203300_torch_capture0005_stop0012/.../step00000005.{txt,json}`
- Both runs used:
- `speedup_task/config_nondet_adamw.toml`
- `--profile_capture_step 5 --profile_stop_step 12 --profile_with_torch`

Observed profiler deltas (`step00000005`):
- `Self CPU time total`: `9.983s -> 10.001s` (`+0.18%`)
- `Self CUDA time total`: `5.630s -> 5.637s` (`+0.12%`)
- Step wall span from trace events: `5648.454ms -> 5655.585ms` (`+0.13%`)
- `cudaLaunchKernel` calls: `7752 -> 7692` (`-60`, about `-0.77%`)
- `cudaLaunchKernel` self CPU: `2.785s -> 2.790s` (`+0.18%`)
- `Command Buffer Full` self CPU: `5.083s -> 5.096s` (`+0.26%`)
- `aten::to/_to_copy/copy_` CPU totals: small increases (~`+0.19%` to `+0.23%`)

Behavioral sanity:
- `compare_profile_artifacts.py` with `speedup_task/nondet_n5_m12_adamw/analysis_summary.json` shows candidate remains inside nondeterminism envelope.

Conclusion:
- Fewer `cudaLaunchKernel` calls did not translate into faster step time.
- H1 is discarded for now as a speedup direction in this configuration.
- Tested H1 patch was reverted from code after verification.

### H7 phase-0 (2026-02-22)

Goal: add source-level attribution tooling before implementing H7 optimizations.

Implemented:
- Enabled `torch.profiler(..., with_stack=True)` for profiling runs.
- Added `record_function` ranges around major train-step sections:
- `train/batch_prep`
- `train/timestep_sampling_and_noisy_input`
- `train/loss_weighting`
- `train/dit_forward`
- `train/loss_compute`
- `train/backward`
- `train/grad_sync_and_clip`
- `train/optimizer_step`
- `train/lr_scheduler_step`
- `train/optimizer_zero_grad`
- `train/sampling_and_checkpoint_hooks`
- `train/step_logging`

Verification run:
- `speedup_task/flux2_profile_bench.py --config_file speedup_task/config_nondet_adamw.toml --profiler torch --capture_step 5 --stop_step 12 --output_root speedup_task/h7_phase0_check`
- Artifacts:
- `speedup_task/h7_phase0_check/20260222204605_torch_capture0005_stop0012/artifacts/20260222204612_capture00000005_stop00000012/torch_profiler/rank00/step00000005.{txt,json}`

Observed:
- Trace metadata confirms stack collection: `"with_stack": 1`.
- `train/*` ranges are present in trace (`user_annotation` and `gpu_user_annotation`).
- `python_function` events with file:line callsites are present in trace (e.g., `hv_train_network.py(...)`, `flux_2_train_network.py(...)`).

Conclusion:
- H7 phase-0 instrumentation is complete and usable for source-level hotspot attribution.
- Next H7 step is to extract top `aten::to/_to_copy/copy_` and element-wise callsites from this capture and rank by impact.

### H7 phase-1 (2026-02-22)

Goal: map cast/element-wise hotspots to concrete Python callsites and train-step stages.

Method:
- Reused phase-0 trace with stacks:
- `speedup_task/h7_phase0_check/20260222204605_torch_capture0005_stop0012/artifacts/20260222204612_capture00000005_stop00000012/torch_profiler/rank00/step00000005.json`
- Added analysis script:
- `speedup_task/h7_phase1_trace_attribution.py`
- Output artifacts:
- `speedup_task/h7_phase1_analysis/h7_phase1_group_summary.csv`
- `speedup_task/h7_phase1_analysis/h7_phase1_op_stage_callsite.csv`
- `speedup_task/h7_phase1_analysis/h7_phase1_stage_callsite.csv`
- `speedup_task/h7_phase1_analysis/h7_phase1_attribution_report.md`

Measured (single captured step):
- Cast ops (`aten::to/_to_copy/copy_`): `7038` calls, `2168.920 ms` summed CPU duration.
- Element-wise ops: `5495` calls, `2084.233 ms` summed CPU duration.
- Stage split:
- Cast in `train/backward`: `4709` calls, `1738.350 ms`; in `train/dit_forward`: `2318` calls, `430.452 ms`.
- Element-wise in `train/backward`: `3658` calls, `1838.633 ms`; in `train/dit_forward`: `1579` calls, `245.192 ms`.
- Top project callsites (cast + element-wise, aggregated):
- `train/backward`: `flux_2/flux2_models.py(969): forward` -> `374.575 ms`
- `train/backward`: `flux_2/flux2_models.py(1012): apply_rope` -> `236.125 ms`
- `train/dit_forward`: `flux_2/flux2_models.py(1012): apply_rope` -> `153.192 ms`
- `train/dit_forward`: `flux_2/flux2_models.py(969): forward` -> `145.260 ms`

Conclusion:
- H7 phase-1 confirms the main cast/element-wise pressure is concentrated in backward, with strongest project-local hotspots around `RMSNorm.forward` and `apply_rope`.
- Best next step is phase-2 targeted instrumentation in these two locations before applying a minimal optimization patch.

### H7 phase-2 (2026-02-22)

Goal: decompose `RMSNorm.forward` and `apply_rope` into narrow sub-sections (`cast_in`, `core`, `cast_out`) and quantify their contribution.

Instrumentation added:
- `src/musubi_tuner/flux_2/flux2_models.py`
- `h7/rmsnorm/cast_in`
- `h7/rmsnorm/core`
- `h7/rmsnorm/cast_out_scale`
- `h7/apply_rope/cast_in`
- `h7/apply_rope/core`
- `h7/apply_rope/cast_out`

Run:
- `speedup_task/flux2_profile_bench.py --config_file speedup_task/config_nondet_adamw.toml --profiler torch --capture_step 5 --stop_step 12 --output_root speedup_task/h7_phase2_check`
- Artifact:
- `speedup_task/h7_phase2_check/20260222212950_torch_capture0005_stop0012/artifacts/20260222212956_capture00000005_stop00000012/torch_profiler/rank00/step00000005.{txt,json}`

Analysis tooling and outputs:
- Script: `speedup_task/h7_phase2_marker_analysis.py`
- `speedup_task/h7_phase2_analysis/h7_phase2_marker_cpu_summary.csv`
- `speedup_task/h7_phase2_analysis/h7_phase2_marker_group_breakdown.csv`
- `speedup_task/h7_phase2_analysis/h7_phase2_marker_op_breakdown.csv`
- `speedup_task/h7_phase2_analysis/h7_phase2_marker_coverage.csv`
- `speedup_task/h7_phase2_analysis/h7_phase2_marker_report.md`

Measured (single captured step):
- Marker CPU totals:
- `h7/rmsnorm/core`: `225.432 ms`
- `h7/apply_rope/core`: `201.238 ms`
- `h7/rmsnorm/cast_out_scale`: `119.307 ms`
- `h7/rmsnorm/cast_in`: `34.675 ms`
- `h7/apply_rope/cast_in + cast_out`: `66.668 ms`
- Combined `h7/*` marker CPU duration: `647.320 ms` (about `12.15%` of `train/*` marker wall span in this capture).
- Coverage of global cast/element-wise costs by H7 markers:
- Cast: `449.645 ms` inside H7 out of `2140.085 ms` total cast CPU (`21.01%`).
- Element-wise: `499.737 ms` inside H7 out of `2073.098 ms` total element-wise CPU (`24.11%`).
- Top ops inside H7 markers:
- `h7/rmsnorm/core` (backward): `aten::rsqrt` (`82.200 ms`), `aten::pow` (`48.573 ms`).
- `h7/apply_rope/core` (backward): `aten::mul + aten::add` (`129.422 ms` together).

Conclusion:
- Phase-2 confirms that most actionable H7 pressure inside these hotspots is in `core` math sections (especially backward), not only in `cast_in/out`.
- Next practical step is a minimal phase-3 code change targeting these `core` paths first, then A/B validation.

### H7 phase-3B (2026-02-22): narrow `torch.compile` on hotspots

Goal: test whether compiling only `RMSNorm` and `apply_rope` reduces step-time/launch overhead without full block compile.

Code changes for candidate:
- `src/musubi_tuner/flux_2/flux2_models.py`
- Added env-gated narrow compile path:
- `MUSUBI_H7_NARROW_COMPILE=1`
- `h7/rmsnorm/compiled` marker path for compiled RMSNorm hotpath
- `h7/apply_rope/compiled` marker path for compiled apply_rope hotpath
- Eager path remains unchanged when env flag is off.

Runs:
- Baseline (same code, compile disabled):
- `speedup_task/h7_phase3b_check/baseline/20260222214506_torch_capture0005_stop0012/.../step00000005.{json,txt}`
- Candidate attempt #1 (`MUSUBI_H7_NARROW_COMPILE=1`, reduce-overhead):
- failed with OOM before capture artifact creation (`speedup_task/h7_phase3b_check/candidate/20260222214639_torch_capture0005_stop0012`)
- Candidate attempt #2 (tuned):
- env: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`, `MUSUBI_H7_NARROW_COMPILE=1`, `MUSUBI_H7_NARROW_COMPILE_MODE=default`, `MUSUBI_H7_NARROW_COMPILE_DYNAMIC=0`
- artifact: `speedup_task/h7_phase3b_check/candidate_tuned/20260222214741_torch_capture0005_stop0012/.../step00000005.{json,txt}`
- comparison outputs:
- `speedup_task/h7_phase3b_analysis/h7_phase3b_compare.md`
- `speedup_task/h7_phase3b_analysis/h7_phase3b_compare.csv`

Observed A/B (baseline vs candidate_tuned):
- `train_marker_wall_ms`: `5326.610 -> 8638.977` (`+62.19%`, worse)
- `Self CPU time total`: `10.167s -> 14.420s` (`+41.83%`, worse)
- `Self CUDA time total`: `5.632s -> 5.678s` (`+0.82%`, effectively flat/worse)
- Compile warning during run: `torch._dynamo hit config.recompile_limit (8)` for `_rmsnorm_hotpath` due shape mismatch (`512` vs `3191`), indicating unstable shape specialization.
- Candidate markers show large compiled-section CPU durations:
- `h7/rmsnorm/compiled`: `768.266 ms`
- `h7/apply_rope/compiled`: `1381.469 ms`

Conclusion:
- In this setup, narrow `torch.compile` for these two hotspots does not improve step time and significantly worsens wall time.
- Primary suspected reason: recompilation/shape instability under current bucketed sequence lengths.
- Candidate B is not a good next optimization direction right now.

### H7 phase-3C (2026-02-22): bucket prewarm + persistent cache + guard diagnostics

Goal: retry candidate B with explicit prewarm for all image buckets and persistent compile caches, then validate guard/recompile reasons.

Code changes:
- `src/musubi_tuner/flux_2/flux2_models.py`
- Added optional H7 prewarm for narrow compile hotpaths:
- `MUSUBI_H7_NARROW_COMPILE_PREWARM=1`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM_BUCKETS=...`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM_IMAGE_SEQLENS=...` (alternative input format)
- `MUSUBI_H7_NARROW_COMPILE_PREWARM_TXT_SEQLEN=512`

Run setup:
- Candidate run:
- `speedup_task/h7_phase3c_check/candidate_prewarm/20260222221423_torch_capture0005_stop0012/.../step00000005.{json,txt}`
- Env:
- `TORCHINDUCTOR_CACHE_DIR=speedup_task/h7_phase3c_cache/inductor`
- `TRITON_CACHE_DIR=speedup_task/h7_phase3c_cache/triton`
- `TORCH_LOGS=recompiles,guards`
- `MUSUBI_H7_NARROW_COMPILE=1`
- `MUSUBI_H7_NARROW_COMPILE_MODE=default`
- `MUSUBI_H7_NARROW_COMPILE_DYNAMIC=0`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM=1`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM_BUCKETS=624x1104,688x992,752x912,832x832,912x752,992x688`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM_VERBOSE=1`
- Persistent cache populated:
- `speedup_task/h7_phase3c_cache/inductor` (`11M`, 88 files)
- `speedup_task/h7_phase3c_cache/triton` (`8.3M`, 489 files)

Prewarm confirmation from run log:
- `[H7 prewarm] RMSNorm warmed seq_lens=[512, 2666, 2679, 2691, 2704, 3178, 3191, 3203, 3216]`
- `[H7 prewarm] apply_rope warmed seq_lens=[3178, 3191, 3203, 3216]`

Measured A/B (baseline vs phase-3C candidate):
- Comparison output: `speedup_task/h7_phase3c_analysis/h7_phase3b_compare.md`
- `train_marker_wall_ms`: `5326.610 -> 6342.061` (`+19.06%`, still worse)
- `Self CPU time total`: `10.167s -> 10.813s` (`+6.35%`, worse)
- `Self CUDA time total`: `5.632s -> 5.584s` (`-0.85%`, slightly better but not enough)
- `cudaLaunchKernel` calls: `7752 -> 7752` (unchanged)

Delta vs previous tuned candidate (phase-3B #2):
- Comparison output: `speedup_task/h7_phase3c_analysis/tuned_vs_prewarm/h7_phase3b_compare.md`
- `train_marker_wall_ms`: `8638.977 -> 6342.061` (`-26.59%`, improved vs previous candidate)
- Still slower than baseline.

Recompile/guard findings (`TORCH_LOGS=recompiles,guards`):
- `Recompiling function _rmsnorm_hotpath`: 8 times.
- `Recompiling function _apply_rope_hotpath`: 8 times.
- `torch._dynamo hit config.recompile_limit (8)` for both hotpaths.
- Guard failures are not only shape mismatches:
- `GLOBAL_STATE changed: grad_mode`
- stride mismatches (e.g., tensor stride at index 1 in `_apply_rope_hotpath`)
- This explains why prewarm+cache reduced damage vs phase-3B #2 but still did not remove compile churn.

Additional sub-attempt (phase-3C v2, same date):
- Tried a grad-aware/stride-aware prewarm variant to reduce guard misses further.
- Run failed at step 0 with checkpointing error:
- `torch.utils.checkpoint.CheckpointError: A different number of tensors was saved during the original forward and recomputation.`
- This variant was rolled back as unsafe for current `gradient_checkpointing` setup.

Conclusion:
- Points (1)-(3) were implemented and validated.
- Result remains negative for step wall: candidate is still slower than eager baseline.
- This conclusion applies to the prewarm/cache branch only.

### H7 phase-3D (2026-02-22): narrow-compile stability fix (no prewarm, dynamic shapes)

Goal: retry narrow compile with guard-stability controls, without prewarm, and validate on a later capture window (`step20`) to avoid compile warmup noise.

Run setup:
- Baseline:
- `speedup_task/h7_phase3c_check/narrow_fix/baseline/20260222223343_torch_capture0020_stop0030/.../step00000020.{json,txt}`
- Candidate:
- `speedup_task/h7_phase3c_check/narrow_fix/candidate/20260222223700_torch_capture0020_stop0030/.../step00000020.{json,txt}`
- Candidate env:
- `MUSUBI_H7_NARROW_COMPILE=1`
- `MUSUBI_H7_NARROW_COMPILE_MODE=default`
- `MUSUBI_H7_NARROW_COMPILE_DYNAMIC=1`
- `MUSUBI_H7_NARROW_COMPILE_FORCE_CONTIGUOUS=1`
- `MUSUBI_H7_NARROW_COMPILE_RECOMPILE_LIMIT=64`
- `MUSUBI_H7_NARROW_COMPILE_ACCUM_RECOMPILE_LIMIT=2048`
- `MUSUBI_H7_NARROW_COMPILE_PREWARM=0`
- Comparison artifact:
- `speedup_task/h7_phase3c_analysis/narrow_fix_step20/h7_phase3b_compare.md`
- Follow-up code defaults aligned with this profile in `src/musubi_tuner/flux_2/flux2_models.py` (still overrideable via env).

Observed A/B (`step00000020`):
- `train_marker_wall_ms`: `5355.079 -> 5182.288` (`-3.23%`, better)
- `Self CPU time total`: `10.218s -> 8.798s` (`-13.90%`)
- `Self CUDA time total`: `5.661s -> 4.963s` (`-12.33%`)
- `cudaLaunchKernel` calls: `7742 -> 4254` (`-45.05%`)
- `Command Buffer Full` calls: `5564 -> 470` (`-91.55%`)
- `cast` calls: `7035 -> 4091` (`-41.85%`)

Compiler stability signals:
- Candidate log (`speedup_task/h7_phase3c_check/narrow_fix_candidate_run.log`) has no `recompile`/`recompile_limit` entries.
- `CompiledFxGraph` appears in profile (`336` total calls on captured step), indicating compiled paths are active and reused.

Conclusion:
- Narrow compile is no longer blocked in this configuration.
- The combination `dynamic=1 + force_contiguous + higher dynamo limits + no prewarm` is currently the first H7 variant with measured step-time win.
- Next required validation is variance control across multiple nearby capture steps before finalizing this as default.

### H7 phase-3E (2026-02-22): `dynamic=0` check against `dynamic=1`

Goal: explicitly test `MUSUBI_H7_NARROW_COMPILE_DYNAMIC=0` vs current working profile (`dynamic=1`) on the same capture window.

Run setup:
- Baseline (`dynamic=1`):
- `speedup_task/h7_phase3e_dynamic0_check/dyn1/20260222230641_torch_capture0020_stop0030/.../step00000020.{json,txt}`
- Candidate (`dynamic=0`):
- `speedup_task/h7_phase3e_dynamic0_check/dyn0/20260222230959_torch_capture0020_stop0030/.../step00000020.{json,txt}`
- Common settings:
- `MUSUBI_H7_NARROW_COMPILE=1`
- `MUSUBI_H7_NARROW_COMPILE_MODE=default`
- `MUSUBI_H7_NARROW_COMPILE_FORCE_CONTIGUOUS=1`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`
- Comparison artifact:
- `speedup_task/h7_phase3e_dynamic0_analysis/h7_phase3b_compare.md`

Observed A/B (`step00000020`):
- `train_marker_wall_ms`: `5177.174 -> 5087.784` (`-1.73%`)
- `self_cpu_total_s`: `8.785 -> 8.773` (`-0.14%`)
- `self_cuda_total_s`: `4.952 -> 4.875` (`-1.55%`)
- `cudaLaunchKernel_calls`: `4254 -> 4254` (no change)
- `cast_calls`: `4091 -> 4091` (no change)
- `command_buffer_full_calls`: `467 -> 230` (`-50.75%`)
- `h7_apply_rope_compiled_ms`: `26.912 -> 380.654` (large regression inside compiled hotpath)

Stability signals:
- `dynamic=0` run printed multiple `_rmsnorm_hotpath` recompiles with `size mismatch` on varying sequence lengths during early steps.
- Compiled graph surface increased from 4 to 8 graph IDs in `step00000020.txt` (same total calls `336`), indicating heavier graph fragmentation under `dynamic=0`.
- Candidate run ended with `No space left on device` at final `save_state`, but `step00000020` profiler artifacts were successfully written before failure.

Conclusion:
- `dynamic=0` is not a clear improvement over `dynamic=1` in this setup.
- Despite a small single-step wall gain, graph behavior is less stable and compiled hotspot cost distribution is worse.
- Keep `dynamic=1` as the default narrow-compile mode for now.

### Allocator sawtooth root-cause check (2026-02-23)

Goal:
- explain the strong `nvtop` "sawtooth" behavior observed in a previous session.

Controlled setup:
- `flux2_profile_bench.py`, `--profiler torch`, `capture_step=20`, `stop_step=30`
- narrow compile enabled (`dynamic=1`, `force_contiguous=1`)
- same training config and data path, `save_every_n_epochs=999999` to avoid extra epoch saves
- only allocator config changed via `PYTORCH_CUDA_ALLOC_CONF`

Compared variants:
1. `(none)`
2. `max_split_size_mb:128`
3. `expandable_segments:True`
4. `max_split_size_mb:128,expandable_segments:True`

Primary metrics (GPU/MEM sampled every 250 ms):
- `mem_changes_ge64`: number of adjacent samples with `|delta mem_used_mib| >= 64`
- `tail_mem_range_mib`: MEM range on last `60..5` sec window (steady-state proxy)

Observed:

| Variant | duration_ms | gpu_lt90_pct | mem_changes_ge64 | tail_mem_range_mib |
|---|---:|---:|---:|---:|
| none | 175544 | 17.161 | 21 | 12 |
| max_split_size_mb:128 | 191516 | 18.247 | 382 | 2280 |
| expandable_segments:True | 175470 | 15.312 | 23 | 52 |
| max_split_size_mb:128 + expandable_segments:True | 184244 | 29.018 | 341 | 7360 |

Key findings:
- The sawtooth is primarily triggered by `max_split_size_mb:128`.
- `expandable_segments:True` alone stays close to baseline in both runtime and MEM stability.
- Worst overall behavior is the combined setting (`max_split_size_mb:128,expandable_segments:True`), especially in steady-state MEM oscillation.
- Practical recommendation for this workload: avoid `max_split_size_mb:128`; if allocator tuning is needed, prefer trying `expandable_segments:True` without `max_split_size_mb`.

## Why These Hypotheses Exist (Brief)

- Profiling basis: `torch.profiler` single-step captures from `run9/run10/run11` (capture step 5).
- Main compute hotspot: `aten::mm` is about `~53%` of self CUDA time (`~3.0s` out of `~5.6-5.7s` self CUDA), so we treat matmul path and surrounding overhead as primary targets.
- CPU launch overhead is large: `Command Buffer Full` is about `~51%` self CPU and `cudaLaunchKernel` is about `~28-38%` self CPU, which directly motivates H3/H7.
- Transfer/cast overhead is visible: `aten::to`, `aten::_to_copy`, and `aten::copy_` show high CPU totals (roughly `~0.8-1.8s` in traces), which motivates H1/H2/H6.
- Attention is not the top bottleneck in current traces: `flash_attn` fwd+bwd is materially smaller than matmul+launch overhead, so backend switching is not the first optimization priority.
- Optimizer step is tiny in the profile (`Optimizer.step#AdamW.step` is around `~4ms`, `~0.07%` self CUDA), so optimizer replacement is not a first-order step-time lever.
- Code-path evidence supports the above:
- `call_dit` performs per-step casting of inputs.
- RoPE tensors are moved to device inside rotary application.
- `preserve_distribution_shape=true` uses rejection sampling loops in timestep generation.
- Dataset batching loads safetensors and assembles tensors on CPU in `__getitem__`.
- `gradient_checkpointing=true` is enabled in the active training config.
- End-to-end note: epoch sampling/checkpoint/state save settings can increase wall-clock per epoch even if pure step time is unchanged.
- Validation strategy: `speedup_task/nondet_n5_m12_adamw` is used only as a regression envelope to verify optimization safety.

## Note

The `speedup_task/nondet_n5_m12_adamw` folder is used as a regression control for training behavior after optimizations, not as the target production configuration.
