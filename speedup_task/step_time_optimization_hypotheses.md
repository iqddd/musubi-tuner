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
| H7 | Reduce per-step launch overhead | Reduce number of small kernel launches and host overhead (per profiler: `cudaLaunchKernel`, `Command Buffer Full`) | Medium | in progress (phase-1 attribution done) |
| H8 | Add step-time metrics to profiling artifacts | Write `step_wall_ms/fwd_bwd_ms/data_wait_ms` to `run_result.json` and compare automatically | Infrastructure impact (faster optimization loop) | not tested |

## TODO

- H7 phase-2: add narrow `record_function`/NVTX around top phase-1 callsites (`flux2_models.RMSNorm.forward`, `flux2_models.apply_rope`) to split cast vs element-wise contribution more precisely.
- H7 phase-3: implement one minimal code change in top hotspot, run A/B on the same capture settings, and validate against nondeterminism envelope.
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
