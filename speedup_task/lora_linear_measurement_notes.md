# LoRA / Linear Profiling Notes (FLUX.2)

Date: 2026-02-23  
Scope: clarification of LoRA-related hotspots, measurement pitfalls, and correct profiling procedure.

## 1) Setup used in this investigation

- Run type: `N=5`, `M=12`, single-step torch profiler capture at step 5
- Narrow compile: **OFF** (`MUSUBI_H7_NARROW_COMPILE=0`)
- LoRA forward markers: **ON** (`MUSUBI_LORA_FORWARD_MARKERS=1`)
- Artifact used:
  - `speedup_task/profile_lora_markers_n5_m12/20260223220140_torch_capture0005_stop0012/artifacts/20260223220147_capture00000005_stop00000012/torch_profiler/rank00/step00000005.json`

## 2) What LoRA computes in train path

In train/inference-time forward of `LoRAModule`, the code computes:

- `org_forwarded = org_forward(x)`
- `lx = lora_down(x)`
- `lx = lora_up(lx)`
- `output = org_forwarded + lx * scale`

So the train path does **not** explicitly materialize `ΔW = B @ A` as a full weight matrix.
It computes the equivalent expression in activation space.

Reference: `src/musubi_tuner/networks/lora.py` (`LoRAModule.forward`).

## 3) Why an earlier interpretation was wrong

A misleading pattern appeared when looking at `user_annotation` CPU wall markers only:

- `lora/forward/up` looked larger than `lora/forward/org_forward` in backward.

This does **not** mean `up/down` dominate true GPU compute.
`user_annotation` in this case mostly reflects CPU-side dispatch/launch overlap and checkpoint/autograd scheduling effects.

When using GPU-side evidence:

- `gpu_user_annotation` totals (`lora/forward/*`, step 5):
  - `org_forward`: **1984.981 ms**
  - `residual_add`: **159.020 ms**
  - `up`: **61.839 ms**
  - `down`: **46.388 ms**

And by approximate GEMM volume from `aten::linear` input/weight shapes (forward stage):

- `org_forward`: `2.227e14` MAC
- `down`: `7.027e11` MAC
- `up`: `1.372e12` MAC
- Ratios: `org/down ≈ 317x`, `org/up ≈ 162x`

Conclusion: **base linear (`org_forward`) dominates compute**, not LoRA `up/down`.

## 4) Shapes observed for LoRA markers (step 5)

Typical shapes captured from `aten::linear` input dims:

- `org_forward`:
  - input `[5, 3191, 4096]`, weight `[36864, 4096]`
  - input `[5, 3191, 16384]`, weight `[4096, 16384]`
- `down`:
  - input `[5, *, 4096|12288|16384]`, weight `[32, *]`
- `up`:
  - input `[5, *, 32]`, weight `[4096|12288|24576|36864, 32]`

Here `3191 = 2679 + 512` after `txt + img` concat in `flux2_models.py`.

## 5) How to measure correctly (to avoid wrong conclusions)

Do:

1. Use **GPU-grounded metrics** for compute attribution:
   - `kernel` category durations
   - `gpu_user_annotation`
   - `Self CUDA`/operator CUDA totals
2. Treat `user_annotation` (CPU) as orchestration/dispatch timing, not pure compute.
3. Split analysis by stage (`train/dit_forward`, `train/backward`) and keep stage-local totals.
4. For backward under checkpointing, separate:
   - recompute forward fragments
   - true backward kernels
5. Validate interpretation with shape/FLOP sanity checks (as above).
6. Compare multiple runs and use median/p50, not a single run, when deciding optimization priority.

Do not:

- infer GPU bottlenecks from CPU `record_function` durations alone.
- conclude dominance from backward marker totals without checking kernel-level evidence.

## 6) /dev/shm limitation and required execution mode

In this environment, sandboxed runs can fail with:

- `PermissionError: [Errno 13] Permission denied: '/dev/shm/pym-...'`
- or `SemLock` permission errors from Python multiprocessing.

For repeatable profiling here, run benchmark commands with **escalated execution** each time (outside sandbox constraints).

Practical note:

- In Codex tool usage, launch `flux2_profile_bench.py` with escalation/approval enabled.
- Otherwise, profiling can fail before training starts, independent of model code.

Example command used in this investigation:

```bash
PATH=/workspace/musubi-tuner/.venv/bin:$PATH \
MUSUBI_LORA_FORWARD_MARKERS=1 \
MUSUBI_H7_NARROW_COMPILE=0 \
/workspace/musubi-tuner/.venv/bin/python speedup_task/flux2_profile_bench.py \
  --config_file speedup_task/config_nondet_adamw_no_epochsave.toml \
  --profiler torch \
  --capture_step 5 \
  --stop_step 12 \
  --output_root speedup_task/profile_lora_markers_n5_m12
```

## 7) Current optimization implication

- Main compute bottleneck remains base GEMM (`org_forward` / large linear ops).
- LoRA `up/down` and cast/copy overhead are secondary targets.
- LoRA markers added in `networks/lora.py` are useful for decomposition, but must be interpreted with GPU-side metrics.
