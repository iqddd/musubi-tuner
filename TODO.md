# Krea2 complementary alpha output preservation

## Current status (2026-09-28)

Implementation and preservation-mechanism checks are complete. The earlier WIP
blocker was localized to inherited eager/Inductor numerical differences and
nondeterministic FA2 dQ accumulation, rather than preservation packing, teacher
state or checkpointing. The user explicitly accepted that inherited compiler
numerical differences alone need not block this feature. No compiler restriction
is imposed; the original failed thresholds remain documented, not relabeled.

This is **not** evidence of preservation quality or semantic/visual equivalence
after training. No optimizer steps, overfit runs, or training A/B were performed.

Measurements, limitations and reproducible commands:
[docs/krea2_preservation_validation.md](docs/krea2_preservation_validation.md).

## Completed

- [x] Mean alpha includes all pixels in each 16x16 tile; only mean-zero tokens
  are dropped. Mixed boundary tiles survive; raw-alpha caches remain usable.
- [x] Opt-in complementary teacher/target/preservation passes reuse noise,
  timesteps, captions, original RoPE positions and the inverse packed plan.
- [x] LoRA-free detached teacher; both student branches restore LoRA state and
  use full-image loss denominators. No mask-area normalization or division by two.
- [x] Ordinary/mixed-resolution batches, native/sharedkv/logbias attention,
  complementary shared-K/V decisions, empty/full/missing-alpha cases.
- [x] Teacher state restoration, exception paths, separate autocast scopes,
  checkpoint plan reuse, existing backend/gamma/LoRA/block-swap restrictions.
- [x] Removed the all-blocks no-checkpoint OOM control. Optional failures retain
  main results. Selective bypass for blocks 0/14/27 on identical cropped inputs
  gives bitwise equal gradients to the all-checkpointed baseline.
- [x] Separate target/preservation gradients, ordinary training comparison,
  zero-up up/down gradients, FP64 metrics and explicit undefined zero-norm cosine.
- [x] Text-fusion and first/last-block tracing; matched-input/cotangent VJPs.
  Ordinary-path cotangents give bitwise-equal preservation-target gradients.
- [x] All three real-weight GPU modes: finite outputs and all 528 LoRA gradients;
  stable graph counts `[4, 4, 4]` across new masks/seeds at fixed retained counts.
- [x] AOT eager control passes original thresholds for native/sharedkv. Separate
  deterministic fa2-alpha build makes logbias pass as well (gradient relative L2
  0.000337836, cosine 0.999999943, exact eager repeats). Installed wheel unchanged.
- [x] Attention-level deterministic FA2 validation: FP32 reference, MHA/GQA,
  ragged/empty/single-key segments, compiled checkpointing and exact repeatability.
- [x] CPU suite: **104 passed**, 32 warnings. Diagnostic scripts pass Ruff.
- [x] Additional all-mode control on the archived manifest (freshness unverified):
  **all three modes pass** with deterministic FA2 and AOT eager. Predictions/loss
  are exact; gradient relative L2 is 0.000108–0.000256. This is an additional
  input-pair check, not a claimed reproduction of the last WIP run. Jobs finished;
  VRAM was released.

## Remaining quality evaluation

- [ ] Agree a training A/B comparing preservation off/on with identical seeds,
  data, captions, timesteps, rank, optimizer and update budget; inspect held-out
  samples for target learning and changes outside the alpha mask. This is a
  training-quality experiment, not a prerequisite for asserting the mechanism
  checks above passed. Do not claim visual equivalence from these diagnostics.

## Local artifacts

New results: `/workspace/alpha_diagnostics/output_preservation_20260928_resume/`.
Each completed probe retains JSON results and logs; the original failed runs are
also retained. The first probes used a replacement two-image manifest because
historical artifacts were initially unavailable.

The user then supplied `/workspace/alpha_diagnostics.7z`. It was extracted to
`/workspace/alpha_diagnostics_restored_20260928/`; the original archive is intact.
The archived fa2-alpha sources and calibration manifest are under
`alpha_diagnostics/fa2_alpha_20260926/` in that directory. The user does not
guarantee this manifest matches the last WIP run; the additional probe must not
be described as an exact reproduction of those inputs.

`tests/patches/fa2_alpha_deterministic.patch` and
`tests/krea2_deterministic_alpha_probe.py` reproduce the isolated deterministic
control. No production FA2 dependency was replaced. Use fresh output directories,
confirm prior jobs released VRAM, and use supervisor for long-running diagnostics.
The abandoned full saved-tensor offload experiment must not be revived.
