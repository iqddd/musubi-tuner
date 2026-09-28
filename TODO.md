# WIP: Krea2 complementary alpha output preservation

## State at handoff (2026-09-28)

Implementation is present, but **GPU numerical validation has NOT passed**.
Do not treat this WIP as a validated training feature. No optimizer steps or
overfit runs were performed for this implementation. Diagnostic jobs are stopped
or exited; no new GPU work was launched after the last OOM.

### Implemented

- With `--alpha_masked_token_drop`, average all pixels in each 16x16 tile,
  including zeros. Use the mean for MSE and attention; drop only mean-zero tiles.
  This intentionally retains mixed zero/positive boundary tiles. Without
  token-drop, existing masking is unchanged. Raw-alpha caches remain usable.
- Add opt-in `--alpha_masked_output_preservation`: teacher without LoRA/gradients
  uses `1-a`, target student uses `a`, preservation student uses `1-a` and learns
  the detached teacher output. Both students have LoRA enabled.
- Sum complementary weighted losses over each full image's original denominator,
  then average images. No mask-area renormalization or division by two.
- Reuse input noise/timesteps/captions, original RoPE coordinates, and the exact
  inverse packed plan between teacher and preservation student. Support ordinary
  and mixed-resolution batches and native/sharedkv/logbias attention.
- Sharedkv uses one uniform per image: target K/V when `u<a`, inverse K/V when
  `u>=a`; text always remains. Plans are reused during checkpoint recomputation.
- Restore LoRA multipliers/training flags before students, including exception
  paths. Separate teacher/student autocast scopes and clear the teacher cast cache.
- Require standard Krea2 LoRA and token-drop; reject block swap for preservation.
  Preserve existing backend/gamma validation. Sampling is unchanged.
- Keep bucketing based on target-Q count; add inverse length/padding dry-run stats.
- Add documentation, CPU tests, and explicit opt-in GPU diagnostic scripts.

### Verified on CPU

```bash
.venv/bin/python -m pytest tests/test_krea2*.py tests/test_alpha_masked_loss.py -q --disable-warnings --maxfail=2
```

Last completed result: **95 passed**, 32 warnings. Tests include all three modes,
mixed resolutions, zero/full/missing alpha, teacher state restoration, plan reuse,
BF16 autocast gradient preservation, opaque-mask equivalence to the old training
path, and combined gradient equivalence to the sum of the loss-branch gradients.
Production code has not changed since that successful suite; subsequent changes
were in the diagnostic scripts. Run the suite again after further implementation.

## GPU findings and caveats

Environment: RTX 5090 32 GB, real RAW scaled-FP8 weights, BF16 autocast, FP32
rank-32 LoRA, FlashAttention, 28 DiT blocks. No optimizer. Main full-size inputs
come from the first two entries of the calibration manifest listed below.

1. Initial probe completed in approximately 207 seconds. All modes had finite
   outputs/gradients and matching sets of parameters with gradients. After warmup,
   graph counts stayed `[4, 4, 4]` across fresh masks/seeds. Predictions were close
   (~0.38-0.55% relative L2), total loss differences ~0.02-0.12%, but LoRA gradient
   differences exceeded acceptance thresholds substantially.
2. LoRA-up was initialized to **nonzero normal values, std=0.005**, deliberately.
   The initial failure therefore cannot be attributed directly to zero-up
   initialization. A separate zero-up control is written but has not run yet.
3. Ordinary FA backward showed repeat-to-repeat gradient differences of roughly
   3% for eager and 1.7% for compiled, despite identical predictions/loss.
4. Deterministic FA backward removed that repeat noise completely. However,
   compiled vs eager still had **gradient relative L2 = 1.7329567, cosine =
   0.4261774**. Reference gradient norm ~0.010285; compiled norm ~0.019585.
   Do not dismiss this as normal rounding without further isolation.
5. Initial diagnostic cosine calculations used FP32 reductions on huge vectors
   and could even exceed 1. They are not reliable. The helper now uses FP64 and
   reports both vector norms/zero norms. The deterministic figures above use the
   corrected metric (tiny deviations above 1 at ~1e-12 are reduction roundoff).
6. Full-size no-checkpoint control with CPU saved-tensor offload was stopped after
   consuming over 120 GiB RAM. Releasing pinned memory/VRAM was slow. An overlapping
   restart failed during weight loading; the script now waits for free VRAM.
7. A separate small-input no-checkpoint control also OOMed, this time during the
   target student at `fp8_optimization_utils.py`'s BF16 weight dequantization.
   Without checkpointing, saved dequantized weights accumulate independently of
   image size. The diagnostic failure is not an OOM in the normal checkpointed
   preservation path. Even this smaller no-checkpoint control must be replaced.

## Next actions, in order

- [ ] Fix `tests/krea2_preservation_gradient_diagnostic.py` **before restarting**:
  it still attempts the known-OOM small-input all-blocks no-checkpoint control.
  Replace it with selective checkpoint bypass for one block at a time (first,
  middle, last initially), keeping other blocks checkpointed. Compare against an
  all-checkpoint baseline with identical inputs, weights and deterministic FA.
  Do not revive the >120 GiB full saved-tensor offload experiment.
- [ ] Move optional memory-heavy controls after the main branch-isolation tests,
  and record an optional control failure without losing other diagnostic results.
- [ ] Complete the already-written separate target/preservation gradient tests.
  They were not reached because the earlier checkpoint control failed.
- [ ] Complete the original single-target training-path comparison against the
  preservation target branch. This is written but has not executed yet.
- [ ] Complete the zero-up control. Report up/down matrix gradients separately;
  zero down-gradient norms at initialization must not be called a cosine failure.
- [ ] If eager/compile differences persist, localize forward and backward changes
  at text fusion and first/last DiT blocks, then compare a block's eager/compiled
  VJP using identical block inputs and output cotangents. Distinguish cumulative
  BF16/FP8 sensitivity from incorrect packing, LoRA state or checkpoint handling.
- [ ] Check whether the discrepancy also exists without preservation before
  attributing it to the feature. Do not silently relax thresholds to obtain a pass.
- [ ] After fixes/isolation, rerun CPU and all three GPU modes with corrected
  FP64 metrics; retain reports, branch-wise errors and gradient availability.
  Original acceptance criteria: prediction relative L2 <= 5e-3, total loss
  relative difference <= 5e-3, aggregate LoRA gradient relative L2 <= 1e-2 and
  cosine >= 0.999. Explain any justified revision separately.
- [ ] Only after numerical validation discuss a training A/B with the user.
  Current tests do not establish preservation quality or suppression of artifacts.

## Reproduction and local artifacts

Scripts in this commit:

- `tests/krea2_preservation_gpu_probe.py`: all-mode eager/compile/graph-stability probe.
- `tests/krea2_preservation_gradient_diagnostic.py`: isolation probe; see OOM warning above.

Local environment artifacts (outside this repository, not included in commit):

- Weights: `/workspace/models/raw.safetensors`.
- Manifest: `/workspace/alpha_diagnostics/fa2_alpha_20260926/calibration/manifest.json`.
- Root: `/workspace/alpha_diagnostics/output_preservation_20260928/`.
- Initial results: `report.json`, `REPORT.md`, `run.log`, `CPU_REPORT.md`.
- Interrupted ordinary-FA isolation: `gradient_isolation/report.json`,
  `gradient_isolation.log`; its persisted `running` status is stale.
- Deterministic isolation/OOM: `gradient_deterministic/report.json`,
  `gradient_deterministic/REPORT.md`, `gradient_deterministic.log`.
- Supervisor jobs/configs: `krea2_preservation_probe`,
  `krea2_preservation_gradient`, `krea2_preservation_deterministic`, under
  `/etc/supervisor/conf.d/`. They do not automatically restart.

Use a fresh output directory for subsequent attempts to preserve these results.
Do not start another GPU process before confirming the previous one released
VRAM. If a scenario is expected to take over 10 minutes, leave it running under
supervisor and report an ETA; the user will prompt when it finishes.
