# Krea2 preservation numerical validation (2026-09-28)

No optimizer steps or training A/B were performed. This report concerns numerical
consistency, not preservation quality or suppression of artifacts.

## Reproduction

RTX 5090 32 GB; PyTorch 2.10.0+cu130; FlashAttention 2.8.3; RAW weights quantized
with the existing scaled-FP8 loader; BF16 autocast; FP32 rank-32/alpha-32 LoRA;
LoRA-up initialized with normal std=0.005, plus a separate zero-up control.
All 28 DiT blocks are present. Native/sharedkv use deterministic FA backward.
The custom fa2-alpha logbias operator exposes no deterministic-backward option.

The historical calibration manifest and reports referenced by the handoff notes
were absent on this instance. The replacement manifest uses two existing alpha/text/latent cache pairs
with image sizes 768x896 and 640x1024, and 144/158 text tokens. These are new
measurements, not exact reproductions of the historical values.

Local artifacts are retained under
`/workspace/alpha_diagnostics/output_preservation_20260928_resume/`:

- `manifest.json`: exact input paths.
- `gradient/`: initial repeat-backward OOM, retained for diagnosis.
- `gradient_retry/`: branch isolation, zero-up, ordinary-path controls; full-input
  selective checkpoint controls exceeded VRAM after the main results were saved.
- `localize/`: initial VJP parameter-selection error, retained.
- `localize_v2/`: successful matched-input block VJPs and selective checkpoints.
- `precision_casts/`: Inductor with `emulate_precision_casts=True`.
- `aot_eager/`: AOTAutograd without Inductor kernel generation, with matched
  output-cotangent control and block VJPs.

Each run has `report.json`; logs are alongside the run directories. Use a fresh
output directory. Start only after the previous process has released VRAM.
`PYTORCH_ALLOC_CONF=expandable_segments:True` and collection between cases were
needed for repeated full-input passes on this 32 GB device. The diagnostic scripts
perform no optimizer updates. Optional checkpoint controls bypass one block at a
time on a 128x128 image crop (16x16 latent) with 32 text tokens; all other blocks remain
checkpointed. The abandoned all-blocks no-checkpoint/offload experiment is removed.

Example (replace the paths with local weights/caches):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  tests/krea2_preservation_gradient_diagnostic.py \
  --dit /workspace/models/raw.safetensors \
  --manifest /workspace/alpha_diagnostics/output_preservation_20260928_resume/manifest.json \
  --output /workspace/alpha_diagnostics/new_isolation \
  --deterministic-fa --checkpoint-controls --localize
```

`--compile-backend aot_eager` selects the AOT control;
`--emulate-precision-casts` selects the Inductor rounding control;
`--localize-only` skips the main branch suite and runs boundary/VJP localization.
The all-mode probe accepts the same backend/rounding flags and deterministic FA.

## Isolation results

All reductions in comparison metrics use FP64. A cosine with either zero norm is
reported as undefined; two zero vectors are equal, not a cosine failure.

| Comparison | LoRA gradient relative L2 | Cosine |
|---|---:|---:|
| Deterministic eager repeat | 0 | 1 |
| Deterministic Inductor repeat | 0 | 1 |
| Inductor vs eager, total | 0.976196 | 0.225157 |
| Inductor vs eager, target | 0.976247 | 0.224563 |
| Inductor vs eager, preservation | 0.369891 | 0.933530 |
| Inductor vs eager, ordinary training without preservation | 0.975585 | 0.226949 |
| Inductor with precision-cast emulation vs eager, total | 1.000918 | 0.167536 |
| AOT eager vs eager, total | 0.000187676 | 0.999999982 |
| AOT eager vs eager, each separate loss branch | 0 | 1 |
| AOT eager vs eager, ordinary training | 0 | 1 |
| AOT eager vs eager, zero-up | 0 | 1 |
| Bypass checkpoint block 0 / 14 / 27 vs all checkpointed (crop) | 0 | 1 |

The ordinary target forward and preservation target forward are bitwise equal.
Their output cotangents differ by relative L2 2.37137e-5; the resulting LoRA
gradients differ by 0.0207388. Reusing the **exact ordinary-path output cotangent**
on the preservation target makes their LoRA gradients bitwise equal. This isolates
that difference to loss arithmetic and its amplification by BF16 backward, rather
than packing, teacher state, or checkpointing. The ordinary path performs another
area reduction on its already-aligned alpha mask; the preservation branch reuses
the original tile mean directly.

Text-fusion forward outputs are bitwise equal across eager/Inductor. With identical
block inputs and output cotangents, first-block input VJP errors are 0.08699 and
0.10619 for the target/preservation branches; last-block errors are 0.001035 and
0.000981. Precision-cast emulation reduces the first-block errors to 0.04205 and
0.04672, but does not satisfy the end-to-end gradient threshold. This supports
cumulative numerical sensitivity in Inductor computations; it does not establish
that every discrepancy is benign or identify a single faulty kernel.

Inductor's zero-up preservation loss is 3.36e-6 versus eager's exact zero; AOT eager
also gives exact zero. Both down-matrix gradient norms at zero-up are exactly zero.

The original thresholds remain prediction relative L2 <= 5e-3, total loss relative
difference <= 5e-3, aggregate gradient relative L2 <= 1e-2 and cosine >= 0.999.
No thresholds have been relaxed.

All four matched-input first/last-block VJPs are bitwise equal between eager and
AOT eager, including input, modulation and every block LoRA gradient.


## Acceptance clarification

The user clarified that inherited eager/compile discrepancies alone need not
block preservation; practical semantic/visual results should be approximately
equivalent. The original thresholds are retained as diagnostics, and failing runs
are not relabeled as passes. No compiler restriction is imposed. This validation
does not establish training quality or visual equivalence.

## Full-model all-mode results

All 528 LoRA parameter gradients are present and finite. Graph counts remain
`[4, 4, 4]` with new tile-translated masks and seeds (fixed retained counts).
Reports: `all_modes_inductor/` and `all_modes_aot/` under the artifact root.

| Backend | Mode | Max prediction rel-L2 | Total loss relative difference | Gradient rel-L2 | Cosine | Eager repeat gradient rel-L2 |
|---|---|---:|---:|---:|---:|---:|
| Inductor | native | 0.00458259 | 0.000871635 | 0.976196 | 0.225157392 | 0 |
| Inductor | sharedkv | 0.00453784 | 0.000199539 | 5.13183 | 0.468471948 | 0 |
| Inductor | logbias | 0.00522694 | 0.000817962 | 2.785 | 0.29656994 | 0.122807 |
| AOT eager | native | 0 | 0 | 0.000187676 | 0.999999982 | 0 |
| AOT eager | sharedkv | 0 | 0 | 0.000318583 | 0.999999949 | 0 |
| AOT eager | logbias | 0 | 0 | 0.0484401 | 0.998838425 | 0.0491423 |

AOT eager satisfies the original thresholds for native/sharedkv. Its logbias
forward and loss are exact, but backward variation is comparable to repeating the
same eager computation. The archived adapter uses nondeterministic dQ atomic
accumulation; it does not enable the deterministic backward already present in
its upstream FA2 kernels.

## Archived source and deterministic control

The user supplied `/workspace/alpha_diagnostics.7z`, extracted separately to
`/workspace/alpha_diagnostics_restored_20260928/`. Sources are in
`alpha_diagnostics/fa2_alpha_20260926/` there. The original archive and installed
wheel remain unchanged.

`tests/patches/fa2_alpha_deterministic.patch` adds upstream FA2's deterministic
dQ buffer allocation, split stride and flag to the archived adapter, with a
boolean backward argument. Build it separately using the archived `build.py`,
changing `name` to `fa2_alpha_deterministic_cuda` and `FLASH_NAMESPACE` to
`fa2_alpha_deterministic_experiment`. Keep the archived patched vendor tree;
this patch is for the adapter, not unmodified upstream FA2.

The local build is in `fa2_deterministic/` under the artifact root. Its extension
is `/root/.cache/torch_extensions/py312_cu130/fa2_alpha_deterministic_cuda/fa2_alpha_deterministic_cuda.so`.
Only diagnostic processes redirect the installed Python wrapper to this separate
binary. It is not installed as a training dependency.

`tests/krea2_deterministic_alpha_probe.py --extension PATH --output NEW_DIRECTORY`
checks FP32-reference forward/backward, exact repeatability, MHA/GQA, ragged
segments (including empty and single-key segments), and compiled checkpointing.
Reports are in `deterministic_operator_v2/`; lengths 128, 257 and 512 all pass.
Maximum forward rel-L2: 0.00204107; maximum Q/K/V gradient rel-L2: 0.00293247; minimum gradient cosine: 0.9999957.

For the full-model probe, pass `--deterministic-alpha-extension PATH` alongside
`--deterministic-fa`. `--modes logbias` selects just this mode; without that flag
all three modes run. This isolates atomic accumulation noise without changing
training behavior or weakening comparison thresholds.


The full-model `logbias_deterministic_aot/` control **passes** the original
thresholds: exact predictions and loss, gradient relative L2 0.000337836,
cosine 0.999999943, exact eager repeats, 528 finite parameter gradients, stable
`[4, 4, 4]` graph counts. The archive also contains a calibration manifest. **The user explicitly does not
confirm its freshness or that it matches the last WIP run.** An additional
all-mode check on its first two entries is recorded in
`original_manifest_deterministic_aot/` (the directory name predates this
clarification). Treat this as another input pair, not an exact reproduction of
the last WIP run.


Additional archived-manifest control: **all modes pass** with AOT eager and
deterministic FA2. Predictions and both loss terms are bitwise equal; eager
repeats are exact; all 528 parameter gradients are finite and graph counts are
`[4, 4, 4]` for every mode.

| Mode | Gradient relative L2 | Cosine |
|---|---:|---:|
| native | 0.000256150914 | 0.9999999672 |
| sharedkv | 0.00010800182 | 0.9999999942 |
| logbias | 0.000176261094 | 0.9999999845 |

The preservation implementation's diagnostic WIP is resolved under the user's
clarified acceptance scope. Default Inductor still fails the original numerical
thresholds and the installed alpha kernel still has nondeterministic backward;
neither result is concealed by the passing isolated control. The default training
backend and installed wheel remain unchanged. Training-quality/visual A/B is the
remaining evaluation and has not been run.
