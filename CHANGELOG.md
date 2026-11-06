# Changelog

All notable changes to this project are documented here. This file is maintained by
[commitizen](https://commitizen-tools.github.io/commitizen/) from conventional-commit
messages. The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Features

- **`StptRegistration` masks now reach the deformable stage.** They previously
  restricted the rigid and affine stages only -- the deformable stage, which does the
  fine alignment, ignored them and would happily deform to match the structure the mask
  was meant to exclude. The driver now masks the MRF control grid as well: a node with
  no in-mask evidence takes a flat zero data cost and is placed by the regularizer
  alone. Masked-out regions are therefore smoothly extrapolated, not held still, and
  masking is per control node rather than per sample. The moving mask is carried
  through the linear transform and re-displaced by the deformation field at every
  pyramid level. Masked runs are faster, since a skipped node skips the sampling loop.
  Requires a `gridpoint` build from commit `8769b8e` or later.

- **`StptRegistration` declares `inverse_warp`.** With `write_deformation_field=True`,
  `gridpoint` now writes `stpt_InverseWarp.nii.gz` beside `stpt_Warp.nii.gz`, and the
  wrapper declares it as the `"inverse_warp"` output. The deformable stage is symmetric,
  so this backward field comes out of the optimization rather than from inverting the
  forward one; it is the direction needed to carry fixed-space data (atlas labels,
  annotations, point sets) onto the moving image. It is deliberately **not** added to the
  benchmark's `transforms` chain, which is the forward composition only. Requires a
  `gridpoint` build from commit `c413a3b` or later.

## 0.4.0

### Features

- **`StptRegistration(upstream_optimizer=...)`** exposes the driver's
  `--upstream-optimizer`, which uses the Allen optimizer constants verbatim (translation
  scale 1e7, step 60 -> 0.1) instead of deriving them from the image extent. Worth turning
  on for fidelity on CCF-scale volumes.
- **Input paths are validated before launch.** A path that does not exist now raises
  `MissingInputError` naming the parameter and the value, instead of reaching the tool
  and surfacing as an ITK exception that mentions neither. Applies to image and point-set
  inputs across ANTs, elastix and STPT; `ImageMath` scalar operands and the ANTs `NA`
  mask placeholder are exempt, since neither is a path. Only `run()` validates --
  `build_command()` and `to_shell()` still assemble commands naming files you are about
  to create.

- **STPT/CCF backend scaffolding** (`commandants.stpt`): binary resolution for the
  user-built Allen `idp*` / `gridpoint` binaries via `$STPTPATH` + `PATH`, and a shared
  `StptToolCommand` base. `ToolSpec` gained `downloadable` / `install_command`, so an
  install-less tool is expressible: `spec_for_binary` is now a table lookup and
  `resolve_binary` is spec-driven rather than branching on ANTs-vs-elastix. Install-less
  tools never attempt an auto-download and their not-found message omits `install-*`.
  `commandants version` reports STPT availability. The ANTs and elastix paths are unchanged.
- **Allen STPT/CCF wrapper** (`commandants.stpt`): `StptRegistration` plus
  `presets.rigid/affine/ccf_global/ccf_stpt`, driving the `gridpoint` ITK-only driver
  (built from a separate Allen-licensed repo; the upstream code is BSD-2 + non-commercial, so
  it is not vendored here). Full commandants ergonomics -- `to_shell()` preview, `run()`,
  declared outputs, `extra_args()`, and SimpleITK images anywhere a path is accepted. Registered
  as the third `benchmark` backend, so `compare()` now runs ANTs vs elastix vs STPT by default.
- **Cross-backend benchmark harness** (`commandants.benchmark`): `compare()` registers the
  same image pair with several backends and returns machine-readable trials
  (`BenchmarkReport.summary()` / `to_table()` / `to_csv()`). A `Backend` adapter normalizes
  the genuinely different builder signatures and result shapes (ANTs transform-prefix +
  `expected_transforms()` vs elastix output-directory + `declared_outputs()`) into a uniform
  `{warped, transforms, log}`. Adds fairness controls — pinned threads, streaming forced off,
  equalized warped-image format, per-backend failure isolation, and `warmup`/`repeats`.
  New backends register with one `register_backend()` call.

- **ANTs STPT/CCF analogue presets** (`commandants.presets`): `ccf_stpt` (Rigid →
  Affine Mattes MI, 64 bins → `BSplineSyN` cross-correlation) and `ccf_global` (linear
  only), the ANTs counterpart to the elastix `ccf_stpt` preset. Uses `BSplineSyN` — ANTs'
  B-spline-regularized *symmetric* diffeomorphic transform — as the closest match to the
  Allen STPT local step's symmetric, invertible B-spline; CC is the direct NCC analogue.
- **elastix CCF/STPT presets** (`commandants.elastix.presets`): `ccf_stpt` (rigid →
  affine Mattes MI → B-spline NCC) and `ccf_global` (linear only), plus
  `ccf_parameter_maps()` for editing. Mimics the AllenInstitute/stpt_registration
  template-construction schedule (64-bin MI, center-of-gravity init, coarse-to-fine
  from shrink 8; 3rd-order B-spline, NCC, 4-level grid ~30→4 voxels). The original
  deformable step uses a discrete MRF optimizer, so the B-spline stage matches in
  spirit, not in optimizer mechanism (documented in the preset).

## 0.3.0

### Features

- **elastix support** (`commandants.elastix`): transparent CLI wrappers for the
  `elastix` (`Elastix`) and `transformix` (`Transformix`) binaries, reusing the shared
  commandants core (run/stream/materialization/exit-codes). Configuration is via
  pass-through `ParameterMap`s (read/write elastix `(Key value)` files) with curated
  `presets` (rigid/affine/bspline/translation/affine_bspline). Added an
  `install-elastix` provisioner (downloads official prebuilt elastix binaries),
  tool-aware `which`/`list`/`version`, and a `benchmark_elastix_vs_ants.py` example.
- `CompletedAnts.duration_seconds` — wall-clock time of a run, for benchmarking.

## 0.2.0

First public release. A thin, transparent Python wrapper around the ANTs
command-line binaries.

### Features

- **Registration**: `AntsRegistration` multi-stage / multi-metric builder, including
  point-set metrics (`PSE`/`ICP`/`JHCT`) to constrain warps with points;
  `AntsApplyTransforms` and `AntsApplyTransformsToPoints`.
- **Presets** (`commandants.presets`): `rigid`, `affine`, `syn`, `syn_only`,
  `translation`, `similarity` — ANTsPyX-style builders with a center-of-mass init by
  default and ANTsPyX-matched linear schedules.
- **Per-stage metric masks** via `add_stage(fixed_mask=, moving_mask=)`.
- **In-memory images**: pass/receive `SimpleITK.Image` objects (auto temp files with
  exposed paths); `CompletedAnts.load()` returns SimpleITK.
- **Preprocessing**: `N4BiasFieldCorrection`, `ThresholdImage`, `ImageMath`,
  `ResampleImage`.
- **ANTs provisioner**: `commandants install-ants` downloads official prebuilt ANTs
  binaries; `resolve_binary` discovers them.
- **Resource estimator** (`estimate_resources`), **exit-code explainer**
  (`explain_exit_code`; `-9` = SIGKILL/OOM), and **live output streaming**
  (`run(stream=, on_line=, log_file=)`).
- `AntsApplyTransforms` `output_data_type=` (stored pixel type); single precision
  (`use_float=True`) is the default for registration and apply.

### Notes

- Zero required runtime dependencies (core shells out to ANTs); `[io]` extra adds
  SimpleITK + numpy.
