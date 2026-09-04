# Changelog

All notable changes to this project are documented here. This file is maintained by
[commitizen](https://commitizen-tools.github.io/commitizen/) from conventional-commit
messages. The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Features

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
