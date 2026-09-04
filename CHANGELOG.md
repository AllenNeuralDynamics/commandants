# Changelog

All notable changes to this project are documented here. This file is maintained by
[commitizen](https://commitizen-tools.github.io/commitizen/) from conventional-commit
messages. The format is based on [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

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
