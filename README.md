# command**ANTs**

A thin, transparent Python wrapper for the [ANTs](https://github.com/ANTsX/ANTs)
command-line tools.

Unlike ANTsPyX — which mixes compiled bindings with opinionated defaults and
exposes only a curated slice of the CLI — `commandants` shells out to the ANTs
binaries directly and **hides nothing**:

- **Every option is reachable.** Typed builders cover the common flags; a
  `.extra_args(...)` escape hatch reaches anything else (present or future).
- **Every command is inspectable.** `.build_command()` / `.to_shell()` return the
  exact argv *without running anything* — so you can see, log, diff, and unit-test
  commands even on a machine with no ANTs installed.
- **The real `antsRegistration` grammar.** Compose arbitrary multi-stage
  pipelines, put **multiple metrics in one stage** (multivariate registration),
  and use **point-set metrics** (`PSE` / `ICP` / `JHCT`) to **constrain a warp
  with corresponding points** — the thing ANTsPyX won't let you do.

## Install

```bash
pip install commandants            # core, zero dependencies
pip install 'commandants[io]'      # + SimpleITK/numpy for in-memory images & point CSVs
pip install 'commandants[dev]'     # + pytest (and the io deps)
```

`commandants` calls the ANTs binaries, so you also need ANTs available.

### Getting ANTs binaries

You can point `commandants` at an existing ANTs install (on `PATH`, via
`$ANTSPATH`, or `ants_path=...`), **or** let it fetch the official prebuilt
binaries for you:

```bash
pip install commandants
commandants install-ants          # downloads official prebuilt ANTs for your OS
commandants version               # -> commandants 0.1.0 / ANTs 2.6.5.x
commandants which antsRegistration
```

`install-ants` downloads the matching archive from the
[ANTsX/ANTs releases](https://github.com/ANTsX/ANTs/releases), unpacks it into a
managed per-user directory (`commandants info` shows where), and `resolve_binary`
discovers it automatically — after PATH, so a system ANTs still wins if present.

> Note: unlike ANTsPyX (which bundles compiled bindings into its wheels), the ANTs
> CLI archives are 400–750 MB, far over PyPI's limits — so they can't ship inside
> the pip wheel. `commandants` stays tiny and fetches them on demand instead.

Nothing downloads implicitly. If you *want* auto-fetch on first use, opt in:

```python
import os

os.environ["COMMANDANTS_AUTO_INSTALL"] = "1"  # or resolve_binary(..., auto_install=True)
```

Other CLI subcommands: `commandants list`, `commandants uninstall-ants`,
`commandants install-ants --version latest`, `--asset <name>` to override the
platform pick.

## Quickstart

```python
from commandants import (
    AntsRegistration,
    Rigid,
    SyN,
    MI,
    CC,
    PSE,
    Convergence,
)

reg = AntsRegistration(
    dimensionality=3,
    output="out_",
    warped_output="out_Warped.nii.gz",
    write_composite_transform=True,
    verbose=True,
)

# center-of-mass initialization  ->  --initial-moving-transform [fixed,moving,1]
reg.initialize_from_images("fixed.nii.gz", "moving.nii.gz", init="center-of-mass")

# Stage 1: rigid, mutual information
reg.add_stage(
    transform=Rigid(gradient_step=0.1),
    metrics=MI("fixed.nii.gz", "moving.nii.gz", weight=1.0, bins=32, sampling="Regular", sampling_pct=0.25),
    convergence=Convergence([1000, 500, 250, 100], threshold=1e-6, window=10),
    shrink_factors=[8, 4, 2, 1],
    smoothing_sigmas=[3, 2, 1, 0],
)

# Stage 2: SyN, driven by CC *and* constrained by a point set
reg.add_stage(
    transform=SyN(gradient_step=0.1, update_field_variance=3, total_field_variance=0),
    metrics=[
        CC("fixed.nii.gz", "moving.nii.gz", weight=1.0, radius=4),
        PSE(
            "fixed_points.csv",
            "moving_points.csv",  # <-- constrain the warp
            weight=0.5,
            point_set_sigma=1.0,
            sampling_pct=0.5,
        ),
    ],
    convergence=Convergence([100, 70, 50, 20]),
    shrink_factors=[8, 4, 2, 1],
    smoothing_sigmas=[3, 2, 1, 0],
)

print(reg.to_shell())  # inspect the exact command (no ANTs needed)
result = reg.run()  # execute; raises AntsRuntimeError on failure
warped = result.load("warped")  # -> SimpleITK.Image (needs the [io] extra)
```

### Presets (ANTsPyX-style)

Don't want to hand-build stages? Use a preset — it returns a ready
`AntsRegistration` (you still inspect/estimate/run it):

```python
from commandants import presets

reg = presets.syn(
    "fixed.nii.gz", "moving.nii.gz", "out_", warped_output="out_Warped.nii.gz", use_float=True, verbose=True
)
reg.run(stream=True)
```

| Preset | Stages | Center-of-mass init? |
|--------|--------|----------------------|
| `rigid` | Rigid | yes |
| `affine` | Rigid → Affine | yes |
| `syn` | Rigid → Affine → SyN (≈ antsRegistrationSyN) | yes |
| `syn_only` | SyN | **no** (assumes pre-aligned; pass `init=` or `initial_transform=`) |
| `translation`, `similarity` | (linear) | yes |

Defaults mirror ANTsPyX (single precision, Mattes MI, light winsorization, and the
same linear schedule — `Rigid/Affine[0.25]`, `2100x1200x1200x10` iterations,
`6x4x2x1` shrink to full resolution) so the presets recover large offsets the way
ANTsPyX does. Every schedule/metric/step is an overridable argument — nothing hidden. Unlike a
hand-built registration, the linear presets and `syn` **include the
center-of-mass initialization by default**, so you won't get the identity-transform
surprise from a missing `--initial-moving-transform`. Worked example (incl. a
memory-safe SyN and an affine→SyN-only warm-start):
[`examples/using_presets.py`](examples/using_presets.py).

### In-memory images (SimpleITK)

Because it wraps the ANTs *binaries*, `commandants` is path-first — the binaries
read and write files. But you don't have to manage those files: pass a
`SimpleITK.Image` anywhere a path is expected and it's written to a temp file
automatically (SimpleITK is used because it preserves origin/spacing/direction, so
the disk round-trip is lossless). The temp files are **not hidden** — the result
exposes exactly where they went:

```python
import SimpleITK as sitk

fixed = sitk.ReadImage("fixed.nii.gz")
moving = sitk.ReadImage("moving.nii.gz")

reg = AntsRegistration(3, output="out_")
reg.initialize_from_images(fixed, moving)  # SimpleITK images
reg.add_stage(SyN(), CC(fixed, moving), Convergence([100, 70, 50]), [4, 2, 1], [2, 1, 0])

result = reg.run()  # writes temp inputs, runs ANTs
print(result.temp_dir)  # e.g. .../commandants_ab12cd
print(result.workspace.files)  # every temp file written
print(result.workspace.inputs)  # {'init_fixed': '.../init_fixed.nii.gz', ...}
```

Control where temp files live and whether they persist:

```python
from commandants import TempWorkspace

result = reg.run(temp_dir="D:/scratch", keep_temp=True)  # choose the dir; keep files
# or hand in a managed workspace that cleans up on exit:
with TempWorkspace(base="D:/scratch", keep=False) as ws:
    reg.run(workspace=ws)
```

Previews never touch disk — `to_shell()` / `build_command()` render in-memory
images as `<sitk:...>` placeholders. The same image object passed to several
arguments is written only once.

### Estimate memory / runtime before running

Get a rough peak-memory (and weak runtime) estimate for a built registration
*before* launching it — handy for sizing a SyN job:

```python
est = reg.estimate_resources(shape=(256, 256, 180), threads=8)  # or fixed=<path/SITK image>
print(est.summary())
est.peak_memory_bytes  # ~2.5 GiB here; drops to ~1.3 GiB with use_float=True
est.per_stage  # per-stage breakdown (the SyN stage dominates)
est.est_runtime_seconds  # very rough
```

The image dimensions come from the fixed image (inferred from your init/metric
images, or pass `fixed=` / `shape=`). **This is a documented heuristic, not a
measurement** — order-of-magnitude for planning, not a guarantee. See
[`commandants/estimate.py`](src/commandants/estimate.py) for the model.

### Live progress for long runs

By default `run()` buffers output and returns it at the end. For a long job, stream
it live instead (needs `verbose=True` on the builder so ANTs emits progress):

```python
reg = AntsRegistration(3, output="reg_", verbose=True, ...)

reg.run(stream=True)                       # echo each line to the console
reg.run(log_file="reg.log")                # write each line to a file (live, tail-able)
reg.run(stream=True, log_file="reg.log")   # tee: console AND file
reg.run(on_line=lambda ln: logging.info(ln.rstrip()))   # or route to logging/tqdm/...

result = reg.run(stream=True)
print(len(result.stdout))                  # output is still captured regardless
```

`stream` (console), `log_file` (a path — opened/closed for you — or an open file
handle), and `on_line` (callback) are independent and combine freely. When
streaming, stdout and stderr are merged so ordering is preserved (`result.stdout`
holds the merged text). If output looks chunky rather than per-iteration, that's
the child process buffering, not the wrapper.

**Jupyter:** `stream=True` works in notebooks — the wrapper reads ANTs' output and
re-emits it through Python's `sys.stdout` (which Jupyter redirects into the cell),
flushing per line. (This is why streaming shows up where raw inherited subprocess
output would not.) For very long runs, prefer `log_file=` (or an `on_line` that
updates a single tqdm bar) to avoid flooding the cell with thousands of lines.

### Understanding exit codes (e.g. `-9`)

ANTs has no big numbered error-code table — it exits `0`/`1` and puts detail in
stderr; anything else is a **signal** or a **Windows exception**. `commandants`
decodes them:

```bash
commandants explain -9      # or 137, -11, 3221225477, ...
```

```python
result = reg.run(check=False)
if result.returncode != 0:
    print(result.explain().text())
```

`AntsRuntimeError` also embeds the decoding in its message. The big one: **`-9`
(or `137`) is SIGKILL — almost always the out-of-memory killer or a cluster memory
limit**, not an ANTs bug. Reach for `estimate_resources()` and `use_float=True`.

### Nothing is hidden

Reach any flag the typed API doesn't model yet:

```python
reg.extra_args("--use-estimate-learning-rate-once", "1")
```

### Apply transforms

```python
from commandants import AntsApplyTransforms

apply = AntsApplyTransforms(
    3, "moving.nii.gz", "fixed.nii.gz", "resampled.nii.gz", interpolation="BSpline[3]", output_data_type="float"
)  # stored pixel type (char/short/int/float/...)
apply.add_transform("out_1Warp.nii.gz")
apply.add_transform("out_0GenericAffine.mat")  # applied first (ANTs order)
apply.run()
```

> **Precision defaults to single (`--float 1`).** `AntsRegistration` and
> `AntsApplyTransforms` compute in single precision by default (like ANTsPyX's
> `singleprecision=True`, and easier on memory for large volumes). Pass
> `use_float=False` for double, or `use_float=None` to omit the flag entirely (ANTs'
> own default is double). This is *computation* precision — `output_data_type=`
> separately controls the stored pixel type of the written image.

> **Argument order — `input`/`reference`, not `fixed`/`moving`.** commandants uses
> ANTs' own `antsApplyTransforms` terminology, which is **not** ANTsPyX's. The 2nd
> arg is `input_image` (`-i`, the image whose pixels are warped) and the 3rd is
> `reference_image` (`-r`, the grid/space of the output) — so applying a
> moving→fixed transform is `AntsApplyTransforms(3, moving, fixed, out)`. ANTs uses
> "input" (not "moving") on purpose: the input is often **not** the original moving
> image — it may be a label map, another channel, or a point set. Coming from
> ANTsPyX's `apply_transforms(fixed, moving, ...)`? The mapping is
> **`moving → input_image`, `fixed → reference_image`**. Use keywords to be safe:
> `AntsApplyTransforms(3, input_image=moving, reference_image=fixed, output=out)`.
> (Registration is different: `antsRegistration` metrics *do* use fixed/moving, so
> commandants' metrics take `(fixed, moving)` — each tool follows the CLI it wraps.)

### How output transforms are saved

ANTs' transform-file naming trips everyone up. With `collapse_output_transforms`
on (standard), a **Rigid → Affine → SyN** run with `output="reg_"` writes:

| File | What it is |
|------|-----------|
| `reg_0GenericAffine.mat` | Rigid **and** Affine collapsed into one linear transform |
| `reg_1Warp.nii.gz` | SyN forward deformation (moving → fixed) |
| `reg_1InverseWarp.nii.gz` | SyN inverse deformation (fixed → moving) |

The number is the position in the *collapsed* stack, not the stage index — so the
two linear stages share index `0` and SyN is `1`. `reg.expected_transforms()`
predicts these filenames and hands back ready-to-apply `-t` lists:

```python
info = reg.expected_transforms()  # optionally pass cwd=... (where run() launches)
info["output_dir"]  # absolute folder the files land in
info["files_abs"]  # the files as absolute paths
info["forward"]  # ['reg_1Warp.nii.gz', 'reg_0GenericAffine.mat']  (moving -> fixed)
info["inverse"]  # [('reg_0GenericAffine.mat', True), 'reg_1InverseWarp.nii.gz']
```

The folder comes from your `output` prefix: a bare `"reg_"` lands in the working
directory at run time; `"/data/sub01/reg_"` lands in `/data/sub01/`. **ANTs will
not create a missing output folder** — make sure it exists (or pass `run(cwd=...)`).

Set `write_composite_transform=True` to get a single `reg_Composite.h5` /
`reg_InverseComposite.h5` instead. Full worked example:
[`examples/rigid_affine_syn.py`](examples/rigid_affine_syn.py).

### Preprocessing

```python
from commandants import N4BiasFieldCorrection

N4BiasFieldCorrection(
    3,
    "raw.nii.gz",
    "n4.nii.gz",
    bias_output="bias.nii.gz",
    shrink_factor=4,
    convergence_iterations=[50, 50, 50, 50],
    convergence_threshold=0.0,
    bspline_distance=200,
).run()
```

## elastix (benchmark against ANTs)

commandants also wraps [elastix](https://elastix.dev/) (`elastix` + `transformix`)
through the same core, so you can run both tools with one harness. elastix is
configured by **parameter maps** (not flags), which commandants models as
pass-through `ParameterMap`s plus curated presets.

### Getting elastix binaries

Exactly like ANTs: point `commandants` at an existing elastix (on `PATH`, via
`$ELASTIXPATH`, or `elastix_path=...`), **or** let it fetch the official prebuilt
binaries:

```bash
commandants install-elastix          # downloads official prebuilt elastix for your OS
commandants version                  # -> ... / elastix version: 5.3.1
commandants which elastix            # and: commandants which transformix
```

`install-elastix` downloads the matching archive from the
[SuperElastix/elastix releases](https://github.com/SuperElastix/elastix/releases)
(default `5.3.1`), unpacks it into a managed per-user directory (`commandants
list --tool elastix` shows where), and grabs `transformix` alongside `elastix`.
Resolution order mirrors ANTs — `elastix_path=` → `$ELASTIXPATH` → `PATH` →
managed — so a system elastix still wins if present.

> Unlike the ANTs CLI (and unlike the `SimpleElastix` Python package, which ships
> no command-line tools), elastix ships **shared libraries** next to its binaries.
> For a managed install, `commandants` records that `lib/` dir and injects it onto
> the loader path (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH`) automatically
> at run time. If you instead point at your own elastix via `$ELASTIXPATH`/`PATH`,
> make sure its `lib/` is discoverable the way the official archive lays it out.

`COMMANDANTS_AUTO_INSTALL=1` (or `auto_install=True`) opts elastix into the same
fetch-on-first-use behavior; `commandants install-elastix --version latest`,
`--asset <name>`, and `commandants uninstall-elastix` round out the CLI.

### Usage

```python
from commandants import elastix

reg = elastix.presets.affine("fixed.nii.gz", "moving.nii.gz", "out_dir")
print(reg.to_shell())  # elastix -f ... -m ... -out out_dir -p <param0>
result = reg.run(stream=True)  # writes out_dir/result.0.nii + TransformParameters.0.txt
print(result.duration_seconds)  # wall-clock time, for benchmarking vs ANTs

# apply the transform to another image:
elastix.Transformix("out_dir/TransformParameters.0.txt", "warp_out", moving="other.nii.gz").run()
```

Parameter maps are fully editable (`elastix.ParameterMap` / `presets.parameter_map("rigid")`),
round-trip elastix `.txt` files, and multiple maps become sequential `-p` stages.
A worked comparison: [`examples/benchmark_elastix_vs_ants.py`](examples/benchmark_elastix_vs_ants.py).

## Design

```
commandants/
  core/           binary resolution, the AntsCommand base, result objects
  registration/   AntsRegistration, metrics (incl. PSE/ICP/JHCT), transforms, stages, apply
  elastix/        Elastix, Transformix, ParameterMap, presets (elastix backend)
  preprocessing/  N4BiasFieldCorrection, ThresholdImage, ImageMath, ResampleImage
  io/             point-set CSV + SimpleITK in-memory support
  install.py      download/manage prebuilt ANTs + elastix binaries (ToolSpec)
  __main__.py     the `commandants` CLI
```

Every builder subclasses `AntsCommand`, so they all share `.build_command()`,
`.to_shell()`, `.run(...)`, `.extra_args(...)`, and a `CompletedAnts` result with
declared output paths and an optional lazy `.load()`.

## Status

v1 covers registration + core preprocessing. Segmentation (`Atropos`,
`KellyKapowski`) and motion correction (`antsMotionCorr`) are natural next
additions — the layered core makes them straightforward.

## License

MIT
