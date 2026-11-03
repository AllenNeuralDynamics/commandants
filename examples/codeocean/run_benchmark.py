#!/usr/bin/env python3
"""Three-way registration benchmark for a Code Ocean capsule.

Registers one image pair with every backend that is installed (ANTs, elastix, and the
Allen STPT/CCF driver) and writes a table plus machine-readable rows to /results.

Inputs: the first ``*fixed*.nii.gz`` / ``*moving*.nii.gz`` pair found under /data. If
there isn't one, a textured synthetic pair is generated so the capsule still proves the
toolchain end to end.

    python run_benchmark.py [fixed.nii.gz moving.nii.gz]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from commandants import benchmark

DATA = Path("/data")
RESULTS = Path("/results")
PRESET = os.environ.get("PRESET", "affine")
THREADS = int(os.environ.get("THREADS", "4"))
REPEATS = int(os.environ.get("REPEATS", "1"))


def find_pair():
    """First fixed/moving pair under /data, else (None, None)."""
    if not DATA.is_dir():
        return None, None
    fixed = sorted(DATA.rglob("*fixed*.nii*"))
    moving = sorted(DATA.rglob("*moving*.nii*"))
    if fixed and moving:
        return str(fixed[0]), str(moving[0])
    return None, None


def synthetic(workdir: Path, n: int = 96):
    """Textured phantom plus a known nonlinear warp.

    Texture matters: the STPT similarity term is an uncentered cross-correlation, which
    is degenerate inside large uniform regions.
    """
    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    a = (
        60 * np.sin(xx / 7) * np.sin(yy / 9) * np.sin(zz / 8)
        + 40 * np.sin(xx / 3 + yy / 4)
        + 30 * np.cos(yy / 5 + zz / 6)
    )
    k = (
        np.fft.fftfreq(n)[:, None, None] ** 2
        + np.fft.fftfreq(n)[None, :, None] ** 2
        + np.fft.fftfreq(n)[None, None, :] ** 2
    )
    sm = np.real(np.fft.ifftn(np.fft.fftn(rng.normal(0, 1, (n, n, n))) * np.exp(-k * 400)))
    a = a + 40 * sm / sm.std() + 120
    r2 = ((xx - n / 2) / (n * 0.33)) ** 2 + ((yy - n / 2) / (n * 0.30)) ** 2 + ((zz - n / 2) / (n * 0.28)) ** 2
    fixed = sitk.GetImageFromArray(np.ascontiguousarray(np.where(r2 < 1, a, 0).astype(np.float32)))

    dx = 8 * np.sin(2 * np.pi * 2 * yy / n) * np.sin(np.pi * zz / n)
    dy = 8 * np.sin(2 * np.pi * 2 * xx / n) * np.sin(np.pi * zz / n)
    field = np.stack([dx, dy, np.zeros_like(dx)], -1).astype(np.float64)
    disp = sitk.GetImageFromArray(field, isVector=True)
    disp.CopyInformation(fixed)
    moving = sitk.Resample(
        fixed, fixed, sitk.DisplacementFieldTransform(sitk.Cast(disp, sitk.sitkVectorFloat64)), sitk.sitkLinear, 0.0
    )

    fp = workdir / "fixed.nii.gz"
    mp = workdir / "moving.nii.gz"
    sitk.WriteImage(fixed, str(fp))
    sitk.WriteImage(moving, str(mp))
    return str(fp), str(mp)


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else float("nan")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("commandants backends registered:", sorted(benchmark.BACKENDS))
    available = benchmark.available_backends()
    print("available in this capsule   :", available)
    if not available:
        print("FATAL: no registration backend resolved. Check ANTSPATH/ELASTIXPATH/STPTPATH.")
        return 1

    if len(sys.argv) >= 3:
        fixed, moving = sys.argv[1], sys.argv[2]
    else:
        fixed, moving = find_pair()
        if fixed is None:
            print("no fixed/moving pair under /data -- using a synthetic textured pair")
            fixed, moving = synthetic(RESULTS)
    print("fixed :", fixed)
    print("moving:", moving)
    print()

    report = benchmark.compare(
        fixed,
        moving,
        str(RESULTS / "registrations"),
        backends=tuple(available),
        preset=PRESET,
        threads=THREADS,
        repeats=REPEATS,
        warmup=0,
    )

    table = report.to_table()
    print(table)
    (RESULTS / "benchmark_table.txt").write_text(table + "\n", encoding="utf-8")
    report.to_csv(RESULTS / "benchmark.csv")

    # Timing alone does not say whether a backend aligned anything, so score them too.
    print()
    ref = sitk.GetArrayFromImage(sitk.ReadImage(fixed))
    mov = sitk.GetArrayFromImage(sitk.ReadImage(moving))
    print("NCC before: %.4f" % ncc(mov, ref))
    lines = ["backend,duration_s,ncc_after"]
    for trial in report.timed():
        warped = sitk.GetArrayFromImage(sitk.ReadImage(trial.warped))
        value = ncc(warped, ref)
        print("  %-9s %7.2fs   NCC after %.4f" % (trial.backend, trial.duration_seconds, value))
        lines.append("%s,%.3f,%.4f" % (trial.backend, trial.duration_seconds, value))
    (RESULTS / "quality.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [t for t in report.trials if not t.ok]
    for t in failures:
        print("  FAILED %-9s %s" % (t.backend, (t.error or "")[:160]))

    print()
    print("wrote:", ", ".join(sorted(p.name for p in RESULTS.glob("*.csv"))), "+ benchmark_table.txt")
    # A backend that is installed but cannot run is a real failure worth surfacing.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
