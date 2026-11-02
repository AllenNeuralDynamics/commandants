"""Benchmark every available registration backend on the same image pair.

Registers one (synthetic, or your own) image pair with each backend registered in
``commandants.benchmark`` -- ANTs and elastix today, plus any others that have been
added -- and reports wall-clock time with normalized output paths.

The comparison is apples-to-apples because every backend goes through the same
``run()`` path (so timing is measured identically), and the harness additionally pins
threads, disables streaming, equalizes the warped-image format, and isolates failures
so one missing tool can't discard the others' numbers.

Accuracy analysis (Dice on labels, landmark error, ...) is your science and is out of
scope here -- the harness just hands you each backend's warped image and transforms at
predictable paths so you can score them yourself.

    python examples/benchmark_elastix_vs_ants.py [fixed.nii.gz] [moving.nii.gz]
"""

from __future__ import annotations

import os
import sys
import tempfile

from commandants import benchmark


def synth_pair(workdir: str) -> tuple[str, str]:
    """Write a tiny synthetic fixed/moving pair (needs the [io] extra)."""
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover
        sys.exit(
            "No input images given and SimpleITK/numpy aren't installed.\n"
            "Pass real files:  python examples/benchmark_elastix_vs_ants.py fixed.nii.gz moving.nii.gz\n"
            "or:               pip install 'commandants[io]'"
        )

    def blob(shift: int):
        a = np.zeros((64, 64, 64), np.float32)
        a[16 + shift : 40 + shift, 16 + shift : 40 + shift, 16:40] = 100.0
        return sitk.GetImageFromArray(a)

    fixed = os.path.join(workdir, "fixed.nii.gz")
    moving = os.path.join(workdir, "moving.nii.gz")
    sitk.WriteImage(blob(0), fixed)
    sitk.WriteImage(blob(6), moving)
    return fixed, moving


def main() -> None:
    out = tempfile.mkdtemp(prefix="bench_")
    if len(sys.argv) >= 3:
        fixed, moving = sys.argv[1], sys.argv[2]
    else:
        print("No input images given -- synthesizing a tiny pair.")
        fixed, moving = synth_pair(out)

    known = sorted(benchmark.BACKENDS)
    available = benchmark.available_backends()
    print(f"registered backends: {known}")
    print(f"available right now: {available or '(none -- install ANTs and/or elastix)'}")
    if not available:
        return

    # warmup absorbs page-cache / binary-load effects; threads are pinned for fairness.
    report = benchmark.compare(
        fixed,
        moving,
        os.path.join(out, "results"),
        backends=known,
        preset="affine",
        threads=4,
        repeats=3,
        warmup=1,
    )

    print("\n" + report.to_table())

    print("\nper-trial outputs:")
    for trial in report.timed():
        print(f"  [{trial.backend}] {trial.duration_seconds:.2f}s -> {trial.warped}")
        for path in trial.transforms:
            print(f"      transform: {path}")

    csv_path = report.to_csv(os.path.join(out, "benchmark.csv"))
    print(f"\nrows written to: {csv_path}")
    print(f"outputs under:   {out}")


if __name__ == "__main__":
    main()
