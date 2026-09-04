"""Benchmark elastix against ANTs through the same commandants machinery.

Runs an affine registration with each backend on the same (synthetic) image pair
and reports wall-clock time (``result.duration_seconds``) and the output paths --
apples-to-apples, since both go through the same run/measurement code. If a backend
isn't installed, its command is printed instead of run.

Accuracy analysis (Dice on labels, landmark error, ...) is your science and is out
of scope for the wrapper -- this just gives you the harness + timing.

    python examples/benchmark_elastix_vs_ants.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import SimpleITK as sitk

import commandants as cants
from commandants import elastix
from commandants.core.executable import is_available


def _blob(shift: int = 0) -> sitk.Image:
    a = np.zeros((64, 64, 64), np.float32)
    a[16 + shift:40 + shift, 16 + shift:40 + shift, 16:40] = 100.0
    return sitk.GetImageFromArray(a)


def main() -> None:
    d = tempfile.mkdtemp(prefix="bench_")
    fixed = os.path.join(d, "fixed.nii.gz"); sitk.WriteImage(_blob(0), fixed)
    moving = os.path.join(d, "moving.nii.gz"); sitk.WriteImage(_blob(6), moving)

    # --- ANTs affine ------------------------------------------------------
    ants_out = os.path.join(d, "ants_")
    ants_reg = cants.presets.affine(fixed, moving, ants_out,
                                    warped_output=ants_out + "warped.nii.gz")
    print("=" * 66, "\nANTs affine\n", "=" * 66, sep="")
    if is_available("antsRegistration"):
        r = ants_reg.run()
        print(f"  returncode={r.returncode}  time={r.duration_seconds:.2f}s")
        print(f"  warped: {r.outputs.get('warped')}")
    else:
        print("  (ANTs not installed -- command:)")
        print("  " + ants_reg.to_shell())

    # --- elastix affine ---------------------------------------------------
    el_out = os.path.join(d, "elastix_out")
    el_reg = elastix.presets.affine(fixed, moving, el_out)
    print("\n" + "=" * 66, "\nelastix affine\n", "=" * 66, sep="")
    if is_available("elastix"):
        r = el_reg.run()
        print(f"  returncode={r.returncode}  time={r.duration_seconds:.2f}s")
        print(f"  warped: {r.outputs.get('warped')}")
        print(f"  transform: {r.outputs.get('transform')}")
    else:
        print("  (elastix not installed -- run `commandants install-elastix`; command:)")
        print("  " + el_reg.to_shell())

    print(f"\noutputs under: {d}")


if __name__ == "__main__":
    main()
