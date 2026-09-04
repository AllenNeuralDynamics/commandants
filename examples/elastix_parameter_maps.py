"""Full elastix example: build, tweak, save, and run parameter maps by hand.

The ``elastix.presets`` builders are the quick path; this is the *transparent* path --
the elastix analogue of hand-composing ``antsRegistration`` stages. elastix's whole
algorithm lives in its parameter files (one ``(Key value)`` per line), and commandants
models that file as :class:`~commandants.elastix.ParameterMap`. Three ways to make one,
all shown below:

  1. from scratch, key by key (you never need a preset);
  2. start from a curated preset and tweak it (the common case);
  3. load an existing ``.txt``, edit it, write it back.

Nothing is hidden -- a ``ParameterMap`` round-trips the exact text elastix reads, and
multiple maps become repeated ``-p`` flags (sequential multi-stage registration).

Runs end-to-end if elastix is installed (``commandants install-elastix``); otherwise it
writes the parameter files, prints them, and prints the exact commands (no elastix
needed) so you can still inspect everything.

    python examples/elastix_parameter_maps.py [fixed.nii.gz] [moving.nii.gz]
"""

from __future__ import annotations

import os
import sys
import tempfile

from commandants import elastix
from commandants.core.executable import is_available
from commandants.elastix import ParameterMap, presets


def make_maps() -> tuple[ParameterMap, ParameterMap, ParameterMap]:
    """Build three stages (rigid -> affine -> b-spline), each a different way."""
    # 1) FROM SCRATCH -- a minimal rigid map, key by key. No preset involved.
    rigid = ParameterMap()
    rigid.update(
        {
            "Registration": "MultiResolutionRegistration",
            "Transform": "EulerTransform",  # rigid (rotation + translation)
            "Metric": "AdvancedMattesMutualInformation",
            "Optimizer": "AdaptiveStochasticGradientDescent",
            "Interpolator": "BSplineInterpolator",
            "ResampleInterpolator": "FinalBSplineInterpolator",
            "Resampler": "DefaultResampler",
            "FixedImagePyramid": "FixedSmoothingImagePyramid",
            "MovingImagePyramid": "MovingSmoothingImagePyramid",
            "ImageSampler": "RandomCoordinate",
            "NumberOfResolutions": 3,
            "MaximumNumberOfIterations": 256,
            "NumberOfSpatialSamples": 2048,
            "AutomaticScalesEstimation": True,  # python bool -> "true"
            "AutomaticTransformInitialization": True,
            "ResultImageFormat": "nii",
        }
    )
    # .set() is fluent (returns self) and takes multiple tokens -- e.g. a per-resolution,
    # per-axis smoothing schedule: 3 resolutions x 3 axes = 9 values.
    rigid.set("FixedImagePyramidSchedule", 4, 4, 4, 2, 2, 2, 1, 1, 1)

    # 2) START FROM A PRESET AND TWEAK IT (the usual case).
    affine = presets.parameter_map("affine")
    affine.set("MaximumNumberOfIterations", 512)  # run the optimizer longer
    affine.set("NumberOfResolutions", 5)  # one extra pyramid level
    affine["NumberOfSpatialSamples"] = 4096  # dict-style assignment works too
    affine.set("WriteIterationInfo", True)  # add a key the preset didn't have
    # Values are stored as tuples; read one back with [key][0]:
    print(f"  affine: {affine['MaximumNumberOfIterations'][0]} iters, {affine['NumberOfResolutions'][0]} resolutions")

    # 3) B-SPLINE stage, also from a preset, with a tuned control-point grid.
    bspline = presets.parameter_map("bspline")
    bspline.set("FinalGridSpacingInPhysicalUnits", 12)  # denser warp (smaller = more DOF)
    bspline.set("MaximumNumberOfIterations", 400)

    return rigid, affine, bspline


def demo_round_trip(pm: ParameterMap) -> None:
    """A ParameterMap is exactly its text: build -> to_text -> parse is lossless."""
    text = pm.to_text()
    reparsed = ParameterMap.parse(text)
    assert reparsed["Transform"] == pm["Transform"]
    # ...and you can load someone else's .txt, edit it, and hand it back:
    #   pm = ParameterMap.from_file("Par0001affine.txt"); pm.set("Metric", "...")


def synth_pair(workdir: str) -> tuple[str, str]:
    """Make a tiny synthetic fixed/moving pair (needs the [io] extra)."""
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover
        sys.exit(
            "No input images given and SimpleITK/numpy aren't installed.\n"
            "Pass real files:  python examples/elastix_parameter_maps.py fixed.nii.gz moving.nii.gz\n"
            "or:               pip install 'commandants[io]'"
        )

    def blob(shift: int) -> "sitk.Image":
        a = np.zeros((64, 64, 64), np.float32)
        a[16 + shift : 40 + shift, 16 + shift : 40 + shift, 16:40] = 100.0
        return sitk.GetImageFromArray(a)

    fixed = os.path.join(workdir, "fixed.nii.gz")
    moving = os.path.join(workdir, "moving.nii.gz")
    sitk.WriteImage(blob(0), fixed)
    sitk.WriteImage(blob(6), moving)
    return fixed, moving


def main() -> None:
    out = tempfile.mkdtemp(prefix="elastix_full_")
    if len(sys.argv) >= 3:
        fixed, moving = sys.argv[1], sys.argv[2]
    else:
        print("No input images given -- synthesizing a tiny pair.")
        fixed, moving = synth_pair(out)

    print("\nBuilding parameter maps:")
    rigid, affine, bspline = make_maps()
    demo_round_trip(affine)

    # Save the maps as .txt so they're inspectable and reproducible. You can also pass
    # the ParameterMap objects straight to Elastix (commandants writes temp files for
    # you); saving them just makes the command concrete and dry-run friendly.
    stages = [("Rigid", rigid), ("Affine", affine), ("BSpline", bspline)]
    paths = [pm.write(os.path.join(out, f"{name}.txt")) for name, pm in stages]

    print("\n--- Rigid.txt (built from scratch) ---")
    print(rigid.to_text().rstrip())

    # Three sequential stages: rigid -> affine -> b-spline (one repeated -p each).
    reg = elastix.Elastix(
        fixed,
        moving,
        out,
        paths,  # a list -> sequential stages; pass ParameterMap objects here too
        threads=4,
        # fixed_mask=..., moving_mask=...,   # optional metric masks (path or SimpleITK image)
        # initial_transform="prior/TransformParameters.0.txt",  # -t0 warm start
    )

    print("\n--- elastix command ---")
    print(reg.to_shell())
    print("\n--- declared outputs ---")
    for key, path in reg.declared_outputs().items():
        print(f"  {key}: {path}")

    if not is_available("elastix"):
        print("\n(elastix not installed -- run `commandants install-elastix` to execute.)")
        # transformix applies the *final* stage's transform; it chains back through
        # the earlier stages via each file's InitialTransformParametersFileName.
        tp = os.path.join(out, f"TransformParameters.{len(paths) - 1}.txt")
        tx = elastix.Transformix(tp, os.path.join(out, "applied"), moving=moving, deformation_field=True)
        print("\n--- transformix command (would run after elastix) ---")
        print(tx.to_shell())
        print(f"\nparameter files + would-be outputs under: {out}")
        return

    print("\nRunning elastix...")
    r = reg.run(stream=True)
    print(f"  returncode={r.returncode}  time={r.duration_seconds:.2f}s")
    print(f"  warped:    {r.outputs.get('warped')}")
    print(f"  transform: {r.outputs.get('transform')}")

    # Apply the final transform to the moving image (or a label map, another channel,
    # a point set, ...). Here we also write the deformation field and the Jacobian.
    tp = r.outputs["transform"]
    tx = elastix.Transformix(tp, os.path.join(out, "applied"), moving=moving, jacobian=True)
    print("\n--- transformix command ---")
    print(tx.to_shell())
    rr = tx.run()
    print(f"  returncode={rr.returncode}")
    print(f"  warped:   {rr.outputs.get('warped')}")
    print(f"  jacobian: {rr.outputs.get('jacobian')}")
    print(f"\noutputs under: {out}")


if __name__ == "__main__":
    main()
