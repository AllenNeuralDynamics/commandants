"""Register two .nii.gz images with a Rigid -> Affine -> SyN pipeline, and show
exactly how ANTs saves the output transforms.

Run it with no ANTs installed -- it prints the command (via dry_run) and the exact
transform filenames that will be produced:

    python examples/rigid_affine_syn.py [fixed.nii.gz] [moving.nii.gz]
"""

from __future__ import annotations

import sys

from commandants import (
    CC,
    MI,
    Affine,
    AntsApplyTransforms,
    AntsRegistration,
    Convergence,
    Rigid,
    SyN,
)


def build_registration(fixed: str, moving: str, prefix: str = "reg_") -> AntsRegistration:
    reg = AntsRegistration(
        dimensionality=3,
        output=prefix,
        warped_output=f"{prefix}Warped.nii.gz",
        inverse_warped_output=f"{prefix}InverseWarped.nii.gz",
        # collapse_output_transforms=True is the ANTs-standard behavior: the Rigid
        # and Affine stages are combined into ONE GenericAffine .mat on disk.
        collapse_output_transforms=True,
        # Flip this to True to instead get a single reg_Composite.h5 /
        # reg_InverseComposite.h5 bundling the whole chain (see notes below).
        write_composite_transform=False,
        interpolation="Linear",
        winsorize=(0.005, 0.995),
        use_histogram_matching=True,
        verbose=True,
    )

    # Align image centers of mass first (-> --initial-moving-transform [f,m,1]).
    reg.initialize_from_images(fixed, moving, init="center-of-mass")

    # Stage 1: Rigid (Mattes MI).
    reg.add_stage(
        transform=Rigid(gradient_step=0.1),
        metrics=MI(fixed, moving, weight=1.0, bins=32, sampling="Regular", sampling_pct=0.25),
        convergence=Convergence([1000, 500, 250, 100], threshold=1e-6, window=10),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )

    # Stage 2: Affine (Mattes MI).
    reg.add_stage(
        transform=Affine(gradient_step=0.1),
        metrics=MI(fixed, moving, weight=1.0, bins=32, sampling="Regular", sampling_pct=0.25),
        convergence=Convergence([1000, 500, 250, 100], threshold=1e-6, window=10),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )

    # Stage 3: SyN deformable (neighborhood cross-correlation).
    reg.add_stage(
        transform=SyN(gradient_step=0.1, update_field_variance=3, total_field_variance=0),
        metrics=CC(fixed, moving, weight=1.0, radius=4),
        convergence=Convergence([100, 70, 50, 20], threshold=1e-6, window=10),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )
    return reg


def main() -> None:
    fixed = sys.argv[1] if len(sys.argv) > 1 else "fixed.nii.gz"
    moving = sys.argv[2] if len(sys.argv) > 2 else "moving.nii.gz"
    prefix = "reg_"

    reg = build_registration(fixed, moving, prefix)

    print("=" * 70)
    print("Command that will run:")
    print("=" * 70)
    print(reg.to_shell())

    info = reg.expected_transforms()
    print("\n" + "=" * 70)
    print("How the output transforms are saved")
    print("=" * 70)
    print(
        f"""
With output prefix {prefix!r} and collapse_output_transforms=True, ANTs writes:

  {prefix}0GenericAffine.mat   <- Rigid AND Affine collapsed into one linear .mat
  {prefix}1Warp.nii.gz         <- SyN forward deformation field (moving -> fixed)
  {prefix}1InverseWarp.nii.gz  <- SyN inverse deformation field (fixed -> moving)
  {prefix}Warped.nii.gz        <- moving resampled into fixed space
  {prefix}InverseWarped.nii.gz <- fixed resampled into moving space

The number is the transform's position in the (collapsed) stack, NOT the stage
index -- that's why the two linear stages share index 0 and SyN is index 1.

Predicted by reg.expected_transforms():
  output_dir : {info["output_dir"]}
               (a bare prefix like 'reg_' lands here, the working directory at
                run time; ANTs will NOT create this folder -- make sure it exists.
                Use output='/path/to/out/reg_' or reg.run(cwd=...) to place them.)
  files      : {info["files"]}
  forward    : {info["forward"]}   (moving -> fixed; feed straight to -t)
  inverse    : {info["inverse"]}   (fixed -> moving; affine is inverted)
"""
    )

    print("=" * 70)
    print("Applying the result")
    print("=" * 70)

    # Warp moving INTO fixed space using the forward transform list. Note the
    # order: warp first, then affine (ANTs applies the -t list right-to-left).
    fwd = AntsApplyTransforms(3, moving, fixed, "moving_in_fixed.nii.gz", interpolation="Linear")
    for t in info["forward"]:
        fwd.add_transform(t)
    print("\nforward (moving -> fixed):")
    print(fwd.to_shell())

    # Warp fixed INTO moving space: inverse warp + inverted affine.
    inv = AntsApplyTransforms(3, fixed, moving, "fixed_in_moving.nii.gz", interpolation="Linear")
    for t in info["inverse"]:
        if isinstance(t, tuple):
            path, invert = t
            inv.add_transform(path, invert=invert)
        else:
            inv.add_transform(t)
    print("\ninverse (fixed -> moving):")
    print(inv.to_shell())

    print(
        """
Tip: set write_composite_transform=True on the AntsRegistration to instead get a
single reg_Composite.h5 (and reg_InverseComposite.h5). Then applying is just:

    AntsApplyTransforms(3, moving, fixed, out).add_transform("reg_Composite.h5")

To actually run: reg.run()  (requires ANTs -- e.g. `commandants install-ants`).
"""
    )


if __name__ == "__main__":
    main()
