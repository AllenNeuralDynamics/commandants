"""Apply an already-fitted transform to (1) a new image and (2) a set of points.

The subtlety: images and points move in OPPOSITE directions for the same
transform.

  * An ITK/ANTs transform maps output(fixed) -> input(moving) *coordinates*.
  * Resampling an IMAGE with the FORWARD transforms therefore carries image
    content moving -> fixed:      -t Warp -t GenericAffine
  * Applying those same transforms to a POINT sends it fixed -> moving (the
    opposite way). So to carry POINTS moving -> fixed you use the INVERSE
    transforms:                   -t [GenericAffine,1] -t InverseWarp

So: same conceptual direction (moving -> fixed), but images use the forward list
and points use the inverse list.

Runs without ANTs (prints the commands); if the inputs and ANTs are present it
actually executes. Point I/O uses commandants.io.

    python examples/apply_transform_to_image_and_points.py
"""

from __future__ import annotations

import os

from commandants import AntsApplyTransforms, AntsApplyTransformsToPoints
from commandants.io import read_points, write_points

# --- what you fitted earlier -------------------------------------------------
# A SyN registration (moving -> fixed) with prefix "reg_" produced these.
# For a rigid/affine-only fit you'd have ONLY the .mat (see the note below).
FIXED = "fixed.nii.gz"  # reference grid for the output image
AFFINE = "reg_0GenericAffine.mat"
WARP = "reg_1Warp.nii.gz"
INVERSE_WARP = "reg_1InverseWarp.nii.gz"

# The new data you want to move (both live in the MOVING image's space):
NEW_IMAGE = "new_moving_channel.nii.gz"  # e.g. another channel/modality
POINTS_IN = "points_moving.csv"  # points in the moving image's space
POINTS_OUT = "points_in_fixed.csv"
IMAGE_OUT = "new_in_fixed.nii.gz"


def main() -> None:
    # ------------------------------------------------------------------ #
    # (1) IMAGE: moving -> fixed, using the FORWARD transforms.
    #     Order matters: warp listed first, affine second (ANTs applies the
    #     -t list right-to-left, so the affine is applied to the grid first).
    # ------------------------------------------------------------------ #
    apply_img = AntsApplyTransforms(
        3,
        NEW_IMAGE,
        FIXED,
        IMAGE_OUT,
        interpolation="Linear",  # use "genericLabel" instead for a label image
    )
    apply_img.add_transform(WARP)  # forward warp
    apply_img.add_transform(AFFINE)  # forward affine (applied first)
    print("=" * 70, "\n(1) Image  moving -> fixed  (FORWARD transforms)\n", "=" * 70, sep="")
    print(apply_img.to_shell())

    # ------------------------------------------------------------------ #
    # (2) POINTS: moving -> fixed, using the INVERSE transforms.
    #     Invert the affine ([mat,1]) and use the InverseWarp, and reverse the
    #     order relative to the image call.
    # ------------------------------------------------------------------ #
    # Make a small demo point set in the moving image's physical space (mm).
    if not os.path.exists(POINTS_IN):
        write_points(POINTS_IN, [[10.0, 12.0, 8.0], [4.5, 7.0, 15.0]], labels=[1, 2])

    apply_pts = AntsApplyTransformsToPoints(3, POINTS_IN, POINTS_OUT)
    apply_pts.add_transform(AFFINE, invert=True)  # inverse affine
    apply_pts.add_transform(INVERSE_WARP)  # inverse warp (applied first)
    print("\n" + "=" * 70, "\n(2) Points moving -> fixed  (INVERSE transforms)\n", "=" * 70, sep="")
    print(apply_pts.to_shell())

    print(
        """
Note:
  * If you have the AntsRegistration object still, skip the bookkeeping:
        info = reg.expected_transforms()
        # image  moving->fixed:  info["forward"]  (list of paths)
        # points moving->fixed:  info["inverse"]  (affine entries are (path, True))
  * Rigid/affine-only fit (no warp): the image uses  -t reg_0GenericAffine.mat
    and the points use  -t [reg_0GenericAffine.mat,1]  (just invert the affine).
  * Going the other way (fixed->moving) flips which list each one uses.
"""
    )

    # Actually run, only if the inputs and ANTs are present.
    from commandants import is_available

    have_inputs = all(os.path.exists(p) for p in (FIXED, AFFINE, WARP, INVERSE_WARP, NEW_IMAGE))
    if have_inputs and is_available("antsApplyTransforms"):
        apply_img.run()
        apply_pts.run()
        coords, extra = read_points(POINTS_OUT, as_array=False)
        print("transformed points (in fixed space):", coords, extra)
    else:
        print("(inputs or ANTs not found -- printed commands only)")


if __name__ == "__main__":
    main()
