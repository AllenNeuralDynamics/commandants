"""Constrain a SyN warp with a corresponding point set.

This is the capability ANTsPyX does not expose. Run it with no ANTs installed --
it prints the exact command via ``dry_run`` so you can see the ``PSE[...]`` metric
that carries the point constraint.

    python examples/point_constrained_warp.py
"""

from __future__ import annotations

from commandants import (
    CC,
    MI,
    PSE,
    AntsRegistration,
    Convergence,
    Rigid,
    SyN,
)


def build() -> AntsRegistration:
    reg = AntsRegistration(
        dimensionality=3,
        output="out_",
        warped_output="out_Warped.nii.gz",
        write_composite_transform=True,
        verbose=True,
    )
    reg.initialize_from_images("fixed.nii.gz", "moving.nii.gz", init="center-of-mass")

    reg.add_stage(
        transform=Rigid(gradient_step=0.1),
        metrics=MI("fixed.nii.gz", "moving.nii.gz", weight=1.0, bins=32, sampling="Regular", sampling_pct=0.25),
        convergence=Convergence([1000, 500, 250, 100], threshold=1e-6, window=10),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )

    reg.add_stage(
        transform=SyN(gradient_step=0.1, update_field_variance=3, total_field_variance=0),
        metrics=[
            CC("fixed.nii.gz", "moving.nii.gz", weight=1.0, radius=4),
            # The point-set constraint on the deformation:
            PSE("fixed_points.csv", "moving_points.csv", weight=0.5, point_set_sigma=1.0, sampling_pct=0.5),
        ],
        convergence=Convergence([100, 70, 50, 20]),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )
    return reg


if __name__ == "__main__":
    reg = build()
    print("Command that would run:\n")
    print(reg.to_shell())
    # To actually run it (requires ANTs + real inputs):
    # result = reg.run()
    # print(result.stdout)
