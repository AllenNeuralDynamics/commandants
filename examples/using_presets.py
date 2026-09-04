"""Using the ANTsPyX-style preset builders.

Each preset returns a ready-to-run AntsRegistration, so you keep the
inspect -> estimate -> run workflow. Runs with no ANTs installed: it prints the
commands the presets generate.

    python examples/using_presets.py [fixed.nii.gz] [moving.nii.gz]
"""

from __future__ import annotations

import sys

from commandants import presets


def main() -> None:
    fixed = sys.argv[1] if len(sys.argv) > 1 else "fixed.nii.gz"
    moving = sys.argv[2] if len(sys.argv) > 2 else "moving.nii.gz"

    # --- 1. The simplest thing: a full rigid+affine+SyN pipeline --------------
    # (center-of-mass init included by default, so no identity-transform surprise)
    reg = presets.syn(
        fixed,
        moving,
        "out_",
        warped_output="out_Warped.nii.gz",
        use_float=True,  # single precision (half the memory) -- the default
        verbose=True,
    )
    print("=" * 70, "\npresets.syn  (Rigid -> Affine -> SyN, COM init)\n", "=" * 70, sep="")
    print(reg.to_shell())
    print("\ntransform files it will write:", reg.expected_transforms()["files"])

    # To run it (needs ANTs; stream progress + tee to a log):
    #     result = reg.run(stream=True, log_file="reg.log")

    # --- 2. Rigid-only, then affine-only --------------------------------------
    print("\n" + "=" * 70, "\npresets.rigid / presets.affine\n", "=" * 70, sep="")
    print("rigid :", presets.rigid(fixed, moving, "rig_").to_shell())
    print("\naffine:", presets.affine(fixed, moving, "aff_").to_shell())

    # --- 3. Memory-safe SyN for a very large volume ---------------------------
    # Drop the finest (shrink=1) level off the SyN stage; keep everything float.
    print("\n" + "=" * 70, "\nMemory-safe SyN for a huge image\n", "=" * 70, sep="")
    big = presets.syn(
        fixed,
        moving,
        "big_",
        syn_shrink_factors=(8, 4, 2),  # no full-res level
        syn_smoothing_sigmas=(3, 2, 1),
        syn_iterations=(100, 70, 50),
    )
    est = big.estimate_resources(shape=(1570, 1050, 1390))  # your volume
    print("estimated peak memory:", est.peak_memory_human)

    # --- 4. Two-step: affine, then SyN-only warm-started from that affine -----
    # Useful when you want to reuse / inspect the affine, or split the work.
    print("\n" + "=" * 70, "\nAffine, then SyN-only warm-started from it\n", "=" * 70, sep="")
    aff = presets.affine(fixed, moving, "step1_")
    # After running `aff`, its linear result is step1_0GenericAffine.mat:
    affine_mat = aff.expected_transforms()["files"][0]
    warp = presets.syn_only(
        fixed,
        moving,
        "step2_",
        initial_transform=affine_mat,  # prepend the affine (no COM needed)
    )
    print("affine mat:", affine_mat)
    print("syn_only  :", warp.to_shell())

    # --- 5. Override the metric (e.g. CC for the deformable part) -------------
    print("\n" + "=" * 70, "\nOverride the SyN metric to CC\n", "=" * 70, sep="")
    cc = presets.syn(fixed, moving, "cc_", syn_metric="cc", syn_radius=4)
    print(cc.to_shell())


if __name__ == "__main__":
    main()
