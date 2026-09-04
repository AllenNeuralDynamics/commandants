"""Two ways to mask a registration: all stages vs one stage (e.g. SyN only).

ANTs binds masks to stages by order, so masking a single stage means the others
emit the [NA,NA] placeholder -- commandants handles that for you. Runs without ANTs
(prints the generated commands).

    python examples/masking.py [fixed.nii.gz] [moving.nii.gz] [mask.nii.gz]
"""

from __future__ import annotations

import re
import sys

from commandants import MI, Affine, AntsRegistration, Convergence, Rigid, SyN


def build_base(fixed: str, moving: str) -> AntsRegistration:
    """A Rigid -> Affine -> SyN registration with no masks yet."""
    reg = AntsRegistration(3, output="reg_", warped_output="reg_Warped.nii.gz", verbose=True)
    reg.initialize_from_images(fixed, moving, init="center-of-mass")
    reg.add_stage(
        Rigid(0.25),
        MI(fixed, moving, bins=32, sampling="Regular", sampling_pct=0.25),
        Convergence([2100, 1200, 1200, 10]),
        [6, 4, 2, 1],
        [3, 2, 1, 0],
    )
    reg.add_stage(
        Affine(0.25),
        MI(fixed, moving, bins=32, sampling="Regular", sampling_pct=0.25),
        Convergence([2100, 1200, 1200, 10]),
        [6, 4, 2, 1],
        [3, 2, 1, 0],
    )
    reg.add_stage(
        SyN(0.25, 3, 0),
        MI(fixed, moving, bins=32, sampling="Regular"),
        Convergence([100, 70, 50, 20]),
        [6, 4, 2, 1],
        [3, 2, 1, 0],
    )
    return reg


def show_masks(title: str, reg: AntsRegistration) -> None:
    argv = reg.build_command()
    stages = [a for a in argv if a in ("Rigid[0.25]", "Affine[0.25]", "SyN[0.25,3,0]")]
    masks = re.findall(r"--masks (\S+)", reg.to_shell())
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print("stages:", [s.split("[")[0] for s in stages])
    print("--masks (in stage order):", masks if masks else "(none)")


def main() -> None:
    fixed = sys.argv[1] if len(sys.argv) > 1 else "fixed.nii.gz"
    moving = sys.argv[2] if len(sys.argv) > 2 else "moving.nii.gz"
    mask = sys.argv[3] if len(sys.argv) > 3 else "brain_mask.nii.gz"

    # (1) ALL STAGES: one global mask via set_masks() -> a single --masks, applied
    #     to every stage. Here a moving-side mask (fixed side left as NA).
    all_stages = build_base(fixed, moving)
    all_stages.set_masks(moving_mask=mask)
    show_masks("(1) All stages  -- reg.set_masks(moving_mask=...)", all_stages)

    # (2) SYN ONLY: mask just the final (SyN) stage via add_stage(moving_mask=...).
    #     Rigid/Affine automatically get the [NA,NA] placeholder ANTs requires.
    syn_only = build_base(fixed, moving)
    syn_only.stages[-1].moving_mask = mask  # or pass moving_mask=... to add_stage
    show_masks("(2) SyN only    -- moving_mask on the last stage", syn_only)

    # (3) BONUS: global default + a per-stage override (fixed+moving on SyN).
    mixed = build_base(fixed, moving)
    mixed.set_masks(fixed_mask="head_mask.nii.gz")  # default fixed mask, all stages
    mixed.stages[-1].moving_mask = mask  # add a moving mask on SyN only
    show_masks("(3) Global fixed default + moving mask on SyN", mixed)

    print("\n" + "=" * 70)
    print("Full command for case (2):")
    print("=" * 70)
    print(syn_only.to_shell())


if __name__ == "__main__":
    main()
