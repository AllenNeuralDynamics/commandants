"""Command-construction tests for AntsRegistration -- including the PSE point
constraint, the headline capability ANTsPyX cannot express."""

from __future__ import annotations

import pytest

from commandants import (
    CC,
    MI,
    PSE,
    Affine,
    AntsRegistration,
    Convergence,
    Rigid,
    SyN,
)
from commandants.registration import Stage
from commandants.registration.metrics import is_point_set_metric


def _two_stage_reg() -> AntsRegistration:
    reg = AntsRegistration(
        dimensionality=3,
        output="out_",
        warped_output="out_Warped.nii.gz",
        write_composite_transform=True,
        verbose=False,
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
            PSE("fixed_points.csv", "moving_points.csv", weight=0.5, point_set_sigma=1.0, sampling_pct=0.5),
        ],
        convergence=Convergence([100, 70, 50, 20]),
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[3, 2, 1, 0],
    )
    return reg


def test_full_two_stage_command_with_point_constraint():
    argv = _two_stage_reg().build_command()
    expected = [
        "antsRegistration",
        "--dimensionality",
        "3",
        "--float",
        "1",  # single precision is the default
        "--output",
        "[out_,out_Warped.nii.gz]",
        "--write-composite-transform",
        "1",
        "--initial-moving-transform",
        "[fixed.nii.gz,moving.nii.gz,1]",
        # stage 1
        "--transform",
        "Rigid[0.1]",
        "--metric",
        "MI[fixed.nii.gz,moving.nii.gz,1.0,32,Regular,0.25]",
        "--convergence",
        "[1000x500x250x100,1e-06,10]",
        "--shrink-factors",
        "8x4x2x1",
        "--smoothing-sigmas",
        "3x2x1x0vox",
        # stage 2 -- CC image metric + PSE point-set constraint
        "--transform",
        "SyN[0.1,3,0]",
        "--metric",
        "CC[fixed.nii.gz,moving.nii.gz,1.0,4]",
        "--metric",
        "PSE[fixed_points.csv,moving_points.csv,0.5,0.5,,1.0]",
        "--convergence",
        "[100x70x50x20,1e-06,10]",
        "--shrink-factors",
        "8x4x2x1",
        "--smoothing-sigmas",
        "3x2x1x0vox",
        "--verbose",
        "0",
    ]
    assert argv == expected


def test_point_set_metric_is_present_and_flagged():
    reg = _two_stage_reg()
    syn_stage = reg.stages[1]
    point_metrics = [m for m in syn_stage.metrics if is_point_set_metric(m)]
    assert len(point_metrics) == 1
    assert point_metrics[0].to_arg().startswith("PSE[")


def test_dry_run_returns_argv_without_executing():
    result = _two_stage_reg().run(dry_run=True)
    assert result.returncode == -1
    assert result.argv[0] == "antsRegistration"
    assert result.outputs["warped"] == "out_Warped.nii.gz"
    assert result.outputs["composite"] == "out_Composite.h5"


def test_extra_args_escape_hatch_appends_verbatim():
    reg = _two_stage_reg()
    reg.extra_args("--use-estimate-learning-rate-once", 1)
    argv = reg.build_command()
    assert argv[-2:] == ["--use-estimate-learning-rate-once", "1"]


def test_missing_output_raises():
    reg = AntsRegistration(3)
    reg.add_stage(
        transform=Rigid(),
        metrics=MI("f.nii", "m.nii"),
        convergence=Convergence([10]),
        shrink_factors=[1],
        smoothing_sigmas=[0],
    )
    with pytest.raises(ValueError, match="output prefix"):
        reg.build_command()


def test_no_stages_raises():
    reg = AntsRegistration(3, output="out_")
    with pytest.raises(ValueError, match="stage"):
        reg.build_command()


def test_stage_level_mismatch_raises():
    with pytest.raises(ValueError, match="levels"):
        Stage(
            transform=Rigid(),
            metrics=[MI("f.nii", "m.nii")],
            convergence=Convergence([1000, 500]),  # 2 levels
            shrink_factors=[4, 2, 1],  # 3 levels -> mismatch
            smoothing_sigmas=[2, 1, 0],
        )


def test_expected_transforms_rigid_affine_syn():
    reg = AntsRegistration(
        3,
        output="reg_",
        collapse_output_transforms=True,
        warped_output="reg_Warped.nii.gz",
    )
    reg.initialize_from_images("f.nii", "m.nii", init="center-of-mass")
    reg.add_stage(Rigid(), MI("f.nii", "m.nii"), Convergence([10]), [1], [0])
    reg.add_stage(Affine(), MI("f.nii", "m.nii"), Convergence([10]), [1], [0])
    reg.add_stage(SyN(), CC("f.nii", "m.nii"), Convergence([10]), [1], [0])

    info = reg.expected_transforms()
    # Rigid + Affine collapse into 0GenericAffine; SyN is index 1.
    assert info["files"] == [
        "reg_0GenericAffine.mat",
        "reg_1Warp.nii.gz",
        "reg_1InverseWarp.nii.gz",
    ]
    # moving -> fixed: warp first, then affine.
    assert info["forward"] == ["reg_1Warp.nii.gz", "reg_0GenericAffine.mat"]
    # fixed -> moving: inverted affine, then inverse warp.
    assert info["inverse"] == [("reg_0GenericAffine.mat", True), "reg_1InverseWarp.nii.gz"]
    assert info["warped"] == "reg_Warped.nii.gz"


def test_expected_transforms_output_dir():
    import os

    # Bare prefix -> resolves against cwd (or the cwd passed here).
    reg = AntsRegistration(3, output="reg_")
    reg.add_stage(Rigid(), MI("f.nii", "m.nii"), Convergence([10]), [1], [0])
    info = reg.expected_transforms(cwd="/work/out")
    assert info["output_dir"] == os.path.abspath("/work/out")
    assert info["files_abs"][0] == os.path.abspath("/work/out/reg_0GenericAffine.mat")

    # Prefix with a directory component -> that directory wins.
    reg2 = AntsRegistration(3, output="/data/sub01/reg_")
    reg2.add_stage(Rigid(), MI("f.nii", "m.nii"), Convergence([10]), [1], [0])
    info2 = reg2.expected_transforms(cwd="/somewhere/else")
    assert info2["output_dir"] == os.path.abspath("/data/sub01")


def test_expected_transforms_composite():
    reg = AntsRegistration(3, output="reg_", write_composite_transform=True)
    reg.add_stage(Rigid(), MI("f.nii", "m.nii"), Convergence([10]), [1], [0])
    reg.add_stage(SyN(), CC("f.nii", "m.nii"), Convergence([10]), [1], [0])

    info = reg.expected_transforms()
    assert info["files"] == ["reg_Composite.h5", "reg_InverseComposite.h5"]
    assert info["forward"] == ["reg_Composite.h5"]
    assert info["inverse"] == ["reg_InverseComposite.h5"]


def test_per_stage_moving_mask_only_on_last_stage():
    reg = AntsRegistration(3, output="reg_")
    reg.add_stage(Rigid(), MI("f", "m"), Convergence([100]), [2], [1])
    reg.add_stage(Affine(), MI("f", "m"), Convergence([100]), [2], [1])
    reg.add_stage(SyN(), MI("f", "m"), Convergence([50]), [1], [0], moving_mask="brain_moving.nii.gz")
    argv = reg.build_command()
    masks = [argv[i + 1] for i, t in enumerate(argv) if t == "--masks"]
    # One --masks per stage, in order; NA placeholders for the unmasked stages.
    assert masks == ["[NA,NA]", "[NA,NA]", "[NA,brain_moving.nii.gz]"]


def test_per_stage_mask_falls_back_to_global():
    reg = AntsRegistration(3, output="reg_")
    reg.set_masks("global_fixed.nii", "global_moving.nii")
    reg.add_stage(Rigid(), MI("f", "m"), Convergence([100]), [2], [1])
    reg.add_stage(
        SyN(), MI("f", "m"), Convergence([50]), [1], [0], moving_mask="syn_moving.nii"
    )  # overrides moving for this stage
    argv = reg.build_command()
    masks = [argv[i + 1] for i, t in enumerate(argv) if t == "--masks"]
    # Rigid inherits the global pair; SyN uses its own moving mask + global fixed.
    assert masks == ["[global_fixed.nii,global_moving.nii]", "[global_fixed.nii,syn_moving.nii]"]


def test_global_masks_emitted_once_when_no_per_stage():
    reg = AntsRegistration(3, output="reg_")
    reg.set_masks(moving_mask="m.nii")  # moving-only global mask
    reg.add_stage(Rigid(), MI("f", "m"), Convergence([100]), [2], [1])
    reg.add_stage(SyN(), MI("f", "m"), Convergence([50]), [1], [0])
    argv = reg.build_command()
    masks = [argv[i + 1] for i, t in enumerate(argv) if t == "--masks"]
    assert masks == ["[NA,m.nii]"]  # single global, NA for the unset fixed side


def test_restrict_deformation_and_masks():
    reg = AntsRegistration(3, output="out_", restrict_deformation=[1, 1, 0])
    reg.set_masks("fmask.nii.gz", "mmask.nii.gz")
    reg.add_stage(
        transform=SyN(),
        metrics=CC("f.nii", "m.nii"),
        convergence=Convergence([50]),
        shrink_factors=[1],
        smoothing_sigmas=[0],
    )
    argv = reg.build_command()
    assert "--restrict-deformation" in argv
    assert argv[argv.index("--restrict-deformation") + 1] == "1x1x0"
    assert "--masks" in argv
    assert argv[argv.index("--masks") + 1] == "[fmask.nii.gz,mmask.nii.gz]"
