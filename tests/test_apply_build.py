"""Command-construction tests for the apply-transforms wrappers."""

from __future__ import annotations

import pytest

from commandants import AntsApplyTransforms, AntsApplyTransformsToPoints


def test_apply_transforms_with_inversion_and_order():
    apply = AntsApplyTransforms(
        3,
        "moving.nii.gz",
        "fixed.nii.gz",
        "resampled.nii.gz",
        interpolation="BSpline[3]",
        default_value=0,
    )
    apply.add_transform("out_1Warp.nii.gz")
    apply.add_transform("out_0GenericAffine.mat", invert=True)

    argv = apply.build_command()
    assert argv == [
        "antsApplyTransforms",
        "--dimensionality",
        "3",
        "--input",
        "moving.nii.gz",
        "--reference-image",
        "fixed.nii.gz",
        "--output",
        "resampled.nii.gz",
        "--interpolation",
        "BSpline[3]",
        "--default-value",
        "0",
        "--float",
        "1",  # single precision is the default
        "--transform",
        "out_1Warp.nii.gz",
        "--transform",
        "[out_0GenericAffine.mat,1]",
        "--verbose",
        "0",
    ]


def test_apply_transforms_image_type_name():
    apply = AntsApplyTransforms(4, "ts.nii.gz", "ref.nii.gz", "out.nii.gz", image_type="time-series")
    argv = apply.build_command()
    assert "--input-image-type" in argv
    assert argv[argv.index("--input-image-type") + 1] == "3"


def test_apply_transforms_to_points():
    ap = AntsApplyTransformsToPoints(2, "pts_in.csv", "pts_out.csv", precision=0)
    ap.add_transform("out_0GenericAffine.mat", invert=True)
    argv = ap.build_command()
    assert argv == [
        "antsApplyTransformsToPoints",
        "--dimensionality",
        "2",
        "--input",
        "pts_in.csv",
        "--output",
        "pts_out.csv",
        "--transform",
        "[out_0GenericAffine.mat,1]",
        "--precision",
        "0",
    ]


def test_declared_output():
    apply = AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii")
    assert apply.declared_outputs() == {"output": "o.nii"}


def test_default_is_single_precision():
    argv = AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii").build_command()
    assert argv[argv.index("--float") + 1] == "1"


def test_use_float_false_gives_double():
    argv = AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii", use_float=False).build_command()
    assert argv[argv.index("--float") + 1] == "0"


def test_use_float_none_omits_flag():
    argv = AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii", use_float=None).build_command()
    assert "--float" not in argv


def test_output_data_type_emitted():
    argv = AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii", output_data_type="float").build_command()
    assert argv[argv.index("--output-data-type") + 1] == "float"


def test_output_data_type_invalid_raises():
    with pytest.raises(ValueError, match="output_data_type"):
        AntsApplyTransforms(3, "m.nii", "f.nii", "o.nii", output_data_type="float32")
