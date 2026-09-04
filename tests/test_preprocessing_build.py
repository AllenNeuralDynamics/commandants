"""Command-construction tests for the preprocessing wrappers and point I/O."""

from __future__ import annotations

import csv

import pytest

from commandants import ImageMath, N4BiasFieldCorrection, ResampleImage, ThresholdImage
from commandants.io import read_points, write_points


def test_n4_full_command():
    n4 = N4BiasFieldCorrection(
        3,
        "raw.nii.gz",
        "n4.nii.gz",
        bias_output="bias.nii.gz",
        shrink_factor=4,
        convergence_iterations=[50, 50, 50, 50],
        convergence_threshold=0.0,
        bspline_distance=200,
        verbose=False,
    )
    argv = n4.build_command()
    assert argv == [
        "N4BiasFieldCorrection",
        "--image-dimensionality",
        "3",
        "--input-image",
        "raw.nii.gz",
        "--shrink-factor",
        "4",
        "--convergence",
        "[50x50x50x50,0.0]",
        "--bspline-fitting",
        "[200]",
        "--output",
        "[n4.nii.gz,bias.nii.gz]",
        "--verbose",
        "0",
    ]
    assert n4.declared_outputs() == {"output": "n4.nii.gz", "bias_field": "bias.nii.gz"}


def test_threshold_range():
    t = ThresholdImage(3, "in.nii", "out.nii", 100, 200)
    assert t.build_command() == ["ThresholdImage", "3", "in.nii", "out.nii", "100", "200"]


def test_threshold_with_inside_outside():
    t = ThresholdImage(3, "in.nii", "out.nii", 100, 200, inside=1, outside=0)
    assert t.build_command()[-2:] == ["1", "0"]


def test_threshold_otsu():
    t = ThresholdImage.otsu(3, "in.nii", "out.nii", 4)
    assert t.build_command() == ["ThresholdImage", "3", "in.nii", "out.nii", "Otsu", "4"]


def test_threshold_requires_bounds():
    t = ThresholdImage(3, "in.nii", "out.nii")
    with pytest.raises(ValueError, match="lower and upper"):
        t.build_command()


def test_image_math():
    im = ImageMath(3, "out.nii", "MD", "mask.nii", 2)
    assert im.build_command() == ["ImageMath", "3", "out.nii", "MD", "mask.nii", "2"]


def test_resample_spacing():
    r = ResampleImage(3, "in.nii", "out.nii", [1.0, 1.0, 1.0], interpret="spacing")
    assert r.build_command() == ["ResampleImage", "3", "in.nii", "out.nii", "1.0x1.0x1.0", "0"]


def test_resample_size_with_interpolation():
    r = ResampleImage(3, "in.nii", "out.nii", [128, 128, 64], interpret="size", interpolation=1)
    assert r.build_command() == ["ResampleImage", "3", "in.nii", "out.nii", "128x128x64", "1", "1"]


def test_points_round_trip(tmp_path):
    path = tmp_path / "pts.csv"
    coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    write_points(path, coords, labels=[1, 2])

    # Header should include x,y,z and label.
    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ["x", "y", "z", "label"]

    got, extra = read_points(path, as_array=False)
    assert got == coords
    assert extra["label"] == ["1", "2"]
