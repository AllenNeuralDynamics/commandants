"""Tests for the opt-in SimpleITK in-memory layer.

These require SimpleITK (the [io]/[dev] extra) but NOT ANTs -- they exercise temp
materialization and command assembly, not execution.
"""

from __future__ import annotations

import os

import pytest

from commandants import (
    CC,
    AntsApplyTransforms,
    AntsRegistration,
    Convergence,
    SyN,
    TempWorkspace,
    is_sitk_image,
)

sitk = pytest.importorskip("SimpleITK")


def _img():
    img = sitk.Image(8, 8, sitk.sitkFloat32)
    img.SetSpacing((1.5, 2.0))  # non-trivial metadata to prove it round-trips
    return img


def test_is_sitk_image():
    assert is_sitk_image(_img()) is True
    assert is_sitk_image("path.nii.gz") is False
    assert is_sitk_image(42) is False


def test_tempworkspace_writes_and_exposes(tmp_path):
    ws = TempWorkspace(base=str(tmp_path), keep=False)
    img = _img()
    p1 = ws.materialize(img, name="fixed")
    assert os.path.exists(p1)
    assert ws.inputs["fixed"] == p1
    assert p1 in ws.files
    # Same object -> written once, path reused.
    p2 = ws.materialize(img, name="moving")
    assert p2 == p1
    assert len(ws.files) == 1
    # Metadata survives the disk round-trip.
    assert tuple(sitk.ReadImage(p1).GetSpacing()) == (1.5, 2.0)
    ws.cleanup()
    assert not os.path.exists(p1)


def test_preview_uses_placeholder_and_writes_nothing():
    cmd = AntsApplyTransforms(2, _img(), "ref.nii.gz", "out.nii.gz")
    argv = cmd.build_command()  # materialize=False by default
    assert "<sitk:input>" in argv
    assert cmd.workspace is None  # nothing materialized during preview


def test_apply_materializes_sitk_input(tmp_path):
    cmd = AntsApplyTransforms(2, _img(), "ref.nii.gz", "out.nii.gz")
    cmd.set_workspace(TempWorkspace(base=str(tmp_path), keep=False))
    argv = cmd.build_command(materialize=True)
    inp = argv[argv.index("--input") + 1]
    assert os.path.dirname(inp) == cmd.workspace.dir
    assert os.path.exists(inp)


def test_registration_reuses_one_temp_file_per_image(tmp_path):
    fixed, moving = _img(), _img()
    reg = AntsRegistration(2, output="out_")
    reg.set_workspace(TempWorkspace(base=str(tmp_path), keep=False))
    reg.initialize_from_images(fixed, moving)
    reg.add_stage(SyN(), CC(fixed, moving), Convergence([10]), [1], [0])

    argv = reg.build_command(materialize=True)
    ws = reg.workspace
    # fixed and moving are each written once, even though fixed appears in both
    # the initial transform and the CC metric.
    assert len(ws.files) == 2
    metric_arg = argv[argv.index("--metric") + 1]
    assert ws.dir in metric_arg


def test_dry_run_does_not_write(tmp_path):
    reg = AntsRegistration(2, output="out_")
    reg.initialize_from_images(_img(), _img())
    reg.add_stage(SyN(), CC(_img(), _img()), Convergence([10]), [1], [0])
    result = reg.run(dry_run=True)
    assert result.returncode == -1
    assert result.workspace is None  # dry-run never materializes
    # Image placeholders appear inside bracketed metric/transform tokens.
    assert any("<sitk:" in tok for tok in result.argv)
