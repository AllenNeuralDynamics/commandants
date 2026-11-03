"""Allen STPT/CCF integration tests -- skipped unless the `gridpoint` driver is built.

These actually invoke the driver on tiny synthetic volumes. Build it from the separate
Allen-licensed driver repo and put it on PATH or $STPTPATH to enable them.
"""

from __future__ import annotations

import os

import pytest

from commandants import benchmark, stpt
from commandants.core.executable import is_available

pytestmark = pytest.mark.skipif(
    not is_available("gridpoint"),
    reason="gridpoint not found (PATH/STPTPATH); skipping integration tests.",
)

sitk = pytest.importorskip("SimpleITK")
np = pytest.importorskip("numpy")


def _supports_deformable() -> bool:
    """Probe whether the installed driver implements the discrete MRF stage.

    The driver validates --stages before touching any file, so placeholder paths are
    enough: a build without the deformable stage refuses it up front.
    """
    import subprocess

    from commandants.core.executable import resolve_binary

    try:
        proc = subprocess.run(
            [resolve_binary("gridpoint"), "-f", "x", "-m", "y", "-o", "z", "--stages", "deformable"],
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - the gate above already proved it resolves
        return False
    return "not implemented" not in (proc.stderr or "")


needs_deformable = pytest.mark.skipif(
    not _supports_deformable(),
    reason="this gridpoint build has no deformable stage (linear milestone).",
)


def _blob(shift=0):
    a = np.zeros((48, 48, 48), np.float32)
    a[14 + shift : 34 + shift, 14 + shift : 34 + shift, 14:34] = 100.0
    return sitk.GetImageFromArray(a)


def test_linear_registration_produces_a_transform_and_warped_image(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz")
    sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz")
    sitk.WriteImage(_blob(4), moving)
    out = str(tmp_path / "out")

    result = stpt.presets.affine(fixed, moving, out).run()

    assert result.returncode == 0
    assert result.duration_seconds is not None and result.duration_seconds >= 0
    assert os.path.exists(result.outputs["affine"])
    assert os.path.exists(result.outputs["warped"])
    assert result.load("warped").GetSize() == (48, 48, 48)


@needs_deformable
def test_full_pipeline_with_deformation_field(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz")
    sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz")
    sitk.WriteImage(_blob(4), moving)
    out = str(tmp_path / "out")

    result = stpt.presets.ccf_stpt(fixed, moving, out, write_deformation_field=True).run()

    assert result.returncode == 0
    assert os.path.exists(result.outputs["warp"])


def test_accepts_in_memory_images(tmp_path):
    """The headline interface requirement: pass SimpleITK images, get one back."""
    result = stpt.presets.affine(_blob(0), _blob(3), str(tmp_path / "out")).run()

    assert result.returncode == 0
    assert result.temp_dir is not None
    assert len(result.workspace.files) == 2  # fixed + moving materialized once each
    assert result.load("warped") is not None


@pytest.mark.skipif(
    not (is_available("antsRegistration") and is_available("elastix")),
    reason="three-way comparison needs ANTs and elastix too.",
)
def test_three_way_benchmark(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz")
    sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz")
    sitk.WriteImage(_blob(4), moving)

    report = benchmark.compare(fixed, moving, str(tmp_path / "bench"), backends=("ants", "elastix", "stpt"), threads=2)

    assert len(report.timed()) == 3, report.to_table()
    for trial in report.timed():
        assert trial.warped and os.path.exists(trial.warped)
        assert trial.transforms and all(os.path.exists(t) for t in trial.transforms)
