"""elastix integration tests -- skipped unless the elastix binaries are available.

These actually invoke elastix/transformix on tiny synthetic images.
"""

from __future__ import annotations

import os

import pytest

from commandants import elastix
from commandants.core.executable import is_available

pytestmark = pytest.mark.skipif(
    not is_available("elastix"),
    reason="elastix not found (PATH/ELASTIXPATH/managed); skipping integration tests.",
)

sitk = pytest.importorskip("SimpleITK")
np = pytest.importorskip("numpy")


def _blob(shift=0):
    a = np.zeros((48, 48, 48), np.float32)
    a[14 + shift:34 + shift, 14 + shift:34 + shift, 14:34] = 100.0
    return sitk.GetImageFromArray(a)


def test_elastix_rigid_registration(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz"); sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz"); sitk.WriteImage(_blob(4), moving)
    out = str(tmp_path / "out")

    reg = elastix.presets.rigid(fixed, moving, out)
    result = reg.run()
    assert result.returncode == 0
    assert result.duration_seconds is not None and result.duration_seconds >= 0
    assert os.path.exists(os.path.join(out, "TransformParameters.0.txt"))
    assert os.path.exists(os.path.join(out, "result.0.nii"))


def test_transformix_apply(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz"); sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz"); sitk.WriteImage(_blob(4), moving)
    out = str(tmp_path / "out")
    elastix.presets.rigid(fixed, moving, out).run()

    tp = os.path.join(out, "TransformParameters.0.txt")
    tout = str(tmp_path / "tout")
    tx = elastix.Transformix(tp, tout, moving=moving)
    result = tx.run()
    assert result.returncode == 0
    assert os.path.exists(os.path.join(tout, "result.nii"))
