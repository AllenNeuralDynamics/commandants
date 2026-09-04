"""Integration smoke tests -- skipped unless ANTs binaries are on the system.

These actually invoke ANTs on tiny synthetic images. On a machine without ANTs
(or without SimpleITK/numpy) they skip cleanly.
"""

from __future__ import annotations

import os

import pytest

from commandants import (
    MI,
    AntsApplyTransforms,
    AntsRegistration,
    Convergence,
    N4BiasFieldCorrection,
    Rigid,
    TempWorkspace,
    is_available,
)

# Gate on the full resolution order (explicit > ANTSPATH > PATH > managed install),
# not just PATH -- so a `commandants install-ants` binary counts.
pytestmark = pytest.mark.skipif(
    not is_available("antsRegistration"),
    reason="ANTs binaries not found (PATH/ANTSPATH/managed); skipping integration tests.",
)

sitk = pytest.importorskip("SimpleITK")
np = pytest.importorskip("numpy")


def _blob(shift=0):
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[20 + shift : 40 + shift, 20 + shift : 40 + shift] = 100.0
    return sitk.GetImageFromArray(arr)


def _save_blob(path, shift=0):
    sitk.WriteImage(_blob(shift), str(path))
    return str(path)


def test_rigid_registration_produces_outputs(tmp_path):
    fixed = _save_blob(tmp_path / "fixed.nii.gz", shift=0)
    moving = _save_blob(tmp_path / "moving.nii.gz", shift=5)
    prefix = str(tmp_path / "out_")
    warped = str(tmp_path / "warped.nii.gz")

    reg = AntsRegistration(2, output=prefix, warped_output=warped, verbose=False)
    reg.initialize_from_images(fixed, moving, init="center-of-mass")
    reg.add_stage(
        transform=Rigid(0.1),
        metrics=MI(fixed, moving, bins=32, sampling="Regular", sampling_pct=0.5),
        convergence=Convergence([100, 50, 25], threshold=1e-6, window=10),
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    )
    result = reg.run()
    assert result.returncode == 0
    assert os.path.exists(warped)
    assert os.path.exists(prefix + "0GenericAffine.mat")


def test_in_memory_registration_with_sitk_images(tmp_path):
    """Pass SimpleITK images directly; the wrapper writes temp files and runs."""
    fixed = _blob(shift=0)
    moving = _blob(shift=5)
    prefix = str(tmp_path / "out_")

    reg = AntsRegistration(2, output=prefix, verbose=False)
    reg.set_workspace(TempWorkspace(base=str(tmp_path), keep=True))
    reg.initialize_from_images(fixed, moving, init="center-of-mass")
    reg.add_stage(
        transform=Rigid(0.1),
        metrics=MI(fixed, moving, bins=32, sampling="Regular", sampling_pct=0.5),
        convergence=Convergence([100, 50, 25]),
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    )
    result = reg.run()
    assert result.returncode == 0
    # Temp inputs were written and are discoverable.
    assert result.temp_dir is not None
    assert len(result.workspace.files) == 2
    assert os.path.exists(prefix + "0GenericAffine.mat")


def test_apply_transforms_round_trip_and_load(tmp_path):
    fixed = _save_blob(tmp_path / "fixed.nii.gz", shift=0)
    moving = _save_blob(tmp_path / "moving.nii.gz", shift=5)
    prefix = str(tmp_path / "out_")

    reg = AntsRegistration(2, output=prefix, verbose=False)
    reg.initialize_from_images(fixed, moving, init="center-of-mass")
    reg.add_stage(
        transform=Rigid(0.1),
        metrics=MI(fixed, moving, bins=32, sampling="Regular", sampling_pct=0.5),
        convergence=Convergence([100, 50, 25]),
        shrink_factors=[4, 2, 1],
        smoothing_sigmas=[2, 1, 0],
    )
    reg.run()

    out = str(tmp_path / "applied.nii.gz")
    apply = AntsApplyTransforms(2, moving, fixed, out, interpolation="Linear")
    apply.add_transform(prefix + "0GenericAffine.mat")
    result = apply.run()
    assert result.returncode == 0
    # load() returns a SimpleITK image.
    loaded = result.load()
    assert isinstance(loaded, sitk.Image)


def test_n4_produces_output(tmp_path):
    raw = _save_blob(tmp_path / "raw.nii.gz", shift=0)
    out = str(tmp_path / "n4.nii.gz")
    result = N4BiasFieldCorrection(
        2, raw, out, shrink_factor=2, convergence_iterations=[20, 20], convergence_threshold=0.0
    ).run()
    assert result.returncode == 0
    assert os.path.exists(out)
