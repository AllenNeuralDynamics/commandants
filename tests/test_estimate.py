"""Tests for the (heuristic) resource estimator -- no ANTs or image files needed."""

from __future__ import annotations

import pytest

from commandants import (
    CC,
    MI,
    Affine,
    AntsRegistration,
    Convergence,
    Rigid,
    SyN,
)


def _rigid_affine_syn(**kwargs) -> AntsRegistration:
    reg = AntsRegistration(3, output="reg_", **kwargs)
    reg.add_stage(Rigid(), MI("f.nii", "m.nii"), Convergence([1000, 500, 250, 100]), [8, 4, 2, 1], [3, 2, 1, 0])
    reg.add_stage(Affine(), MI("f.nii", "m.nii"), Convergence([1000, 500, 250, 100]), [8, 4, 2, 1], [3, 2, 1, 0])
    reg.add_stage(SyN(), CC("f.nii", "m.nii", radius=4), Convergence([100, 70, 50, 20]), [8, 4, 2, 1], [3, 2, 1, 0])
    return reg


def test_estimate_with_explicit_shape():
    est = _rigid_affine_syn().estimate_resources(shape=(256, 256, 256))
    assert est.n_voxels == 256**3
    assert est.real_bytes == 4  # commandants defaults to single precision
    assert est.peak_memory_bytes > 0
    assert len(est.per_stage) == 3
    # The SyN (deformable) stage should dominate peak memory.
    peak = max(est.per_stage, key=lambda s: s.memory_bytes)
    assert peak.deformable is True
    assert est.est_runtime_seconds and est.est_runtime_seconds > 0
    assert est.peak_memory_human.endswith(("MiB", "GiB"))
    assert "HEURISTIC" in est.summary()


def test_use_float_halves_real_bytes_and_lowers_memory():
    e_double = _rigid_affine_syn(use_float=False).estimate_resources(shape=(128, 128, 128))
    e_float = _rigid_affine_syn(use_float=True).estimate_resources(shape=(128, 128, 128))
    assert e_double.real_bytes == 8 and e_float.real_bytes == 4
    assert e_float.peak_memory_bytes < e_double.peak_memory_bytes


def test_deformable_costs_more_than_linear_only():
    lin = AntsRegistration(3, output="r_")
    lin.add_stage(Rigid(), MI("f", "m"), Convergence([100]), [1], [0])
    syn = AntsRegistration(3, output="r_")
    syn.add_stage(SyN(), CC("f", "m"), Convergence([100]), [1], [0])
    a = lin.estimate_resources(shape=(200, 200, 200))
    b = syn.estimate_resources(shape=(200, 200, 200))
    assert b.peak_memory_bytes > a.peak_memory_bytes
    assert b.work > a.work


def test_threads_reduce_estimated_runtime():
    reg = _rigid_affine_syn()
    one = reg.estimate_resources(shape=(128, 128, 128), threads=1)
    eight = reg.estimate_resources(shape=(128, 128, 128), threads=8)
    assert eight.est_runtime_seconds < one.est_runtime_seconds


def test_shape_inferred_from_sitk_metric():
    sitk = pytest.importorskip("SimpleITK")
    img = sitk.Image(32, 32, 16, sitk.sitkFloat32)
    reg = AntsRegistration(3, output="r_")
    reg.initialize_from_images(img, img)
    reg.add_stage(Rigid(), MI(img, img), Convergence([50]), [1], [0])
    est = reg.estimate_resources()  # shape inferred from the in-memory fixed image
    assert est.image_shape == (32, 32, 16)
    assert est.n_voxels == 32 * 32 * 16


def test_missing_file_raises_helpful_error():
    reg = AntsRegistration(3, output="r_")
    reg.add_stage(Rigid(), MI("does_not_exist.nii.gz", "m.nii"), Convergence([10]), [1], [0])
    with pytest.raises((FileNotFoundError, ValueError)):
        reg.estimate_resources()
