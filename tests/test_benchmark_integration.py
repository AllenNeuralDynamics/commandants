"""Benchmark harness integration tests -- skipped unless real binaries are present.

These actually register a tiny synthetic pair with every available backend and check
that the normalized outputs exist and that timings were recorded.
"""

from __future__ import annotations

import os

import pytest

from commandants import benchmark
from commandants.core.executable import is_available

pytestmark = pytest.mark.skipif(
    not (is_available("antsRegistration") and is_available("elastix")),
    reason="needs both ANTs and elastix to compare backends; skipping.",
)

sitk = pytest.importorskip("SimpleITK")
np = pytest.importorskip("numpy")


def _blob(shift=0):
    a = np.zeros((32, 32, 32), np.float32)
    a[8 + shift : 24 + shift, 8 + shift : 24 + shift, 8:24] = 100.0
    return sitk.GetImageFromArray(a)


def test_compare_runs_every_backend_and_normalizes_outputs(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz")
    sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz")
    sitk.WriteImage(_blob(4), moving)

    report = benchmark.compare(fixed, moving, str(tmp_path / "out"), backends=("ants", "elastix"), threads=2, repeats=1)

    assert len(report.timed()) == 2, report.to_table()
    for trial in report.timed():
        assert trial.returncode == 0
        assert trial.duration_seconds is not None and trial.duration_seconds >= 0
        # The whole point of the harness: same shape regardless of backend.
        assert trial.warped and os.path.exists(trial.warped)
        assert trial.transforms and all(os.path.exists(t) for t in trial.transforms)

    stats = report.summary()
    assert set(stats) == {"ants", "elastix"}
    assert all(s["runs"] == 1 for s in stats.values())


def test_warmup_runs_are_excluded_from_the_summary(tmp_path):
    fixed = str(tmp_path / "fixed.nii.gz")
    sitk.WriteImage(_blob(0), fixed)
    moving = str(tmp_path / "moving.nii.gz")
    sitk.WriteImage(_blob(3), moving)

    report = benchmark.compare(fixed, moving, str(tmp_path / "out"), backends=("ants",), repeats=1, warmup=1)
    assert len(report.trials) == 2  # one warmup + one timed
    assert sum(t.warmup for t in report.trials) == 1
    assert report.summary()["ants"]["runs"] == 1
