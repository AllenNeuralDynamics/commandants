"""Tests for the cross-backend benchmark harness (no binaries required)."""

from __future__ import annotations

import os

import pytest

from commandants import benchmark
from commandants.core.runner import CompletedAnts


@pytest.fixture(autouse=True)
def _restore_registry():
    """Keep register_backend() in one test from leaking into the next."""
    saved = dict(benchmark.BACKENDS)
    yield
    benchmark.BACKENDS.clear()
    benchmark.BACKENDS.update(saved)


def test_registry_has_both_builtin_backends():
    assert set(benchmark.BACKENDS) >= {"ants", "elastix"}
    assert benchmark.BACKENDS["ants"].binary == "antsRegistration"
    assert benchmark.BACKENDS["elastix"].binary == "elastix"


def test_ants_builder_normalizes_prefix_and_masks(tmp_path):
    out = str(tmp_path / "ants")
    cmd = benchmark.BACKENDS["ants"].build(
        "f.nii", "m.nii", out, preset="affine", moving_mask="msk.nii", warped_ext=".nii"
    )
    argv = cmd.build_command()
    # ANTs takes a transform *prefix* plus an explicit warped output, as --output [prefix,warped].
    output_token = argv[argv.index("--output") + 1]
    assert os.path.join(out, "ants_") in output_token
    assert "ants_warped.nii" in output_token
    # The fixed-side mask kwarg is `mask=` (ANTs), moving-side agrees with elastix:
    # an unset fixed mask becomes the NA placeholder.
    assert argv[argv.index("--masks") + 1] == "[NA,msk.nii]"


def test_elastix_builder_uses_out_dir_and_threads(tmp_path):
    out = str(tmp_path / "elx")
    cmd = benchmark.BACKENDS["elastix"].build("f.nii", "m.nii", out, preset="affine", threads=3)
    argv = cmd.build_command()
    # elastix takes an output *directory* and names its own outputs.
    assert argv[argv.index("-out") + 1] == out
    assert argv[argv.index("-threads") + 1] == "3"


def test_normalizers_produce_a_uniform_shape(tmp_path):
    """Both backends must yield {warped, transforms[], log} despite different conventions."""
    ants_cmd = benchmark.BACKENDS["ants"].build("f.nii", "m.nii", str(tmp_path / "a"), preset="affine")
    ants_norm = benchmark.BACKENDS["ants"].normalize(
        ants_cmd, CompletedAnts(argv=[], returncode=0, outputs=ants_cmd.declared_outputs())
    )
    elx_cmd = benchmark.BACKENDS["elastix"].build("f.nii", "m.nii", str(tmp_path / "e"), preset="affine")
    elx_norm = benchmark.BACKENDS["elastix"].normalize(
        elx_cmd, CompletedAnts(argv=[], returncode=0, outputs=elx_cmd.declared_outputs())
    )

    for norm in (ants_norm, elx_norm):
        assert set(norm) == {"warped", "transforms", "log"}
        assert norm["warped"]
        assert norm["transforms"]

    # ANTs transforms come from expected_transforms(); elastix from declared_outputs().
    assert ants_norm["transforms"][0].endswith("GenericAffine.mat")
    assert elx_norm["transforms"][0].endswith("TransformParameters.0.txt")


def test_unknown_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown backend"):
        benchmark.compare("f.nii", "m.nii", str(tmp_path), backends=("nope",))


def test_unavailable_backend_is_skipped_not_fatal(tmp_path):
    benchmark.register_backend(
        benchmark.Backend("bogus", "definitely_not_a_binary", lambda *a, **k: None, lambda c, r: {})
    )
    report = benchmark.compare("f.nii", "m.nii", str(tmp_path), backends=("bogus",))
    assert len(report.trials) == 1
    assert report.trials[0].ok is False
    assert "not found" in report.trials[0].error
    assert report.summary() == {}


def test_build_failure_is_captured_per_backend(tmp_path):
    """A backend that explodes records an error row instead of aborting the comparison."""

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    benchmark.register_backend(benchmark.Backend("boom", "python", boom, lambda c, r: {}))
    report = benchmark.compare("f.nii", "m.nii", str(tmp_path), backends=("boom",))
    assert report.trials[0].ok is False
    assert "kaboom" in report.trials[0].error


def test_report_summary_and_csv(tmp_path):
    report = benchmark.BenchmarkReport(
        trials=[
            benchmark.Trial(backend="ants", trial=0, warmup=True, ok=True, duration_seconds=9.0),
            benchmark.Trial(backend="ants", trial=1, ok=True, duration_seconds=1.0, transforms=["a.mat"]),
            benchmark.Trial(backend="ants", trial=2, ok=True, duration_seconds=3.0),
            benchmark.Trial(backend="elastix", trial=0, ok=False, returncode=1, error="nope"),
        ]
    )
    stats = report.summary()
    # The warmup run is excluded, and the failed backend contributes nothing.
    assert stats["ants"] == {"runs": 2, "best": 1.0, "median": 2.0, "mean": 2.0}
    assert "elastix" not in stats

    table = report.to_table()
    assert "ants" in table and "failures:" in table

    path = report.to_csv(tmp_path / "out.csv")
    assert os.path.exists(path)
    text = open(path, encoding="utf-8").read()
    assert "backend,trial,warmup" in text
    assert "ants" in text and "elastix" in text


def test_thread_env_is_a_full_copy():
    """run(env=) replaces os.environ wholesale, so the pin must carry the rest of the env."""
    env = benchmark._thread_env(7)
    assert env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "7"
    assert len(env) > 1  # not just the one var
    assert benchmark._thread_env(None) is None
