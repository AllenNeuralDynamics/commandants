"""Command-construction tests for Elastix / Transformix (no binaries needed)."""

from __future__ import annotations

import os

from commandants.elastix import Elastix, Transformix, presets


def test_elastix_single_param_file():
    # A param-file PATH (str) is passed through verbatim (no materialization).
    reg = Elastix("fixed.nii", "moving.nii", "outdir", "affine.txt", threads=4)
    argv = reg.build_command()
    assert argv == [
        "elastix",
        "-f",
        "fixed.nii",
        "-m",
        "moving.nii",
        "-out",
        "outdir",
        "-p",
        "affine.txt",
        "-threads",
        "4",
    ]


def test_elastix_masks_and_initial_transform():
    reg = Elastix(
        "f.nii", "m.nii", "out", "p.txt", fixed_mask="fm.nii", moving_mask="mm.nii", initial_transform="t0.txt"
    )
    argv = reg.build_command()
    assert "-fMask" in argv and argv[argv.index("-fMask") + 1] == "fm.nii"
    assert "-mMask" in argv and argv[argv.index("-mMask") + 1] == "mm.nii"
    assert "-t0" in argv and argv[argv.index("-t0") + 1] == "t0.txt"


def test_elastix_multi_stage_two_p():
    reg = Elastix("f.nii", "m.nii", "out", ["a.txt", "b.txt"])
    argv = reg.build_command()
    ps = [argv[i + 1] for i, t in enumerate(argv) if t == "-p"]
    assert ps == ["a.txt", "b.txt"]


def test_elastix_inmemory_map_previews_as_placeholder():
    # A preset uses an in-memory ParameterMap -> preview shows a placeholder,
    # writes no temp files.
    reg = presets.affine("f.nii", "m.nii", "out")
    argv = reg.build_command()  # materialize=False
    assert "<param:param0>" in argv
    assert reg.workspace is None


def test_elastix_inmemory_map_materializes(tmp_path):
    reg = presets.rigid("f.nii", "m.nii", str(tmp_path / "out"))
    argv = reg.build_command(materialize=True)
    p_path = argv[argv.index("-p") + 1]
    assert os.path.exists(p_path) and p_path.endswith(".txt")
    # the written file is a real elastix parameter file
    with open(p_path) as fh:
        assert '(Transform "EulerTransform")' in fh.read()


def test_elastix_declared_outputs(tmp_path):
    out = str(tmp_path / "o")
    reg = Elastix("f.nii", "m.nii", out, ["a.txt", "b.txt"], result_format="nii")
    d = reg.declared_outputs()
    assert d["warped"] == os.path.join(out, "result.1.nii")
    assert d["transform"] == os.path.join(out, "TransformParameters.1.txt")
    assert d["transform0"] == os.path.join(out, "TransformParameters.0.txt")
    assert d["log"] == os.path.join(out, "elastix.log")


def test_transformix_warp_image():
    tx = Transformix("TransformParameters.0.txt", "tout", moving="new.nii")
    argv = tx.build_command()
    assert argv == [
        "transformix",
        "-tp",
        "TransformParameters.0.txt",
        "-out",
        "tout",
        "-in",
        "new.nii",
    ]


def test_transformix_deformation_and_jacobian():
    tx = Transformix("tp.txt", "tout", deformation_field=True, jacobian=True)
    argv = tx.build_command()
    assert argv[argv.index("-def") + 1] == "all"
    assert argv[argv.index("-jac") + 1] == "all"


def test_transformix_points():
    tx = Transformix("tp.txt", "tout", points="pts.txt")
    argv = tx.build_command()
    assert argv[argv.index("-def") + 1] == "pts.txt"
    assert tx.declared_outputs()["points"] == os.path.join("tout", "outputpoints.txt")


def test_transformix_requires_a_task():
    import pytest

    with pytest.raises(ValueError):
        Transformix("tp.txt", "tout")  # nothing to do
    with pytest.raises(ValueError):
        Transformix("tp.txt", "tout", points="p.txt", deformation_field=True)  # both -def


def test_preset_builders_return_ready_elastix():
    reg = presets.affine_bspline("f.nii", "m.nii", "out")
    assert isinstance(reg, Elastix)
    assert len(reg.param_maps) == 2
