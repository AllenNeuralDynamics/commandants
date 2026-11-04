"""Allen STPT/CCF wrapper: argv, declared outputs, presets, and SimpleITK handling.

None of this needs the `gridpoint` binary -- build_command() never resolves it.
"""

from __future__ import annotations

import os

import pytest

from commandants import benchmark, stpt
from commandants.core.runner import CompletedAnts
from commandants.stpt import StptRegistration


def test_minimal_argv():
    argv = StptRegistration("f.nii", "m.nii", "out").build_command()
    assert argv[0] == "gridpoint"
    assert argv[argv.index("-f") + 1] == "f.nii"
    assert argv[argv.index("-m") + 1] == "m.nii"
    assert argv[argv.index("-o") + 1] == "out"
    # All three stages by default, in pipeline order.
    assert argv[argv.index("--stages") + 1] == "rigid,affine,deformable"


def test_stage_subset_and_masks():
    reg = StptRegistration(
        "f.nii", "m.nii", "out", stages=("rigid", "affine"), fixed_mask="fm.nii", moving_mask="mm.nii"
    )
    argv = reg.build_command()
    assert argv[argv.index("--stages") + 1] == "rigid,affine"
    assert argv[argv.index("--fixed-mask") + 1] == "fm.nii"
    assert argv[argv.index("--moving-mask") + 1] == "mm.nii"


def test_all_tuning_flags_are_reachable():
    reg = StptRegistration(
        "f.nii",
        "m.nii",
        "out",
        threads=8,
        bins=64,
        linear_samples=250000,
        linear_iterations=100,
        levels=4,
        grid=(30, 15, 8, 4),
        regularization=1e-5,
        samples=1024,
        write_deformation_field=True,
        verbose=True,
    )
    argv = reg.build_command()
    assert argv[argv.index("--threads") + 1] == "8"
    assert argv[argv.index("--bins") + 1] == "64"
    assert argv[argv.index("--linear-samples") + 1] == "250000"
    assert argv[argv.index("--linear-iterations") + 1] == "100"
    assert argv[argv.index("--levels") + 1] == "4"
    assert argv[argv.index("--grid") + 1] == "30,15,8,4"
    assert argv[argv.index("--lambda") + 1] == "1e-05"
    assert argv[argv.index("--samples") + 1] == "1024"
    assert "--write-deformation-field" in argv
    assert "-v" in argv


def test_grid_accepts_a_preformatted_string():
    argv = StptRegistration("f.nii", "m.nii", "out", grid="30,15,8,4").build_command()
    assert argv[argv.index("--grid") + 1] == "30,15,8,4"


def test_omitted_options_emit_no_flags():
    argv = StptRegistration("f.nii", "m.nii", "out").build_command()
    for flag in ("--threads", "--bins", "--levels", "--grid", "--lambda", "--samples", "-v"):
        assert flag not in argv
    assert "--write-deformation-field" not in argv


def test_declared_outputs():
    reg = StptRegistration("f.nii", "m.nii", "out", write_deformation_field=True)
    outputs = reg.declared_outputs()
    assert outputs["warped"] == os.path.join("out", "stpt_Warped.nii.gz")
    assert outputs["affine"] == os.path.join("out", "stpt_Affine.txt")
    assert outputs["warp"] == os.path.join("out", "stpt_Warp.nii.gz")
    assert outputs["inverse_warp"] == os.path.join("out", "stpt_InverseWarp.nii.gz")
    assert outputs["log"] == os.path.join("out", "stpt.log")


def test_declared_outputs_track_the_requested_work():
    # No deformation field requested -> neither field is declared.
    outputs = StptRegistration("f.nii", "m.nii", "out").declared_outputs()
    assert "warp" not in outputs
    assert "inverse_warp" not in outputs
    # Deformable-only -> no linear transform is produced.
    deformable = StptRegistration("f.nii", "m.nii", "out", stages=("deformable",))
    assert "affine" not in deformable.declared_outputs()


def test_inverse_warp_is_not_part_of_the_forward_transform_chain():
    """The benchmark composes a moving -> fixed chain; the backward field is the other
    direction and must not be spliced into it."""
    from commandants.benchmark import BACKENDS
    from commandants.core.runner import CompletedAnts

    command = StptRegistration("f.nii", "m.nii", "out", write_deformation_field=True)
    normalized = BACKENDS["stpt"].normalize(
        command, CompletedAnts(argv=[], returncode=0, outputs=command.declared_outputs())
    )
    assert normalized["transforms"] == [
        os.path.join("out", "stpt_Warp.nii.gz"),
        os.path.join("out", "stpt_Affine.txt"),
    ]


def test_warped_ext_is_reflected_in_argv_and_outputs():
    reg = StptRegistration("f.nii", "m.nii", "out", warped_ext=".nii")
    assert reg.build_command()[-1] == ".nii"
    assert reg.declared_outputs()["warped"].endswith("stpt_Warped.nii")


def test_invalid_stages_are_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        StptRegistration("f.nii", "m.nii", "out", stages=("rigid", "syn"))
    with pytest.raises(ValueError, match="at least one stage"):
        StptRegistration("f.nii", "m.nii", "out", stages=())


def test_presets_follow_the_shared_convention():
    assert stpt.presets.rigid("f.nii", "m.nii", "out").stages == ("rigid",)
    assert stpt.presets.affine("f.nii", "m.nii", "out").stages == ("rigid", "affine")
    assert stpt.presets.ccf_global("f.nii", "m.nii", "out").stages == ("rigid", "affine")
    full = stpt.presets.ccf_stpt("f.nii", "m.nii", "out")
    assert isinstance(full, StptRegistration)
    assert full.stages == ("rigid", "affine", "deformable")


def test_extra_args_escape_hatch():
    reg = StptRegistration("f.nii", "m.nii", "out").extra_args("--future-flag", "7")
    assert reg.build_command()[-2:] == ["--future-flag", "7"]


# --- SimpleITK images in/out (the headline interface requirement) -------------


def test_sitk_images_preview_as_placeholders_then_materialize(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    np = pytest.importorskip("numpy")

    def blob(shift=0):
        a = np.zeros((8, 8, 8), np.float32)
        a[2 + shift : 6 + shift, 2:6, 2:6] = 100.0
        return sitk.GetImageFromArray(a)

    reg = StptRegistration(blob(0), blob(2), str(tmp_path / "out"), moving_mask=blob(0))

    # Preview writes nothing and shows readable placeholders.
    shell = reg.to_shell()
    assert "<sitk:fixed>" in shell and "<sitk:moving>" in shell and "<sitk:moving_mask>" in shell

    # Materializing writes real temp files and exposes them by role.
    argv = reg.build_command(materialize=True)
    workspace = reg.workspace
    assert set(workspace.inputs) == {"fixed", "moving", "moving_mask"}
    assert all(os.path.exists(path) for path in workspace.files)
    assert workspace.inputs["fixed"] in argv


# --- benchmark integration ----------------------------------------------------


def test_registered_as_a_benchmark_backend():
    import inspect

    backend = benchmark.BACKENDS["stpt"]
    assert backend.binary == "gridpoint"
    # Part of the default three-way comparison.
    default_backends = inspect.signature(benchmark.compare).parameters["backends"].default
    assert "stpt" in default_backends


def test_benchmark_adapter_builds_and_normalizes(tmp_path):
    backend = benchmark.BACKENDS["stpt"]
    command = backend.build("f.nii", "m.nii", str(tmp_path), preset="ccf_stpt", threads=2, warped_ext=".nii")
    argv = command.build_command()
    assert argv[argv.index("--threads") + 1] == "2"
    # The benchmark equalizes the warped format so compression cost matches other backends.
    assert argv[argv.index("--warped-ext") + 1] == ".nii"

    normalized = backend.normalize(command, CompletedAnts(argv=[], returncode=0, outputs=command.declared_outputs()))
    assert set(normalized) == {"warped", "transforms", "log"}
    assert normalized["warped"].endswith("stpt_Warped.nii")
    assert normalized["transforms"] == [command.declared_outputs()["affine"]]


def test_upstream_optimizer_flag_is_opt_in(tmp_path):
    """The Allen constants are reachable, and absent unless asked for."""
    plain = StptRegistration("f.nii.gz", "m.nii.gz", str(tmp_path)).build_command()
    assert "--upstream-optimizer" not in plain

    upstream = StptRegistration("f.nii.gz", "m.nii.gz", str(tmp_path), upstream_optimizer=True).build_command()
    assert "--upstream-optimizer" in upstream
