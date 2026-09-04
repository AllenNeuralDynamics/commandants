"""Tests for the ANTsPyX-style preset registration builders."""

from __future__ import annotations

import commandants as cants
from commandants import presets
from commandants.registration import AntsRegistration


def _stages(reg):
    return [type(s.transform).__name__ for s in reg.stages]


def _has_init(reg):
    return bool(reg._initial_transforms)


def test_rigid_preset():
    reg = presets.rigid("f.nii", "m.nii", "out_")
    assert isinstance(reg, AntsRegistration)
    assert _stages(reg) == ["Rigid"]
    assert _has_init(reg)  # center-of-mass by default


def test_affine_preset():
    reg = presets.affine("f.nii", "m.nii", "out_")
    assert _stages(reg) == ["Rigid", "Affine"]
    assert _has_init(reg)


def test_syn_preset_is_rigid_affine_syn():
    reg = presets.syn("f.nii", "m.nii", "out_")
    assert _stages(reg) == ["Rigid", "Affine", "SyN"]
    assert _has_init(reg)


def test_syn_only_has_no_init_by_default():
    reg = presets.syn_only("f.nii", "m.nii", "out_")
    assert _stages(reg) == ["SyN"]
    assert not _has_init(reg)


def test_syn_only_can_opt_into_init():
    reg = presets.syn_only("f.nii", "m.nii", "out_", init="center-of-mass")
    assert _has_init(reg)


def test_syn_only_initial_transform_prepended():
    reg = presets.syn_only("f.nii", "m.nii", "out_", initial_transform="aff.mat")
    argv = reg.build_command()
    assert "--initial-moving-transform" in argv
    assert "aff.mat" in argv


def test_defaults_float_and_mattes():
    argv = presets.rigid("f.nii", "m.nii", "out_").build_command()
    assert "--float" in argv and argv[argv.index("--float") + 1] == "1"
    assert any(tok.startswith("Mattes[") for tok in argv)


def test_metric_override_to_cc():
    argv = presets.syn("f.nii", "m.nii", "out_", syn_metric="cc", syn_radius=3).build_command()
    assert any(tok.startswith("CC[") and tok.endswith(",3]") for tok in argv)


def test_top_level_exports_and_alias():
    assert cants.rigid is presets.rigid
    assert cants.synonly is presets.syn_only


def test_preset_is_a_normal_reg_object():
    # Presets integrate with the rest of the API (estimate, expected_transforms).
    reg = presets.affine("f.nii", "m.nii", "reg_")
    assert reg.expected_transforms()["files"]  # non-empty
    est = reg.estimate_resources(shape=(64, 64, 64))
    assert est.peak_memory_bytes > 0


def test_linear_schedule_matches_antspyx_defaults():
    # The linear presets should reproduce ANTsPyX's default Rigid/Affine command:
    # Rigid[0.25], mattes[...,32,regular,0.2], 2100x1200x1200x10, shrink 6x4x2x1.
    argv = presets.rigid("f.nii", "m.nii", "out_").build_command()
    assert "Rigid[0.25]" in argv
    assert "6x4x2x1" in argv
    assert argv[argv.index("--convergence") + 1] == "[2100x1200x1200x10,1e-06,10]"
    assert any(t.endswith("Regular,0.2]") for t in argv)


def test_initial_transform_overrides_com_init():
    reg = presets.rigid("f.nii", "m.nii", "out_", initial_transform="prior.mat")
    argv = reg.build_command()
    # The explicit transform is used; no center-of-mass [f,m,1] triple.
    assert "prior.mat" in argv
    assert not any(tok.endswith(",1]") and "f.nii" in tok for tok in argv)


def test_dim_2d():
    reg = presets.rigid("f.nii", "m.nii", "out_", dim=2)
    assert reg.dimensionality == 2


def test_ccf_stpt_ants_analogue():
    # STPT/CCF analogue: Rigid + Affine (Mattes MI, 64 bins) -> BSplineSyN (CC).
    reg = presets.ccf_stpt("f.nii", "m.nii", "out_")
    assert _stages(reg) == ["Rigid", "Affine", "BSplineSyN"]
    assert _has_init(reg)  # center-of-gravity init
    argv = reg.build_command()
    # Linear stages: Mattes MI with 64 histogram bins, coarse start at shrink 8.
    assert any(tok.startswith("Mattes[") and ",64," in tok for tok in argv)
    assert "8x4x2x1" in argv
    # Deformable: symmetric B-spline (BSplineSyN) driven by cross-correlation.
    assert any(tok.startswith("BSplineSyN[") for tok in argv)
    assert any(tok.startswith("CC[") for tok in argv)


def test_ccf_global_is_linear_only():
    reg = presets.ccf_global("f.nii", "m.nii", "out_")
    assert _stages(reg) == ["Rigid", "Affine"]
    argv = reg.build_command()
    assert any(tok.startswith("Mattes[") and ",64," in tok for tok in argv)
    assert not any(tok.startswith("BSplineSyN[") for tok in argv)
