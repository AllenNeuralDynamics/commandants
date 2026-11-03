"""Input paths are validated before a tool is launched, not after it fails.

Handing a bad path to ANTs or gridpoint produces an ITK exception that names a file but
not the parameter it came from. These tests pin the earlier, clearer failure -- and the
two cases that must stay exempt, where a resolved argument is legitimately not a path.
"""

from __future__ import annotations

import pytest

from commandants import MissingInputError, presets
from commandants.preprocessing import ImageMath
from commandants.stpt import presets as stpt_presets


def test_validate_names_the_parameter_and_the_value(tmp_path):
    reg = presets.affine("missing_fixed.nii.gz", "missing_moving.nii.gz", str(tmp_path / "out_"))
    with pytest.raises(MissingInputError) as excinfo:
        reg.build_command(validate=True)
    message = str(excinfo.value)
    assert "fixed" in message
    assert "missing_fixed.nii.gz" in message


def test_validate_covers_the_stpt_backend(tmp_path):
    reg = stpt_presets.affine("missing_fixed.nii.gz", "missing_moving.nii.gz", str(tmp_path))
    with pytest.raises(MissingInputError):
        reg.build_command(validate=True)


def test_assembly_without_validate_still_allows_missing_paths(tmp_path):
    """Previews and `to_shell()` must keep working on paths that do not exist yet."""
    reg = presets.affine("missing_fixed.nii.gz", "missing_moving.nii.gz", str(tmp_path / "out_"))
    assert "missing_fixed.nii.gz" in " ".join(reg.build_command())
    assert "missing_fixed.nii.gz" in reg.to_shell()


def test_existing_paths_validate_cleanly(tmp_path):
    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    fixed.write_bytes(b"")
    moving.write_bytes(b"")
    reg = presets.affine(str(fixed), str(moving), str(tmp_path / "out_"))
    assert str(fixed) in " ".join(reg.build_command(validate=True))


def test_scalar_operands_are_not_treated_as_paths(tmp_path):
    """ImageMath operands may be numbers -- validating them as files would break it."""
    image = tmp_path / "a.nii.gz"
    image.write_bytes(b"")
    cmd = ImageMath(3, str(tmp_path / "out.nii.gz"), "m", str(image), 2)
    assert "2" in cmd.build_command(validate=True)


def test_absent_masks_do_not_validate_the_na_placeholder(tmp_path):
    """ANTs spells "no mask" as the literal NA, which is not a file."""
    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    fixed.write_bytes(b"")
    moving.write_bytes(b"")
    reg = presets.syn(str(fixed), str(moving), str(tmp_path / "out_"), mask=None, moving_mask=None)
    reg.build_command(validate=True)  # must not raise
