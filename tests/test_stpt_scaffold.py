"""STPT (Allen CCF) backend scaffolding: routing, install-less resolution, command base.

Also proves the ANTs/elastix provisioning paths are unaffected by making the two-way
spec branches table-driven.
"""

from __future__ import annotations

import os

import pytest

from commandants import install
from commandants.core import executable
from commandants.core.exceptions import AntsNotFoundError, CommandantsError
from commandants.stpt import StptToolCommand


@pytest.fixture(autouse=True)
def _clear_cache():
    executable.clear_cache()
    yield
    executable.clear_cache()


@pytest.mark.parametrize(
    "binary",
    [
        "idpProjectionAlignmentModule",
        "idpProjectionLocalAlignmentModule",
        "idpProjectionResampleVolumeRGBModule",
        "gridpoint",
        "gridpoint.exe",
        "stpt_register",
    ],
)
def test_stpt_binaries_route_to_the_stpt_spec(binary):
    assert install.spec_for_binary(binary) is install.STPT_SPEC


def test_existing_tools_still_route_correctly():
    assert install.spec_for_binary("antsRegistration") is install.ANTS_SPEC
    assert install.spec_for_binary("elastix") is install.ELASTIX_SPEC
    assert install.spec_for_binary("transformix") is install.ELASTIX_SPEC
    # Anything unknown keeps defaulting to ANTs.
    assert install.spec_for_binary("N4BiasFieldCorrection") is install.ANTS_SPEC


def test_stpt_spec_is_install_less():
    spec = install.STPT_SPEC
    assert spec.downloadable is False
    assert spec.install_command is None
    assert spec.env_var == "STPTPATH"
    assert install.managed_bin_dir_for(spec) is None
    with pytest.raises(CommandantsError, match="no downloadable release"):
        install.install_for(spec)


def test_managed_bin_dir_for_dispatches_to_the_per_tool_helpers(monkeypatch):
    monkeypatch.setattr(install, "managed_bin_dir", lambda version=None: "/m/ants")
    monkeypatch.setattr(install, "managed_elastix_bin_dir", lambda version=None: "/m/elx")
    assert install.managed_bin_dir_for(install.ANTS_SPEC) == "/m/ants"
    assert install.managed_bin_dir_for(install.ELASTIX_SPEC) == "/m/elx"


def test_resolve_stpt_uses_env_var(monkeypatch):
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)
    monkeypatch.setenv("STPTPATH", "/stpt/bin")
    monkeypatch.setattr(executable, "_candidate_in_dir", lambda d, n: f"{d}/{n}" if str(d) == "/stpt/bin" else None)
    assert executable.resolve_binary("gridpoint") == "/stpt/bin/gridpoint"


def test_install_less_not_found_message(monkeypatch):
    monkeypatch.delenv("STPTPATH", raising=False)
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)
    with pytest.raises(AntsNotFoundError) as exc:
        executable.resolve_binary("idpProjectionAlignmentModule")
    message = str(exc.value)
    assert "STPTPATH" in message
    # No install command exists for user-built binaries, so none is advertised.
    assert "install-" not in message
    assert "COMMANDANTS_AUTO_INSTALL" not in message


def test_auto_install_is_never_attempted_for_install_less_tools(monkeypatch):
    """COMMANDANTS_AUTO_INSTALL must not try to download a tool that has no release."""
    monkeypatch.delenv("STPTPATH", raising=False)
    monkeypatch.setenv("COMMANDANTS_AUTO_INSTALL", "1")
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)

    def explode(*args, **kwargs):
        raise AssertionError("install_for must not be called for an install-less tool")

    monkeypatch.setattr(install, "install_for", explode)
    with pytest.raises(AntsNotFoundError):
        executable.resolve_binary("gridpoint")


class _Dummy(StptToolCommand):
    binary_name = "gridpoint"

    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = out_dir

    def _build_args(self):
        return ["-o", str(self.out_dir)]


def test_command_base_creates_the_output_directory(tmp_path):
    out = tmp_path / "made" / "by" / "prepare"
    cmd = _Dummy(str(out))
    assert cmd.build_command()[1:] == ["-o", str(out)]
    assert not out.exists()
    cmd._prepare_execution(None)
    assert os.path.isdir(out)
