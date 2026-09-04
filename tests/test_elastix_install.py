"""Tests for elastix provisioning + resolution (no network). Also proves the
ANTs path is unaffected by the ToolSpec generalization."""

from __future__ import annotations

import pytest

from commandants import install
from commandants.core import executable
from commandants.core.exceptions import AntsNotFoundError, CommandantsError

ELASTIX_ASSETS = [
    "elastix-5.3.1-windows.zip",
    "elastix-5.3.1-ubuntu.zip",
    "elastix-5.3.1-macos.zip",
]


@pytest.fixture(autouse=True)
def _clear_cache():
    executable.clear_cache()
    yield
    executable.clear_cache()


def test_select_elastix_asset_per_os():
    assert "windows" in install.select_elastix_asset(ELASTIX_ASSETS, "Windows")
    assert "ubuntu" in install.select_elastix_asset(ELASTIX_ASSETS, "Linux")
    assert "macos" in install.select_elastix_asset(ELASTIX_ASSETS, "Darwin")


def test_select_elastix_asset_none_raises():
    with pytest.raises(CommandantsError):
        install.select_elastix_asset(["something-else.tar.gz"], "Linux")


def test_bare_vs_v_prefixed_tags():
    assert install.ELASTIX_SPEC.tag("5.3.1") == "5.3.1"
    assert install.ELASTIX_SPEC.tag("v5.3.1") == "5.3.1"
    assert install.ANTS_SPEC.tag("2.6.5") == "v2.6.5"


def test_spec_for_binary_routing():
    assert install.spec_for_binary("elastix") is install.ELASTIX_SPEC
    assert install.spec_for_binary("transformix") is install.ELASTIX_SPEC
    assert install.spec_for_binary("transformix.exe") is install.ELASTIX_SPEC
    assert install.spec_for_binary("antsRegistration") is install.ANTS_SPEC


def test_managed_elastix_discovery(tmp_path, monkeypatch):
    root = tmp_path / "commandants"
    monkeypatch.setattr(install, "user_data_dir", lambda: str(root))
    base = root / "elastix" / "5.3.1" / "elastix-5.3.1-ubuntu"
    bindir = base / "bin"
    libdir = base / "lib"
    bindir.mkdir(parents=True)
    libdir.mkdir(parents=True)
    (bindir / "elastix").write_text("x")
    vdir = root / "elastix" / "5.3.1"
    (vdir / "BIN_PATH.txt").write_text(str(bindir))
    (vdir / "LIB_PATH.txt").write_text(str(libdir))

    assert install.managed_elastix_bin_dir() == str(bindir)
    assert install.managed_elastix_lib_dir() == str(libdir)
    assert set(install.installed_elastix_versions()) == {"5.3.1"}
    # ANTs managed is a separate subtree and unaffected.
    assert install.managed_bin_dir() is None
    assert install.installed_versions() == {}


def test_resolve_binary_elastix_uses_env(monkeypatch):
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)
    monkeypatch.setattr(install, "managed_elastix_bin_dir", lambda version=None: None)
    monkeypatch.setenv("ELASTIXPATH", "/elx/bin")
    monkeypatch.setattr(
        executable,
        "_candidate_in_dir",
        lambda d, n: f"{d}/{n}" if str(d) == "/elx/bin" else None,
    )
    assert executable.resolve_binary("elastix") == "/elx/bin/elastix"


def test_resolve_binary_elastix_managed(monkeypatch):
    monkeypatch.delenv("ELASTIXPATH", raising=False)
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)
    monkeypatch.setattr(install, "managed_elastix_bin_dir", lambda version=None: "/m/elx/bin")
    monkeypatch.setattr(
        executable,
        "_candidate_in_dir",
        lambda d, n: f"{d}/{n}" if str(d) == "/m/elx/bin" else None,
    )
    assert executable.resolve_binary("transformix") == "/m/elx/bin/transformix"


def test_resolve_binary_elastix_not_found_message(monkeypatch):
    monkeypatch.delenv("ELASTIXPATH", raising=False)
    monkeypatch.delenv("COMMANDANTS_AUTO_INSTALL", raising=False)
    monkeypatch.setattr(executable.shutil, "which", lambda n, path=None: None)
    monkeypatch.setattr(install, "managed_elastix_bin_dir", lambda version=None: None)
    with pytest.raises(AntsNotFoundError) as exc:
        executable.resolve_binary("elastix")
    assert "install-elastix" in str(exc.value)
    assert "ELASTIXPATH" in str(exc.value)
