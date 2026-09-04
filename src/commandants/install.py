"""Provision official prebuilt registration binaries on demand (ANTs and elastix).

Neither ANTsPyX nor SimpleElastix ships the standalone command-line executables, so
this module downloads the official prebuilt binaries for the current platform from
the tool's GitHub releases, unpacks them into a managed per-user directory, and
records where the ``bin`` (and, for elastix, ``lib``) directory landed.
:func:`commandants.core.executable.resolve_binary` then discovers them automatically
(as a fallback after PATH).

The machinery is shared between tools via :class:`ToolSpec`; the public ANTs API
(:func:`install_ants` etc.) and the elastix API (:func:`install_elastix` etc.) are
thin wrappers over the same generic core. Nothing downloads implicitly unless you
opt in (``auto_install=True`` or ``COMMANDANTS_AUTO_INSTALL``); normally you run
``commandants install-ants`` / ``commandants install-elastix`` once.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .core.exceptions import CommandantsError

#: Pinned default ANTs version (override with version=... / --version / "latest").
DEFAULT_VERSION = "2.6.5"
#: Pinned default elastix version.
DEFAULT_ELASTIX_VERSION = "5.3.1"

_GITHUB_API = "https://api.github.com/repos"
# Preferred Linux distro archives for ANTs, best first (x86_64 only).
_LINUX_PREFERENCE = [
    "ubuntu-22.04",
    "ubuntu-24.04",
    "ubuntu20.04",
    "ubuntu18.04",
    "almalinux9",
    "almalinux8",
    "centos7",
]


# --------------------------------------------------------------------------- #
# Tool specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolSpec:
    """Everything provisioning/resolution needs to know about one tool."""

    name: str                     # "ants" | "elastix"
    repo: str                     # GitHub "owner/repo"
    managed_subdir: str           # subdir under user_data_dir()
    marker_binaries: frozenset    # e.g. {"antsRegistration", "antsRegistration.exe"}
    env_var: str                  # "ANTSPATH" | "ELASTIXPATH"
    default_version: str
    needs_lib_path: bool          # elastix bundles a lib/ that must be on the loader path
    match_asset: Callable[..., str]
    progress_label: str           # "ANTs" | "elastix"
    tag_prefix: str = ""          # "v" for ANTs tags, "" for bare elastix tags

    def tag(self, version: str) -> str:
        return f"{self.tag_prefix}{version.lstrip('v')}"


# --------------------------------------------------------------------------- #
# Paths / markers
# --------------------------------------------------------------------------- #
def user_data_dir() -> str:
    """Return the per-user data directory commandants writes into."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "commandants")


def _tool_root(spec: "ToolSpec") -> str:
    return os.path.join(user_data_dir(), spec.managed_subdir)


def _ants_root() -> str:  # kept for back-compat
    return _tool_root(ANTS_SPEC)


def _marker_path(version_dir: str) -> str:
    return os.path.join(version_dir, "BIN_PATH.txt")


def _lib_marker_path(version_dir: str) -> str:
    return os.path.join(version_dir, "LIB_PATH.txt")


def _read_marker(marker: str) -> Optional[str]:
    if not os.path.isfile(marker):
        return None
    with open(marker) as fh:
        path = fh.read().strip()
    return path if path and os.path.isdir(path) else None


def _version_key(v: str):
    parts = []
    for chunk in v.lstrip("v").split("."):
        parts.append(int(chunk) if chunk.isdigit() else 0)
    return tuple(parts)


# --------------------------------------------------------------------------- #
# Discovery (generic + per-tool wrappers)
# --------------------------------------------------------------------------- #
def _version_dirs(spec: "ToolSpec") -> Dict[str, str]:
    root = _tool_root(spec)
    out: Dict[str, str] = {}
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        vdir = os.path.join(root, name)
        if _read_marker(_marker_path(vdir)):
            out[name] = vdir
    return out


def _select_version_dir(spec: "ToolSpec", version: Optional[str] = None) -> Optional[str]:
    dirs = _version_dirs(spec)
    if not dirs:
        return None
    if version is not None:
        return dirs.get(version.lstrip("v"))
    return dirs[sorted(dirs, key=_version_key)[-1]]


def _installed_versions(spec: "ToolSpec") -> Dict[str, str]:
    return {v: _read_marker(_marker_path(d)) for v, d in _version_dirs(spec).items()}


def _managed_bin_dir(spec: "ToolSpec", version: Optional[str] = None) -> Optional[str]:
    vdir = _select_version_dir(spec, version)
    return _read_marker(_marker_path(vdir)) if vdir else None


def _managed_lib_dir(spec: "ToolSpec", version: Optional[str] = None) -> Optional[str]:
    vdir = _select_version_dir(spec, version)
    if not vdir:
        return None
    marker = _lib_marker_path(vdir)
    if not os.path.isfile(marker):
        return None
    with open(marker) as fh:
        p = fh.read().strip()
    return p if p and os.path.isdir(p) else None


def installed_versions() -> Dict[str, str]:
    """Return ``{version: bin_dir}`` for every managed ANTs install."""
    return _installed_versions(ANTS_SPEC)


def managed_bin_dir(version: Optional[str] = None) -> Optional[str]:
    """Return the ``bin`` dir of a managed ANTs install (newest if unset), or ``None``."""
    return _managed_bin_dir(ANTS_SPEC, version)


def installed_elastix_versions() -> Dict[str, str]:
    """Return ``{version: bin_dir}`` for every managed elastix install."""
    return _installed_versions(ELASTIX_SPEC)


def managed_elastix_bin_dir(version: Optional[str] = None) -> Optional[str]:
    """Return the ``bin`` dir of a managed elastix install (newest if unset), or ``None``."""
    return _managed_bin_dir(ELASTIX_SPEC, version)


def managed_elastix_lib_dir(version: Optional[str] = None) -> Optional[str]:
    """Return the bundled ``lib`` dir of a managed elastix install, or ``None``."""
    return _managed_lib_dir(ELASTIX_SPEC, version)


# --------------------------------------------------------------------------- #
# Asset selection
# --------------------------------------------------------------------------- #
def select_asset(
    names: List[str],
    system: Optional[str] = None,
    machine: Optional[str] = None,
) -> str:
    """Pick the best ANTs release asset name for a platform."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()

    if system == "windows":
        cands = [n for n in names if "windows" in n.lower()]
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            cands = [n for n in names if "macos" in n.lower() and "arm64" in n.lower()]
        else:
            cands = [
                n
                for n in names
                if "macos" in n.lower() and ("intel" in n.lower() or "x64" in n.lower())
            ]
    else:  # linux and friends
        if machine not in ("x86_64", "amd64", "x64"):
            raise CommandantsError(
                f"No prebuilt ANTs binary is published for {system}/{machine}. "
                "Build ANTs from source or pass an explicit asset= name."
            )
        linux = [
            n for n in names if any(k in n.lower() for k in ("ubuntu", "almalinux", "centos"))
        ]
        cands = []
        for pref in _LINUX_PREFERENCE:
            cands = [n for n in linux if pref in n.lower()]
            if cands:
                break
        if not cands:
            cands = linux

    zips = [n for n in cands if n.lower().endswith(".zip")]
    chosen = zips or cands
    if not chosen:
        raise CommandantsError(
            f"Could not match a prebuilt ANTs asset for {system}/{machine} among "
            f"{names}. Pass asset= explicitly."
        )
    return chosen[0]


def select_elastix_asset(
    names: List[str],
    system: Optional[str] = None,
    machine: Optional[str] = None,
) -> str:
    """Pick the elastix release asset (one .zip per OS: windows/ubuntu/macos)."""
    system = (system or platform.system()).lower()
    key = "windows" if system == "windows" else ("macos" if system == "darwin" else "ubuntu")
    cands = [n for n in names if key in n.lower() and n.lower().endswith(".zip")]
    if not cands:
        raise CommandantsError(
            f"No prebuilt elastix asset for {system} among {names}. Pass asset= explicitly."
        )
    return cands[0]


# --------------------------------------------------------------------------- #
# Network + extraction (generic)
# --------------------------------------------------------------------------- #
def _fetch_release(spec: "ToolSpec", version: Optional[str]) -> dict:
    base = f"{_GITHUB_API}/{spec.repo}/releases"
    url = f"{base}/latest" if version in (None, "latest") else f"{base}/tags/{spec.tag(version)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "commandants", "Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted host)
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network dependent
        raise CommandantsError(
            f"Failed to query {spec.progress_label} release {version!r}: {exc}"
        ) from exc


def _download(url: str, dest: str, label: str = "ANTs", quiet: bool = False) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "commandants"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted host)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 256
        with open(dest, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if not quiet and total:
                    pct = 100 * done / total
                    print(
                        f"\r  downloading {label}: {pct:5.1f}% ({done >> 20}/{total >> 20} MiB)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
        if not quiet and total:
            print(file=sys.stderr)


def _safe_extract(zip_path: str, dest: str) -> None:
    """Extract a zip, guarding against path traversal and restoring exec bits."""
    dest_abs = os.path.abspath(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(dest, member.filename))
            if not target.startswith(dest_abs + os.sep) and target != dest_abs:
                raise CommandantsError(f"Unsafe path in archive: {member.filename!r}")
            zf.extract(member, dest)
            mode = member.external_attr >> 16
            if mode:
                try:
                    os.chmod(os.path.join(dest, member.filename), mode)
                except OSError:  # pragma: no cover - platform dependent
                    pass


def _find_bin_dir(root: str, markers: Optional[frozenset] = None) -> str:
    """Locate the directory containing a marker binary inside an extracted tree."""
    exe_names = markers or ANTS_SPEC.marker_binaries
    for dirpath, _dirs, files in os.walk(root):
        if exe_names & set(files):
            return dirpath
    raise CommandantsError(
        f"Extracted archive under {root!r} but found no {sorted(exe_names)[0]} binary."
    )


def _find_lib_dir(root: str) -> Optional[str]:
    """Locate a bundled ``lib`` directory inside an extracted tree (elastix)."""
    for dirpath, _dirs, _files in os.walk(root):
        if os.path.basename(dirpath) == "lib":
            return dirpath
    return None


# --------------------------------------------------------------------------- #
# Install / uninstall (generic)
# --------------------------------------------------------------------------- #
def _install(
    spec: "ToolSpec",
    version: str,
    dest: Optional[str],
    asset: Optional[str],
    force: bool,
    quiet: bool,
) -> str:
    release = _fetch_release(spec, version)
    tag = release.get("tag_name", "")
    resolved_version = tag.lstrip("v") or (version if version != "latest" else "unknown")
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    if not assets:
        raise CommandantsError(
            f"{spec.progress_label} release {tag or version!r} has no downloadable assets."
        )

    name = asset or spec.match_asset(list(assets))
    if name not in assets:
        raise CommandantsError(
            f"Asset {name!r} not found in release {tag!r}. Available: {sorted(assets)}"
        )

    root = dest or _tool_root(spec)
    version_dir = os.path.join(root, resolved_version)
    marker = _marker_path(version_dir)

    existing = _read_marker(marker)
    if existing and not force:
        if not quiet:
            print(f"{spec.progress_label} {resolved_version} already installed at {existing}",
                  file=sys.stderr)
        return existing

    if os.path.isdir(version_dir) and force:
        shutil.rmtree(version_dir, ignore_errors=True)
    os.makedirs(version_dir, exist_ok=True)

    if not quiet:
        print(f"Installing {spec.progress_label} {resolved_version} ({name})", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        arch_path = os.path.join(tmp, name)
        _download(assets[name], arch_path, spec.progress_label, quiet=quiet)
        _safe_extract(arch_path, version_dir)

    bindir = _find_bin_dir(version_dir, spec.marker_binaries)
    with open(marker, "w") as fh:
        fh.write(bindir)
    if spec.needs_lib_path:
        libdir = _find_lib_dir(version_dir)
        if libdir:
            with open(_lib_marker_path(version_dir), "w") as fh:
                fh.write(libdir)
    if not quiet:
        print(f"{spec.progress_label} {resolved_version} ready. bin: {bindir}", file=sys.stderr)
    return bindir


def _uninstall(spec: "ToolSpec", version: Optional[str] = None) -> List[str]:
    root = _tool_root(spec)
    removed: List[str] = []
    if not os.path.isdir(root):
        return removed
    targets = [version.lstrip("v")] if version else list(_installed_versions(spec))
    for ver in targets:
        vdir = os.path.join(root, ver)
        if os.path.isdir(vdir):
            shutil.rmtree(vdir, ignore_errors=True)
            removed.append(vdir)
    return removed


# --------------------------------------------------------------------------- #
# Public per-tool wrappers
# --------------------------------------------------------------------------- #
def install_ants(
    version: str = DEFAULT_VERSION,
    dest: Optional[str] = None,
    asset: Optional[str] = None,
    force: bool = False,
    quiet: bool = False,
) -> str:
    """Download and unpack prebuilt ANTs binaries; return the ``bin`` directory."""
    return _install(ANTS_SPEC, version, dest, asset, force, quiet)


def uninstall_ants(version: Optional[str] = None) -> List[str]:
    """Remove managed ANTs install(s); return the version dirs removed."""
    return _uninstall(ANTS_SPEC, version)


def install_elastix(
    version: str = DEFAULT_ELASTIX_VERSION,
    dest: Optional[str] = None,
    asset: Optional[str] = None,
    force: bool = False,
    quiet: bool = False,
) -> str:
    """Download and unpack prebuilt elastix binaries; return the ``bin`` directory."""
    return _install(ELASTIX_SPEC, version, dest, asset, force, quiet)


def uninstall_elastix(version: Optional[str] = None) -> List[str]:
    """Remove managed elastix install(s); return the version dirs removed."""
    return _uninstall(ELASTIX_SPEC, version)


# --------------------------------------------------------------------------- #
# Specs + binary → spec routing
# --------------------------------------------------------------------------- #
ANTS_SPEC = ToolSpec(
    name="ants",
    repo="ANTsX/ANTs",
    managed_subdir="ants",
    marker_binaries=frozenset({"antsRegistration", "antsRegistration.exe"}),
    env_var="ANTSPATH",
    default_version=DEFAULT_VERSION,
    needs_lib_path=False,
    match_asset=select_asset,
    progress_label="ANTs",
    tag_prefix="v",
)

ELASTIX_SPEC = ToolSpec(
    name="elastix",
    repo="SuperElastix/elastix",
    managed_subdir="elastix",
    marker_binaries=frozenset({"elastix", "elastix.exe"}),
    env_var="ELASTIXPATH",
    default_version=DEFAULT_ELASTIX_VERSION,
    needs_lib_path=True,
    match_asset=select_elastix_asset,
    progress_label="elastix",
    tag_prefix="",
)

_ELASTIX_BINARIES = frozenset(
    {"elastix", "elastix.exe", "transformix", "transformix.exe"}
)


def spec_for_binary(name: str) -> "ToolSpec":
    """Return the ToolSpec that owns a binary name (elastix/transformix → elastix)."""
    return ELASTIX_SPEC if os.path.basename(str(name)) in _ELASTIX_BINARIES else ANTS_SPEC


__all__ = [
    "DEFAULT_VERSION",
    "DEFAULT_ELASTIX_VERSION",
    "ToolSpec",
    "ANTS_SPEC",
    "ELASTIX_SPEC",
    "spec_for_binary",
    # ANTs
    "install_ants",
    "uninstall_ants",
    "managed_bin_dir",
    "installed_versions",
    "select_asset",
    # elastix
    "install_elastix",
    "uninstall_elastix",
    "managed_elastix_bin_dir",
    "managed_elastix_lib_dir",
    "installed_elastix_versions",
    "select_elastix_asset",
    "user_data_dir",
]
