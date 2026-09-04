"""Shared base for the elastix/transformix command wrappers.

Both tools reuse the binary-agnostic :class:`~commandants.core.runner.AntsCommand`
core; this base adds the two elastix-specific bits: materializing in-memory
parameter maps to temp ``.txt`` files, and preparing execution (creating the
mandatory ``-out`` directory + putting a managed install's bundled ``lib`` on the
loader path).
"""

from __future__ import annotations

import os
import platform
from typing import Any, Mapping, Optional

from ..core.runner import AntsCommand
from .parameters import is_param_map, to_parameter_map


def library_path_env(
    env: Optional[Mapping[str, str]],
    binary: str,
    explicit_path: Any = None,
) -> Optional[Mapping[str, str]]:
    """Return ``env`` with a managed elastix install's libs on the loader path.

    elastix's prebuilt binaries need their bundled ``lib`` dir on
    ``LD_LIBRARY_PATH`` (Linux) / ``DYLD_LIBRARY_PATH`` (macOS); on Windows the
    DLLs sit next to the executables, so its ``bin`` dir goes on ``PATH``. Only
    applied for a commandants-managed install (a PATH/conda elastix manages its
    own libs); otherwise ``env`` is returned unchanged.
    """
    system = platform.system()
    if system == "Windows":
        from ..install import managed_elastix_bin_dir

        extra_dir = managed_elastix_bin_dir()
        var = "PATH"
    else:
        from ..install import managed_elastix_lib_dir

        extra_dir = managed_elastix_lib_dir()
        var = "DYLD_LIBRARY_PATH" if system == "Darwin" else "LD_LIBRARY_PATH"

    if not extra_dir:
        return env

    base = dict(env) if env is not None else dict(os.environ)
    existing = base.get(var, "")
    base[var] = extra_dir + (os.pathsep + existing if existing else "")
    return base


class ElastixToolCommand(AntsCommand):
    """Base for :class:`Elastix`/:class:`Transformix`; each sets ``self.out_dir``."""

    out_dir: Any

    def _resolve_param(self, param: Any, name: str) -> str:
        """Path -> str; in-memory ParameterMap/dict -> a temp ``.txt`` (when
        materializing) or a ``<param:...>`` placeholder (in preview)."""
        if is_param_map(param):
            if not self._materialize:
                return f"<param:{name}>"
            ws = self._ensure_workspace()
            path = os.path.join(ws.dir, f"{name}.txt")
            to_parameter_map(param).write(path)
            ws.files.append(path)
            ws.inputs[name] = path
            return path
        return str(param)

    def _prepare_execution(
        self, env: Mapping[str, str] | None
    ) -> Mapping[str, str] | None:
        os.makedirs(str(self.out_dir), exist_ok=True)
        return library_path_env(env, self.binary, self.ants_path)


__all__ = ["ElastixToolCommand", "library_path_env"]
