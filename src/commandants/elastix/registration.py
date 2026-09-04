"""``elastix`` registration wrapper.

elastix's CLI flags are only I/O -- ``-f -m -out -p -fMask -mMask -t0 -threads`` --
while the algorithm lives in the parameter file(s). This wrapper models exactly that:
you pass a fixed image, a moving image, an output directory, and one or more parameter
maps (a :class:`~commandants.elastix.ParameterMap`, a dict, or a path to a ``.txt``).
Multiple maps become repeated ``-p`` flags (sequential multi-stage registration).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from .parameters import ParameterMap
from ._base import ElastixToolCommand

PathLike = Union[str, Path]
ParamLike = Union[ParameterMap, dict, str, Path]


class Elastix(ElastixToolCommand):
    """Builder for an ``elastix`` invocation.

    Parameters
    ----------
    fixed, moving:
        Fixed and moving images (path or in-memory SimpleITK image).
    out_dir:
        Output directory elastix writes into (created at run time). elastix names
        its own outputs there: ``result.<i>.<ext>``, ``TransformParameters.<i>.txt``,
        ``elastix.log``.
    param_maps:
        One parameter map or a list of them (``ParameterMap`` / dict / path). Each
        becomes a ``-p`` flag; multiple maps run sequentially.
    fixed_mask, moving_mask:
        Optional metric masks (path or SimpleITK image).
    initial_transform:
        Optional initial ``TransformParameters.txt`` (``-t0``).
    threads:
        Thread count (``-threads``).
    result_format:
        Extension of the warped result image (e.g. ``"nii"``); inferred from the
        last map's ``ResultImageFormat`` if unset, defaulting to ``"nii"``.
    elastix_path:
        Explicit directory containing the ``elastix`` binary.
    """

    binary_name = "elastix"

    def __init__(
        self,
        fixed: Any,
        moving: Any,
        out_dir: PathLike,
        param_maps: Union[ParamLike, Sequence[ParamLike]],
        *,
        fixed_mask: Any = None,
        moving_mask: Any = None,
        initial_transform: Optional[ParamLike] = None,
        threads: Optional[int] = None,
        result_format: Optional[str] = None,
        elastix_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=elastix_path)
        self.fixed = fixed
        self.moving = moving
        self.out_dir = out_dir
        if isinstance(param_maps, (list, tuple)):
            self.param_maps: List[ParamLike] = list(param_maps)
        else:
            self.param_maps = [param_maps]
        self.fixed_mask = fixed_mask
        self.moving_mask = moving_mask
        self.initial_transform = initial_transform
        self.threads = threads
        self.result_format = result_format

    def add_param_map(self, param_map: ParamLike) -> "Elastix":
        """Append another parameter map (another ``-p`` / sequential stage)."""
        self.param_maps.append(param_map)
        return self

    def _build_args(self) -> List[str]:
        if not self.param_maps:
            raise ValueError("elastix needs at least one parameter map (-p).")
        args: List[str] = [
            "-f", self._resolve(self.fixed, "fixed"),
            "-m", self._resolve(self.moving, "moving"),
            "-out", str(self.out_dir),
        ]
        for i, pm in enumerate(self.param_maps):
            args += ["-p", self._resolve_param(pm, f"param{i}")]
        if self.fixed_mask is not None:
            args += ["-fMask", self._resolve(self.fixed_mask, "fixed_mask")]
        if self.moving_mask is not None:
            args += ["-mMask", self._resolve(self.moving_mask, "moving_mask")]
        if self.initial_transform is not None:
            args += ["-t0", self._resolve_param(self.initial_transform, "t0")]
        if self.threads is not None:
            args += ["-threads", str(self.threads)]
        return args

    def _result_ext(self) -> str:
        if self.result_format:
            return str(self.result_format)
        for pm in reversed(self.param_maps):
            fmt = _format_from(pm)
            if fmt:
                return fmt
        return "nii"

    def declared_outputs(self) -> dict:
        out = str(self.out_dir)
        n = len(self.param_maps)
        outputs = {"log": os.path.join(out, "elastix.log")}
        if n:
            ext = self._result_ext()
            outputs["warped"] = os.path.join(out, f"result.{n - 1}.{ext}")
            outputs["transform"] = os.path.join(out, f"TransformParameters.{n - 1}.txt")
            for i in range(n):
                outputs[f"transform{i}"] = os.path.join(out, f"TransformParameters.{i}.txt")
        return outputs


def _format_from(pm: ParamLike) -> Optional[str]:
    """Extract ResultImageFormat from a map/dict/file, if present."""
    m: Optional[ParameterMap] = None
    if isinstance(pm, ParameterMap):
        m = pm
    elif isinstance(pm, dict):
        m = ParameterMap(pm)
    elif isinstance(pm, (str, os.PathLike)) and os.path.isfile(pm):
        try:
            m = ParameterMap.from_file(pm)
        except Exception:
            return None
    if m is None or "ResultImageFormat" not in m:
        return None
    v = m["ResultImageFormat"]
    return str(v[0] if isinstance(v, (tuple, list)) else v)


__all__ = ["Elastix"]
