"""``transformix`` wrapper -- apply an elastix transform to an image or points.

elastix stores its result transform as ``TransformParameters.txt``; you apply it with
the separate ``transformix`` binary. This wraps its modes: warp an image (``-in``),
transform a point set (``-def points.txt``), and/or write the deformation field
(``-def all``) or the spatial Jacobian (``-jac all``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union

from .parameters import ParameterMap
from ._base import ElastixToolCommand

PathLike = Union[str, Path]
ParamLike = Union[ParameterMap, dict, str, Path]


class Transformix(ElastixToolCommand):
    """Builder for a ``transformix`` invocation.

    Parameters
    ----------
    transform_parameters:
        The elastix ``TransformParameters.txt`` to apply (path, or an in-memory
        map). Chained transforms are followed via the file's ``InitialTransform...``.
    out_dir:
        Output directory (created at run time).
    moving:
        Image to warp (``-in``; path or SimpleITK image).
    points:
        Point file to transform (``-def <points>``). Note: transformix maps points
        fixed->moving and uses elastix's own point-file format (not the ANTs CSV).
    deformation_field:
        If True, write the deformation field (``-def all``). Mutually exclusive
        with ``points``.
    jacobian:
        If True, write the spatial Jacobian (``-jac all``).
    threads:
        Thread count.
    elastix_path:
        Explicit directory containing the ``transformix`` binary.
    """

    binary_name = "transformix"

    def __init__(
        self,
        transform_parameters: ParamLike,
        out_dir: PathLike,
        *,
        moving: Any = None,
        points: Optional[PathLike] = None,
        deformation_field: bool = False,
        jacobian: bool = False,
        threads: Optional[int] = None,
        elastix_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=elastix_path)
        if points is not None and deformation_field:
            raise ValueError("Pass either points or deformation_field, not both (one -def).")
        if moving is None and points is None and not deformation_field and not jacobian:
            raise ValueError(
                "transformix needs something to do: set moving=, points=, "
                "deformation_field=True, or jacobian=True."
            )
        self.transform_parameters = transform_parameters
        self.out_dir = out_dir
        self.moving = moving
        self.points = points
        self.deformation_field = deformation_field
        self.jacobian = jacobian
        self.threads = threads

    def _build_args(self) -> List[str]:
        args: List[str] = [
            "-tp", self._resolve_param(self.transform_parameters, "tp"),
            "-out", str(self.out_dir),
        ]
        if self.moving is not None:
            args += ["-in", self._resolve(self.moving, "input")]
        if self.points is not None:
            args += ["-def", str(self.points)]
        elif self.deformation_field:
            args += ["-def", "all"]
        if self.jacobian:
            args += ["-jac", "all"]
        if self.threads is not None:
            args += ["-threads", str(self.threads)]
        return args

    def declared_outputs(self) -> dict:
        out = str(self.out_dir)
        outputs = {"log": os.path.join(out, "transformix.log")}
        if self.moving is not None:
            outputs["warped"] = os.path.join(out, "result.nii")
        if self.points is not None:
            outputs["points"] = os.path.join(out, "outputpoints.txt")
        if self.deformation_field:
            outputs["deformation"] = os.path.join(out, "deformationField.nii")
        if self.jacobian:
            outputs["jacobian"] = os.path.join(out, "spatialJacobian.nii")
        return outputs


__all__ = ["Transformix"]
