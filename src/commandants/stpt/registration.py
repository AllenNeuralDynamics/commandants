"""Allen STPT/CCF registration wrapper.

Drives the ``gridpoint`` driver -- a minimal ITK-only executable that exposes the
Allen registration algorithm (MI rigid -> MI affine -> discrete symmetric B-spline MRF)
for two arbitrary 3D volumes. The driver is built from a separate, Allen-licensed repo;
commandants only invokes it. See :mod:`commandants.stpt` for why the upstream pipeline
modules are not driven directly.

The interface mirrors the ANTs and elastix wrappers: build it, inspect it with
``to_shell()``, ``run()`` it, and read declared outputs off the result. Images may be
paths **or** in-memory ``SimpleITK.Image`` objects anywhere a path is expected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from ._base import StptToolCommand

PathLike = Union[str, Path]

#: Pipeline stages the driver understands, in execution order.
STAGES = ("rigid", "affine", "deformable")


class StptRegistration(StptToolCommand):
    """Builder for a ``gridpoint`` invocation.

    Parameters
    ----------
    fixed, moving:
        Fixed and moving images (path or in-memory SimpleITK image).
    out_dir:
        Output directory (created at run time). The driver names its own outputs there:
        ``stpt_Warped<ext>``, ``stpt_Affine.txt``, optionally ``stpt_Warp.nii.gz``, and
        ``stpt.log``.
    fixed_mask, moving_mask:
        Optional metric masks (path or SimpleITK image).
    stages:
        Which stages to run, any ordered subset of :data:`STAGES`.
    threads:
        Thread count.
    bins, linear_samples, linear_iterations:
        Linear (rigid/affine) mutual-information knobs. Upstream defaults are 64 bins,
        250000 spatial samples and 100 iterations; leave None to use the driver's own.
    levels, grid, regularization, samples:
        Deformable (discrete MRF) knobs: pyramid levels, the control-grid spacing schedule
        in voxels (e.g. ``(30, 15, 8, 4)``), the smoothness weight (the driver's
        ``--lambda``), and the random-sample count. Leave None for the driver's defaults.
    upstream_optimizer:
        Use the Allen optimizer constants verbatim (translation scale 1e7, step
        60 -> 0.1). The driver otherwise derives those from the image extent, because
        the originals are calibrated to CCF-sized volumes in their physical units and
        diverge on images of other sizes. Turn this on for fidelity on CCF-scale data;
        leave it off for anything smaller.
    write_deformation_field:
        Also write the dense deformation field.
    warped_ext:
        Extension for the warped image, default ``".nii.gz"``. Set ``".nii"`` to skip
        compression -- the cross-backend benchmark does this so timings stay comparable.
    verbose:
        Verbose driver output.
    stpt_path:
        Explicit directory containing the ``gridpoint`` binary.
    """

    binary_name = "gridpoint"

    def __init__(
        self,
        fixed: Any,
        moving: Any,
        out_dir: PathLike,
        *,
        fixed_mask: Any = None,
        moving_mask: Any = None,
        stages: Sequence[str] = STAGES,
        threads: Optional[int] = None,
        bins: Optional[int] = None,
        linear_samples: Optional[int] = None,
        linear_iterations: Optional[int] = None,
        levels: Optional[int] = None,
        grid: Optional[Union[str, Sequence[int]]] = None,
        regularization: Optional[float] = None,
        samples: Optional[int] = None,
        upstream_optimizer: bool = False,
        write_deformation_field: bool = False,
        warped_ext: str = ".nii.gz",
        verbose: bool = False,
        stpt_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=stpt_path)
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"Unknown stage(s) {unknown}; use any of {list(STAGES)}.")
        if not stages:
            raise ValueError("gridpoint needs at least one stage.")
        self.fixed = fixed
        self.moving = moving
        self.out_dir = out_dir
        self.fixed_mask = fixed_mask
        self.moving_mask = moving_mask
        self.stages = tuple(stages)
        self.threads = threads
        self.bins = bins
        self.linear_samples = linear_samples
        self.linear_iterations = linear_iterations
        self.levels = levels
        self.grid = grid
        self.regularization = regularization
        self.samples = samples
        self.upstream_optimizer = upstream_optimizer
        self.write_deformation_field = write_deformation_field
        self.warped_ext = warped_ext
        self.verbose = verbose

    def _build_args(self) -> List[str]:
        args: List[str] = [
            "-f",
            self._resolve(self.fixed, "fixed", must_exist=True),
            "-m",
            self._resolve(self.moving, "moving", must_exist=True),
            "-o",
            str(self.out_dir),
            "--stages",
            ",".join(self.stages),
        ]
        if self.fixed_mask is not None:
            args += ["--fixed-mask", self._resolve(self.fixed_mask, "fixed_mask", must_exist=True)]
        if self.moving_mask is not None:
            args += ["--moving-mask", self._resolve(self.moving_mask, "moving_mask", must_exist=True)]
        for flag, value in (
            ("--threads", self.threads),
            ("--bins", self.bins),
            ("--linear-samples", self.linear_samples),
            ("--linear-iterations", self.linear_iterations),
            ("--levels", self.levels),
            ("--lambda", self.regularization),
            ("--samples", self.samples),
        ):
            if value is not None:
                args += [flag, str(value)]
        if self.grid is not None:
            grid = self.grid if isinstance(self.grid, str) else ",".join(str(g) for g in self.grid)
            args += ["--grid", grid]
        args += ["--warped-ext", str(self.warped_ext)]
        if self.upstream_optimizer:
            args.append("--upstream-optimizer")
        if self.write_deformation_field:
            args.append("--write-deformation-field")
        if self.verbose:
            args.append("-v")
        return args

    def declared_outputs(self) -> dict:
        out = str(self.out_dir)
        outputs = {
            "warped": os.path.join(out, f"stpt_Warped{self.warped_ext}"),
            "log": os.path.join(out, "stpt.log"),
        }
        if "rigid" in self.stages or "affine" in self.stages:
            outputs["affine"] = os.path.join(out, "stpt_Affine.txt")
        if self.write_deformation_field:
            outputs["warp"] = os.path.join(out, "stpt_Warp.nii.gz")
        return outputs


__all__ = ["StptRegistration", "STAGES"]
