"""``antsRegistration`` wrapper -- the centerpiece of commandants.

Unlike ANTsPyX's ``ants.registration()`` (which expands a curated parameter set
into a fixed, canned sequence of stages), :class:`AntsRegistration` models the
real command grammar: a list of stages, each with its own transform, one or more
metrics, and multi-resolution schedule. That is what makes multivariate and
point-constrained registration possible.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from ..core.params import bracket, xjoin
from ..core.runner import AntsCommand
from .metrics import Metric
from .stage import Convergence, Stage
from .transforms import Transform

PathLike = Union[str, Path]

# initializationFeature codes for image-based initial transforms.
_INIT_FEATURES = {"geometric": 0, "center-of-mass": 1, "com": 1, "origin": 2}


class AntsRegistration(AntsCommand):
    """Builder for a full ``antsRegistration`` invocation.

    Parameters
    ----------
    dimensionality:
        Image dimensionality (2, 3, or 4).
    output:
        Output transform prefix (the ANTs ``outputTransformPrefix``).
    warped_output, inverse_warped_output:
        Optional warped-image outputs; when either is given the ``--output``
        flag is emitted in its bracketed ``[prefix,warped,inverse]`` form.
    interpolation:
        Final-resampling interpolator (e.g. ``"Linear"``, ``"BSpline"``).
    winsorize:
        ``(lower_quantile, upper_quantile)`` for intensity winsorization.
    use_histogram_matching:
        Match moving-to-fixed histograms before registration.
    collapse_output_transforms:
        Collapse the transform stack (ANTs default is on).
    write_composite_transform:
        Also write ``{prefix}Composite.h5`` / ``{prefix}InverseComposite.h5``.
    use_float:
        Use single-precision computation.
    random_seed:
        Deterministic seed.
    restrict_deformation:
        Per-dimension deformation weights, e.g. ``[1, 1, 0]`` to forbid motion
        along the 3rd axis.
    write_interval_volumes:
        Write intermediate volumes every N iterations (debugging).
    verbose:
        Verbose ANTs output.
    ants_path:
        Explicit directory containing the ANTs binaries.
    """

    binary_name = "antsRegistration"

    def __init__(
        self,
        dimensionality: int,
        output: Optional[PathLike] = None,
        *,
        warped_output: Optional[PathLike] = None,
        inverse_warped_output: Optional[PathLike] = None,
        interpolation: Optional[str] = None,
        winsorize: Optional[Sequence[float]] = None,
        use_histogram_matching: Optional[bool] = None,
        collapse_output_transforms: Optional[bool] = None,
        write_composite_transform: bool = False,
        use_float: Optional[bool] = True,  # single precision by default (memory-friendly)
        random_seed: Optional[int] = None,
        restrict_deformation: Optional[Sequence[float]] = None,
        write_interval_volumes: Optional[int] = None,
        verbose: bool = False,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.output = output
        self.warped_output = warped_output
        self.inverse_warped_output = inverse_warped_output
        self.interpolation = interpolation
        self.winsorize = winsorize
        self.use_histogram_matching = use_histogram_matching
        self.collapse_output_transforms = collapse_output_transforms
        self.write_composite_transform = write_composite_transform
        self.use_float = use_float
        self.random_seed = random_seed
        self.restrict_deformation = restrict_deformation
        self.write_interval_volumes = write_interval_volumes
        self.verbose = verbose

        # Stored raw (not pre-rendered) so in-memory images resolve at run time.
        # ("images", fixed, moving, feature) or ("file", transform, use_inverse).
        self._initial_transforms: List[Tuple[Any, ...]] = []
        self._fixed_mask: Optional[Any] = None
        self._moving_mask: Optional[Any] = None
        self.stages: List[Stage] = []

    # -- initial transforms ---------------------------------------------------
    def initialize_from_images(
        self,
        fixed: PathLike,
        moving: PathLike,
        init: Union[str, int] = "center-of-mass",
    ) -> "AntsRegistration":
        """Add an image-based initial moving transform.

        ``init`` selects the ANTs initialization feature: ``"geometric"`` (0),
        ``"center-of-mass"`` (1, the default), or ``"origin"`` (2). An integer is
        passed through unchanged.
        """
        if isinstance(init, str):
            if init not in _INIT_FEATURES:
                raise ValueError(
                    f"init must be one of {sorted(_INIT_FEATURES)} or an int; got {init!r}"
                )
            feature = _INIT_FEATURES[init]
        else:
            feature = int(init)
        self._initial_transforms.append(("images", fixed, moving, feature))
        return self

    def add_initial_moving_transform(
        self, transform: PathLike, use_inverse: bool = False
    ) -> "AntsRegistration":
        """Prepend an existing transform file as the initial moving transform."""
        self._initial_transforms.append(("file", transform, use_inverse))
        return self

    # -- masks ----------------------------------------------------------------
    def set_masks(
        self,
        fixed_mask: Optional[PathLike] = None,
        moving_mask: Optional[PathLike] = None,
    ) -> "AntsRegistration":
        """Set the fixed and/or moving metric masks (applied to all stages)."""
        self._fixed_mask = fixed_mask
        self._moving_mask = moving_mask
        return self

    # -- stages ---------------------------------------------------------------
    def add_stage(
        self,
        transform: Transform,
        metrics: Union[Metric, Sequence[Metric]],
        convergence: Convergence,
        shrink_factors: Sequence[int],
        smoothing_sigmas: Sequence[float],
        smoothing_units: Optional[str] = "vox",
        fixed_mask: Optional[Any] = None,
        moving_mask: Optional[Any] = None,
    ) -> "AntsRegistration":
        """Append a registration stage. ``metrics`` may be one metric or many.

        ``fixed_mask`` / ``moving_mask`` restrict the metric for *this stage only*
        (path or SimpleITK image). When any stage sets a mask, ANTs requires one
        ``--masks`` per stage in order, so unmasked stages emit ``[NA,NA]``; a
        stage with no mask of its own falls back to any global :meth:`set_masks`.
        """
        metric_list = [metrics] if isinstance(metrics, Metric) else list(metrics)
        stage = Stage(
            transform=transform,
            metrics=metric_list,
            convergence=convergence,
            shrink_factors=shrink_factors,
            smoothing_sigmas=smoothing_sigmas,
            smoothing_units=smoothing_units,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )
        self.stages.append(stage)
        return self

    def add(self, stage: Stage) -> "AntsRegistration":
        """Append a pre-built :class:`Stage`."""
        self.stages.append(stage)
        return self

    # -- argv assembly --------------------------------------------------------
    def _output_arg(self) -> str:
        if self.warped_output is not None or self.inverse_warped_output is not None:
            return bracket(self.output, self.warped_output, self.inverse_warped_output)
        return str(self.output)

    def _mask_bracket(self, fixed_mask: Any, moving_mask: Any) -> str:
        """Build a ``[fixedMask,movingMask]`` arg, using ANTs' NA placeholder for
        an unset side (so a moving-only mask is ``[NA,movingMask]``)."""
        f = self._resolve(fixed_mask, "fixed_mask") if fixed_mask is not None else "NA"
        m = self._resolve(moving_mask, "moving_mask") if moving_mask is not None else "NA"
        return f"[{f},{m}]"

    def _build_args(self) -> List[str]:
        if not self.stages:
            raise ValueError("At least one stage must be added before running.")
        if self.output is None:
            raise ValueError("An output prefix must be set.")

        args: List[str] = ["--dimensionality", str(self.dimensionality)]

        if self.use_float is not None:
            args += ["--float", "1" if self.use_float else "0"]

        args += ["--output", self._output_arg()]

        if self.interpolation is not None:
            args += ["--interpolation", self.interpolation]

        if self.use_histogram_matching is not None:
            args += [
                "--use-histogram-matching",
                "1" if self.use_histogram_matching else "0",
            ]

        if self.winsorize is not None:
            args += ["--winsorize-image-intensities", bracket(*self.winsorize)]

        if self.write_composite_transform:
            args += ["--write-composite-transform", "1"]

        if self.collapse_output_transforms is not None:
            args += [
                "--collapse-output-transforms",
                "1" if self.collapse_output_transforms else "0",
            ]

        if self.restrict_deformation is not None:
            args += ["--restrict-deformation", xjoin(self.restrict_deformation)]

        if self.random_seed is not None:
            args += ["--random-seed", str(self.random_seed)]

        if self.write_interval_volumes is not None:
            args += ["--write-interval-volumes", str(self.write_interval_volumes)]

        for item in self._initial_transforms:
            if item[0] == "images":
                _, fixed, moving, feature = item
                rendered = bracket(
                    self._resolve(fixed, "init_fixed"),
                    self._resolve(moving, "init_moving"),
                    feature,
                )
            else:  # ("file", transform, use_inverse)
                _, transform, use_inverse = item
                path = self._resolve(transform, "init_transform")
                rendered = bracket(path, True) if use_inverse else path
            args += ["--initial-moving-transform", rendered]

        # Masks: ANTs takes either one global --masks (all stages) or one per
        # stage in order (with [NA,NA] for unmasked stages). Use per-stage
        # emission if any stage defines its own mask; otherwise a single global.
        per_stage_masks = any(
            (s.fixed_mask is not None or s.moving_mask is not None) for s in self.stages
        )
        global_masks = self._fixed_mask is not None or self._moving_mask is not None

        if global_masks and not per_stage_masks:
            args += ["--masks", self._mask_bracket(self._fixed_mask, self._moving_mask)]

        for stage in self.stages:
            args += stage.to_args(self._resolve)
            if per_stage_masks:
                fm = stage.fixed_mask if stage.fixed_mask is not None else self._fixed_mask
                mm = stage.moving_mask if stage.moving_mask is not None else self._moving_mask
                args += ["--masks", self._mask_bracket(fm, mm)]

        args += ["--verbose", "1" if self.verbose else "0"]
        return args

    def estimate_resources(
        self,
        fixed: Any = None,
        shape: Optional[Sequence[int]] = None,
        threads: int = 1,
        rate: float = 6.0e8,
    ):
        """Estimate peak memory (and a rough runtime) for this registration.

        Heuristic, not a measurement -- see :mod:`commandants.estimate`. Needs the
        fixed image's voxel dimensions: inferred from the registration's init/metric
        images, or pass ``fixed=<path/SimpleITK image>`` or ``shape=(x, y, z)``.

        Returns a :class:`~commandants.estimate.ResourceEstimate`; print it or read
        ``.peak_memory_bytes`` / ``.peak_memory_human`` / ``.est_runtime_seconds``.
        """
        from ..estimate import estimate_registration

        return estimate_registration(self, fixed=fixed, shape=shape, threads=threads, rate=rate)

    def expected_transforms(self, cwd: str | os.PathLike[str] | None = None) -> dict:
        """Predict the transform files ANTs will write, and how to apply them.

        Returns a dict with:

        * ``files``   -- the transform files that will be written, in order
          (as ANTs names them, i.e. prefixed by ``output``);
        * ``files_abs`` -- the same files as absolute paths;
        * ``output_dir`` -- the absolute folder the files will land in;
        * ``forward`` -- the ``-t`` list to warp *moving -> fixed* (deformable
          first, then affine), ready to feed to :class:`AntsApplyTransforms`;
        * ``inverse`` -- the ``-t`` list to warp *fixed -> moving*; affine entries
          appear as ``(path, invert=True)`` tuples;
        * ``warped`` / ``inverse_warped`` -- the warped-image outputs, if set.

        The folder is derived from the ``output`` prefix: a bare prefix like
        ``"reg_"`` resolves against ``cwd`` (defaulting to the current working
        directory, i.e. where ``run()`` will launch ANTs), while a prefix with a
        path component (``"/data/sub01/reg_"``) points at that directory. ANTs
        does **not** create a missing output folder -- make sure it exists.

        Naming rules (matching ANTs):

        * With ``collapse_output_transforms`` on (the recommended default here),
          consecutive **linear** stages (Translation/Rigid/Similarity/Affine) --
          plus the center-of-mass init -- collapse into a single
          ``{prefix}{i}GenericAffine.mat``. Each **deformable** stage (SyN, ...)
          writes ``{prefix}{i}Warp.nii.gz`` and ``{prefix}{i}InverseWarp.nii.gz``.
          ``i`` is the position in the collapsed stack.
        * With ``write_composite_transform`` on, ANTs instead writes single
          ``{prefix}Composite.h5`` / ``{prefix}InverseComposite.h5`` files.

        So a Rigid -> Affine -> SyN run with ``output="out_"`` produces
        ``out_0GenericAffine.mat`` (rigid+affine combined), ``out_1Warp.nii.gz``,
        and ``out_1InverseWarp.nii.gz``.

        This is a best-effort prediction for standard pipelines; the ANTs log is
        always the ground truth.
        """
        from .transforms import Affine, CompositeAffine, Rigid, Similarity, Translation

        linear = (Translation, Rigid, Similarity, Affine, CompositeAffine)
        prefix = "" if self.output is None else str(self.output)
        result: dict = {"prefix": prefix}

        if self.write_composite_transform:
            comp = f"{prefix}Composite.h5"
            inv = f"{prefix}InverseComposite.h5"
            result.update(files=[comp, inv], forward=[comp], inverse=[inv])
        else:
            collapse = (
                True
                if self.collapse_output_transforms is None
                else self.collapse_output_transforms
            )
            # Reduce stages to ordered output "units": collapsed-linear or warp.
            units: list[str] = []
            if collapse:
                j = 0
                while j < len(self.stages):
                    if isinstance(self.stages[j].transform, linear):
                        units.append("affine")
                        while j < len(self.stages) and isinstance(
                            self.stages[j].transform, linear
                        ):
                            j += 1
                    else:
                        units.append("warp")
                        j += 1
            else:
                units = [
                    "affine" if isinstance(s.transform, linear) else "warp"
                    for s in self.stages
                ]

            named: list[tuple] = []
            files: list[str] = []
            for idx, unit in enumerate(units):
                if unit == "affine":
                    mat = f"{prefix}{idx}GenericAffine.mat"
                    files.append(mat)
                    named.append(("affine", mat, None))
                else:
                    fwd = f"{prefix}{idx}Warp.nii.gz"
                    inv = f"{prefix}{idx}InverseWarp.nii.gz"
                    files += [fwd, inv]
                    named.append(("warp", fwd, inv))

            # moving->fixed: apply linear then warp, so the -t list is reversed.
            forward = [u[1] for u in reversed(named)]
            # fixed->moving: inverse-warp then inverse-affine, list in stage order.
            inverse: list = []
            for kind, fwd, inv in named:
                inverse.append((fwd, True) if kind == "affine" else inv)
            result.update(files=files, forward=forward, inverse=inverse)

        if self.warped_output is not None:
            result["warped"] = str(self.warped_output)
        if self.inverse_warped_output is not None:
            result["inverse_warped"] = str(self.inverse_warped_output)

        # Resolve the folder the files will actually land in.
        root = os.path.abspath(str(cwd)) if cwd is not None else os.getcwd()
        prefix_dir = os.path.dirname(prefix)
        result["output_dir"] = (
            os.path.abspath(os.path.join(root, prefix_dir)) if prefix_dir else root
        )
        result["files_abs"] = [
            os.path.abspath(os.path.join(root, f)) for f in result["files"]
        ]
        return result

    def declared_outputs(self) -> dict:
        """Predictable, user-named outputs (warped images and composites).

        The numbered per-transform files (e.g. ``{prefix}0GenericAffine.mat``,
        ``{prefix}1Warp.nii.gz``) depend on the exact stage stack and are not
        declared here; inspect the output directory or the ANTs log for those.
        """
        outputs: dict[str, str] = {}
        if self.warped_output is not None:
            outputs["warped"] = str(self.warped_output)
        if self.inverse_warped_output is not None:
            outputs["inverse_warped"] = str(self.inverse_warped_output)
        if self.write_composite_transform and self.output is not None:
            outputs["composite"] = f"{self.output}Composite.h5"
            outputs["inverse_composite"] = f"{self.output}InverseComposite.h5"
        return outputs


__all__ = ["AntsRegistration"]
