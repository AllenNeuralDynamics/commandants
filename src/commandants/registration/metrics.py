"""Similarity metrics for ``antsRegistration`` -- including point-set metrics.

Each metric renders to the ``Name[...]`` string that follows a ``-m``/``--metric``
flag. Trailing optional parameters that are left as ``None`` are dropped, so the
emitted argument matches idiomatic hand-written ANTs commands.

Two families are modelled, matching ANTs exactly:

* **Intensity metrics** (:class:`MI`, :class:`Mattes`, :class:`CC`,
  :class:`MeanSquares`, :class:`Demons`, :class:`GC`) take a fixed and moving
  *image*::

      Name[fixedImage,movingImage,weight,binsOrRadius,samplingStrategy,samplingPct,useGradientFilter]

* **Point-set metrics** (:class:`PSE`, :class:`ICP`, :class:`JHCT`) take a fixed
  and moving *point set* -- this is the capability ANTsPyX hides and the reason
  you can *constrain a warp with points*::

      PSE[fixedPointSet,movingPointSet,weight,samplingPct,boundaryOnly,pointSetSigma,kNeighborhood]

Point sets are ANTs point-set files (a labelled image, ``.vtk``, or ``.csv`` of
physical-space coordinates -- see :mod:`commandants.io.points`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from ..core.params import bracket, str_resolve

PathLike = Union[str, Path]

# Resolver signature: (value, name, *, must_exist=False) -> str. Defaults to plain
# stringification so a metric renders standalone; on a real run the command builder
# passes one that materializes in-memory images and validates input paths.
Resolve = Callable[..., str]


class Metric:
    """Base class for all metrics."""

    name: str = ""

    def to_arg(self, resolve: Resolve = str_resolve) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_arg()


# --------------------------------------------------------------------------- #
# Intensity metrics
# --------------------------------------------------------------------------- #
@dataclass
class _IntensityMetric(Metric):
    fixed: PathLike
    moving: PathLike
    weight: float = 1.0
    # ``bins`` for MI/Mattes; ``radius`` for CC; NA/ignored for the rest.
    radius_or_bins: Optional[float] = None
    sampling_strategy: Optional[str] = None  # None | "Regular" | "Random"
    sampling_percentage: Optional[float] = None
    use_gradient_filter: Optional[bool] = None

    def to_arg(self, resolve: Resolve = str_resolve) -> str:
        return self.name + bracket(
            resolve(self.fixed, "fixed", must_exist=True),
            resolve(self.moving, "moving", must_exist=True),
            self.weight,
            self.radius_or_bins,
            self.sampling_strategy,
            self.sampling_percentage,
            self.use_gradient_filter,
        )


@dataclass
class MI(_IntensityMetric):
    """Mutual information (histogram); ``radius_or_bins`` is the number of bins."""

    name: str = "MI"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        bins: int = 32,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=bins,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


@dataclass
class Mattes(_IntensityMetric):
    """Mattes mutual information; ``radius_or_bins`` is the number of bins."""

    name: str = "Mattes"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        bins: int = 32,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=bins,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


@dataclass
class CC(_IntensityMetric):
    """Neighborhood cross-correlation; ``radius_or_bins`` is the radius."""

    name: str = "CC"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        radius: int = 4,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=radius,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


@dataclass
class MeanSquares(_IntensityMetric):
    """Mean-squared intensity difference."""

    name: str = "MeanSquares"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        radius: Optional[int] = None,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=radius,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


@dataclass
class Demons(_IntensityMetric):
    """Demons metric."""

    name: str = "Demons"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        radius: Optional[int] = None,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=radius,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


@dataclass
class GC(_IntensityMetric):
    """Global correlation."""

    name: str = "GC"

    def __init__(
        self,
        fixed: PathLike,
        moving: PathLike,
        weight: float = 1.0,
        radius: Optional[int] = None,
        sampling: Optional[str] = None,
        sampling_pct: Optional[float] = None,
        use_gradient_filter: Optional[bool] = None,
    ) -> None:
        super().__init__(
            fixed=fixed,
            moving=moving,
            weight=weight,
            radius_or_bins=radius,
            sampling_strategy=sampling,
            sampling_percentage=sampling_pct,
            use_gradient_filter=use_gradient_filter,
        )


# --------------------------------------------------------------------------- #
# Point-set metrics -- the differentiator vs ANTsPyX
# --------------------------------------------------------------------------- #
@dataclass
class PSE(Metric):
    """Point-Set Expectation metric.

    ``PSE[fixedPointSet,movingPointSet,weight,samplingPct,boundaryOnly,pointSetSigma,kNeighborhood]``

    Use this inside a deformable stage to *constrain the warp with corresponding
    points* while an intensity metric drives the rest of the image.
    """

    name: str = "PSE"

    def __init__(
        self,
        fixed_points: PathLike,
        moving_points: PathLike,
        weight: float = 1.0,
        sampling_pct: Optional[float] = None,
        boundary_only: Optional[bool] = None,
        point_set_sigma: Optional[float] = None,
        k_neighborhood: Optional[int] = None,
    ) -> None:
        self.fixed_points = fixed_points
        self.moving_points = moving_points
        self.weight = weight
        self.sampling_pct = sampling_pct
        self.boundary_only = boundary_only
        self.point_set_sigma = point_set_sigma
        self.k_neighborhood = k_neighborhood

    def to_arg(self, resolve: Resolve = str_resolve) -> str:
        return self.name + bracket(
            resolve(self.fixed_points, "fixed_points", must_exist=True),
            resolve(self.moving_points, "moving_points", must_exist=True),
            self.weight,
            self.sampling_pct,
            self.boundary_only,
            self.point_set_sigma,
            self.k_neighborhood,
        )


@dataclass
class ICP(Metric):
    """Iterative Closest Point metric.

    ``ICP[fixedPointSet,movingPointSet,weight,samplingPct,boundaryOnly]``
    """

    name: str = "ICP"

    def __init__(
        self,
        fixed_points: PathLike,
        moving_points: PathLike,
        weight: float = 1.0,
        sampling_pct: Optional[float] = None,
        boundary_only: Optional[bool] = None,
    ) -> None:
        self.fixed_points = fixed_points
        self.moving_points = moving_points
        self.weight = weight
        self.sampling_pct = sampling_pct
        self.boundary_only = boundary_only

    def to_arg(self, resolve: Resolve = str_resolve) -> str:
        return self.name + bracket(
            resolve(self.fixed_points, "fixed_points", must_exist=True),
            resolve(self.moving_points, "moving_points", must_exist=True),
            self.weight,
            self.sampling_pct,
            self.boundary_only,
        )


@dataclass
class JHCT(Metric):
    """Jensen-Havrda-Charvat-Tsallis point-set metric.

    ``JHCT[fixedPointSet,movingPointSet,weight,samplingPct,boundaryOnly,pointSetSigma,kNeighborhood,alpha,useAnisotropicCovariances]``
    """

    name: str = "JHCT"

    def __init__(
        self,
        fixed_points: PathLike,
        moving_points: PathLike,
        weight: float = 1.0,
        sampling_pct: Optional[float] = None,
        boundary_only: Optional[bool] = None,
        point_set_sigma: Optional[float] = None,
        k_neighborhood: Optional[int] = None,
        alpha: Optional[float] = None,
        use_anisotropic_covariances: Optional[bool] = None,
    ) -> None:
        self.fixed_points = fixed_points
        self.moving_points = moving_points
        self.weight = weight
        self.sampling_pct = sampling_pct
        self.boundary_only = boundary_only
        self.point_set_sigma = point_set_sigma
        self.k_neighborhood = k_neighborhood
        self.alpha = alpha
        self.use_anisotropic_covariances = use_anisotropic_covariances

    def to_arg(self, resolve: Resolve = str_resolve) -> str:
        return self.name + bracket(
            resolve(self.fixed_points, "fixed_points", must_exist=True),
            resolve(self.moving_points, "moving_points", must_exist=True),
            self.weight,
            self.sampling_pct,
            self.boundary_only,
            self.point_set_sigma,
            self.k_neighborhood,
            self.alpha,
            self.use_anisotropic_covariances,
        )


#: Metric classes that operate on point sets rather than images.
POINT_SET_METRICS = (PSE, ICP, JHCT)


def is_point_set_metric(metric: Metric) -> bool:
    """Return True if ``metric`` is a point-set metric."""
    return isinstance(metric, POINT_SET_METRICS)


__all__ = [
    "Metric",
    "MI",
    "Mattes",
    "CC",
    "MeanSquares",
    "Demons",
    "GC",
    "PSE",
    "ICP",
    "JHCT",
    "POINT_SET_METRICS",
    "is_point_set_metric",
]
