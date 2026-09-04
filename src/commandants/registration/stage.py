"""A registration *stage* -- one transform plus its metrics and schedule.

This is the unit that ANTsPyX flattens away. In real ``antsRegistration`` a stage
is a repeated group of::

    -t <transform> -m <metric> [-m <metric> ...] -c <convergence> -f <shrink> -s <smoothing>

where a single stage may carry **multiple metrics** (multivariate registration or
an intensity metric combined with a point-set constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from ..core.params import bracket, str_resolve, xjoin
from .metrics import Metric
from .transforms import Transform


@dataclass
class Convergence:
    """Convergence schedule -> ``[iters,threshold,window]``.

    ``iterations`` is the per-resolution-level iteration vector, e.g.
    ``[1000, 500, 250, 100]``.
    """

    iterations: Sequence[int]
    threshold: Optional[float] = 1e-6
    window: Optional[int] = 10

    def to_arg(self) -> str:
        return bracket(xjoin(self.iterations), self.threshold, self.window)

    @property
    def n_levels(self) -> int:
        return len(self.iterations)


@dataclass
class Stage:
    """One registration stage.

    Parameters
    ----------
    transform:
        The transform model for this stage.
    metrics:
        One or more metrics. Multiple metrics are emitted as repeated ``-m``
        flags (multivariate / point-constrained registration).
    convergence:
        Iteration schedule and stopping criteria.
    shrink_factors, smoothing_sigmas:
        Multi-resolution schedules, one entry per level.
    smoothing_units:
        ``"vox"`` (voxels) or ``"mm"``; appended to the smoothing vector as ANTs
        requires (e.g. ``3x2x1x0vox``). ``None`` emits no suffix.
    """

    transform: Transform
    metrics: List[Metric]
    convergence: Convergence
    shrink_factors: Sequence[int]
    smoothing_sigmas: Sequence[float]
    smoothing_units: Optional[str] = "vox"
    #: Optional metric masks for *this stage only* (path or SimpleITK image).
    fixed_mask: Optional[Any] = None
    moving_mask: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("A stage requires at least one metric.")
        if self.smoothing_units not in (None, "vox", "mm"):
            raise ValueError(f"smoothing_units must be 'vox', 'mm', or None; got {self.smoothing_units!r}")
        n = self.convergence.n_levels
        if len(self.shrink_factors) != n:
            raise ValueError(
                f"shrink_factors has {len(self.shrink_factors)} levels but convergence has {n}; they must match."
            )
        if len(self.smoothing_sigmas) != n:
            raise ValueError(
                f"smoothing_sigmas has {len(self.smoothing_sigmas)} levels but convergence has {n}; they must match."
            )

    def _smoothing_arg(self) -> str:
        vec = xjoin(self.smoothing_sigmas)
        return f"{vec}{self.smoothing_units}" if self.smoothing_units else vec

    def to_args(self, resolve: Callable[..., str] = str_resolve) -> List[str]:
        """Emit this stage's argv tokens in ANTs' expected order.

        ``resolve`` is applied to image-bearing metric arguments so in-memory
        SimpleITK images become temp-file paths at run time.
        """
        tokens: List[str] = ["--transform", self.transform.to_arg()]
        for metric in self.metrics:
            tokens += ["--metric", metric.to_arg(resolve)]
        tokens += ["--convergence", self.convergence.to_arg()]
        tokens += ["--shrink-factors", xjoin(self.shrink_factors)]
        tokens += ["--smoothing-sigmas", self._smoothing_arg()]
        return tokens


__all__ = ["Convergence", "Stage"]
