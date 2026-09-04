"""Rough resource estimates for an ``AntsRegistration`` run.

**This is a heuristic, not a measurement.** ANTs' true memory and runtime depend
on ITK internals, thread count, image content, and hardware that can't be known
without running the job. The model here estimates *peak* memory as

    overhead + max over stages of (image buffers + displacement-field buffers)

scaled to the finest resolution level of each stage, and a very rough runtime from
total ``voxels * iterations * metric-cost`` work. Treat the memory figure as an
order-of-magnitude planning number and the runtime as a weak hint.

The estimate needs the fixed image's voxel dimensions. Those are read from the
image header (no full load) when a path is given, from ``GetSize()`` for an
in-memory SimpleITK image, or you can pass ``shape=(x, y, z)`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import prod
from typing import Any, List, Optional, Sequence, Tuple

# --- model constants (documented heuristics) ------------------------------- #
#: Baseline ANTs/ITK process overhead added to every estimate.
_BASE_OVERHEAD_BYTES = 150 * 1024 * 1024
#: Full-resolution image buffers ANTs keeps live during a stage
#: (fixed, moving, virtual/warped, one resampled moving).
_BASE_IMAGE_BUFFERS = 4
#: Extra full-size image buffers a metric allocates (by class name).
_METRIC_IMAGE_BUFFERS = {
    "CC": 8,  # neighborhood running-sum images
    "MeanSquares": 2,
    "Demons": 3,
    "GC": 3,
    "MI": 0,  # histogram-based -> negligible at image scale
    "Mattes": 0,
    "PSE": 0,
    "ICP": 0,
    "JHCT": 0,
}
#: How many full displacement-field copies a deformable transform holds.
_FIELD_COPIES = {
    "SyN": 5,
    "GaussianDisplacementField": 5,
    "BSplineSyN": 3,
    "BSpline": 2,
    "Exponential": 4,
}
_LINEAR = {"Translation", "Rigid", "Similarity", "Affine", "CompositeAffine"}
#: Very rough throughput (voxel * iteration * metric-cost units per second,
#: single-threaded). Override via ``rate=`` after calibrating on your hardware.
_DEFAULT_RATE = 6.0e8


def _human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


@dataclass
class StageEstimate:
    index: int
    transform: str
    deformable: bool
    finest_shrink: int
    voxels_effective: int
    memory_bytes: float

    @property
    def memory_human(self) -> str:
        return _human_bytes(self.memory_bytes)


@dataclass
class ResourceEstimate:
    """Result of :meth:`AntsRegistration.estimate_resources`."""

    image_shape: Tuple[int, ...]
    dimensionality: int
    n_voxels: int
    real_bytes: int  # 4 (float) or 8 (double)
    peak_memory_bytes: float
    per_stage: List[StageEstimate] = field(default_factory=list)
    work: float = 0.0  # voxel*iteration*metric-cost units (hardware-independent)
    est_runtime_seconds: Optional[float] = None
    threads: int = 1

    @property
    def peak_memory_human(self) -> str:
        return _human_bytes(self.peak_memory_bytes)

    @property
    def est_runtime_human(self) -> Optional[str]:
        return None if self.est_runtime_seconds is None else _human_time(self.est_runtime_seconds)

    def summary(self) -> str:
        lines = [
            "Resource estimate (HEURISTIC -- order-of-magnitude only)",
            f"  image           : {'x'.join(map(str, self.image_shape))} "
            f"({self.n_voxels:,} voxels, {self.real_bytes}-byte real type)",
            f"  peak memory     : ~{self.peak_memory_human}",
        ]
        if self.est_runtime_human:
            lines.append(f"  est. runtime    : ~{self.est_runtime_human} (@{self.threads} thread(s); very rough)")
        lines.append("  per-stage peak memory:")
        for s in self.per_stage:
            kind = "deformable" if s.deformable else "linear"
            lines.append(
                f"    [{s.index}] {s.transform:<26} {kind:<10} finest shrink {s.finest_shrink} -> ~{s.memory_human}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.summary()


# --------------------------------------------------------------------------- #
# Shape resolution
# --------------------------------------------------------------------------- #
def _shape_from_reference(ref: Any) -> Tuple[int, ...]:
    """Return the voxel dimensions of a fixed-image reference.

    Accepts a SimpleITK image, a path (header read only), or a shape sequence.
    """
    # Explicit shape sequence.
    if isinstance(ref, (tuple, list)) and all(isinstance(v, int) for v in ref):
        return tuple(ref)

    # In-memory SimpleITK image.
    from .io.materialize import is_sitk_image

    if is_sitk_image(ref):
        return tuple(int(v) for v in ref.GetSize())

    # A path: read just the header.
    path = os.fspath(ref) if hasattr(ref, "__fspath__") else str(ref)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot read image dimensions: {path!r} does not exist. "
            "Pass shape=(x, y, z) to estimate without the file present."
        )
    try:
        import SimpleITK as sitk  # noqa: PLC0415

        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        return tuple(int(v) for v in reader.GetSize())
    except ImportError:
        pass
    try:
        import nibabel as nib  # noqa: PLC0415

        return tuple(int(v) for v in nib.load(path).shape)
    except ImportError as exc:
        raise ImportError(
            "Reading image dimensions from a file needs SimpleITK (the [io] extra) "
            "or nibabel. Install one, or pass shape=(x, y, z)."
        ) from exc


def _infer_reference(reg: Any) -> Any:
    """Find a fixed-image reference from the registration's init or metrics."""
    for item in getattr(reg, "_initial_transforms", []):
        if item and item[0] == "images":
            return item[1]  # ("images", fixed, moving, feature)
    for stage in reg.stages:
        for metric in stage.metrics:
            fixed = getattr(metric, "fixed", None)
            if fixed is not None:
                return fixed
    return None


# --------------------------------------------------------------------------- #
# Transform / metric model helpers
# --------------------------------------------------------------------------- #
def _field_copies(transform: Any) -> Tuple[int, bool]:
    """Return (displacement-field copies, is_deformable) for a transform."""
    name = type(transform).__name__
    if name in _LINEAR:
        return 0, False
    if name == "TimeVaryingVelocityField":
        n_time = getattr(transform, "number_of_time_indices", 4) or 4
        return 2 * int(n_time), True
    return _FIELD_COPIES.get(name, 4), True


def _metric_cost(metric: Any, dim: int) -> float:
    name = type(metric).__name__
    if name == "CC":
        radius = getattr(metric, "radius_or_bins", None) or 4
        return float((2 * int(radius) + 1) ** dim)
    if name in ("MI", "Mattes"):
        return 1.5
    if name in ("PSE", "ICP", "JHCT"):
        return 0.2  # point sets are cheap relative to dense image metrics
    return 1.0


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def estimate_registration(
    reg: Any,
    fixed: Any = None,
    shape: Optional[Sequence[int]] = None,
    threads: int = 1,
    rate: float = _DEFAULT_RATE,
) -> ResourceEstimate:
    """Estimate peak memory (and rough runtime) for a built ``AntsRegistration``.

    See the module docstring for the model and caveats.
    """
    if not reg.stages:
        raise ValueError("Add at least one stage before estimating resources.")

    if shape is not None:
        img_shape = tuple(int(v) for v in shape)
    else:
        ref = fixed if fixed is not None else _infer_reference(reg)
        if ref is None:
            raise ValueError("Could not determine the fixed image. Pass fixed=<path/image> or shape=(x, y, z).")
        img_shape = _shape_from_reference(ref)

    dim = reg.dimensionality
    n_voxels = int(prod(img_shape))
    real_bytes = 4 if reg.use_float else 8  # ANTs default real type is double

    per_stage: List[StageEstimate] = []
    peak_stage_bytes = 0.0
    work = 0.0

    for i, stage in enumerate(reg.stages):
        finest = min(stage.shrink_factors)
        n_eff = n_voxels / (finest**dim)

        metric_buffers = sum(_METRIC_IMAGE_BUFFERS.get(type(m).__name__, 1) for m in stage.metrics)
        image_bytes = (_BASE_IMAGE_BUFFERS + metric_buffers) * n_eff * real_bytes

        copies, deformable = _field_copies(stage.transform)
        field_bytes = copies * n_eff * dim * real_bytes

        stage_bytes = image_bytes + field_bytes
        peak_stage_bytes = max(peak_stage_bytes, stage_bytes)

        per_stage.append(
            StageEstimate(
                index=i,
                transform=type(stage.transform).__name__,
                deformable=deformable,
                finest_shrink=finest,
                voxels_effective=int(n_eff),
                memory_bytes=stage_bytes,
            )
        )

        # Runtime work: sum over levels of iterations * effective voxels * cost.
        metric_cost = sum(_metric_cost(m, dim) for m in stage.metrics) or 1.0
        transform_factor = 3.0 if deformable else 1.0
        for level, iters in enumerate(stage.convergence.iterations):
            s = stage.shrink_factors[level]
            work += iters * (n_voxels / (s**dim)) * metric_cost * transform_factor

    peak = _BASE_OVERHEAD_BYTES + peak_stage_bytes
    threads = max(1, int(threads))
    runtime = (work / rate) / threads if rate else None

    return ResourceEstimate(
        image_shape=img_shape,
        dimensionality=dim,
        n_voxels=n_voxels,
        real_bytes=real_bytes,
        peak_memory_bytes=peak,
        per_stage=per_stage,
        work=work,
        est_runtime_seconds=runtime,
        threads=threads,
    )


__all__ = ["estimate_registration", "ResourceEstimate", "StageEstimate"]
