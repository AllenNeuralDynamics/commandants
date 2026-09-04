"""Curated default elastix parameter maps + preset builders.

These mirror the well-known elastix default parameter maps (Mattes MI +
AdaptiveStochasticGradientDescent + a random-coordinate sampler, 4 resolutions).
We ship them as data rather than calling ``sitk.GetDefaultParameterMap`` because the
standard pip ``SimpleITK`` wheel does not include elastix. Every value is
overridable -- get a map with :func:`parameter_map`, tweak it, and pass it to
:class:`~commandants.elastix.Elastix`.

Each preset builder returns a ready, un-run :class:`Elastix` (mirroring the ANTs
``commandants.presets`` contract), so you keep the inspect/estimate/run workflow.
"""

from __future__ import annotations

from typing import Any

from .parameters import ParameterMap
from .registration import Elastix

# Settings common to every default map (transform added per-preset).
_COMMON = {
    "FixedImagePyramid": "FixedSmoothingImagePyramid",
    "MovingImagePyramid": "MovingSmoothingImagePyramid",
    "Interpolator": "BSplineInterpolator",
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "Resampler": "DefaultResampler",
    "Registration": "MultiResolutionRegistration",
    "Optimizer": "AdaptiveStochasticGradientDescent",
    "Metric": "AdvancedMattesMutualInformation",
    "ImageSampler": "RandomCoordinate",
    "NumberOfSpatialSamples": 2048,
    "NewSamplesEveryIteration": True,
    "NumberOfResolutions": 4,
    "MaximumNumberOfIterations": 256,
    "NumberOfHistogramBins": 32,
    "BSplineInterpolationOrder": 1,
    "FinalBSplineInterpolationOrder": 3,
    "DefaultPixelValue": 0,
    "WriteResultImage": True,
    "ResultImagePixelType": "float",
    "ResultImageFormat": "nii",
}

_TRANSFORM_EXTRAS = {
    "translation": {
        "Transform": "TranslationTransform",
        "AutomaticTransformInitialization": True,
    },
    "rigid": {
        "Transform": "EulerTransform",
        "AutomaticScalesEstimation": True,
        "AutomaticTransformInitialization": True,
    },
    "affine": {
        "Transform": "AffineTransform",
        "AutomaticScalesEstimation": True,
        "AutomaticTransformInitialization": True,
    },
    "bspline": {
        "Transform": "BSplineTransform",
        "FinalGridSpacingInPhysicalUnits": 16,
    },
}


def parameter_map(kind: str) -> ParameterMap:
    """Return a fresh default parameter map for ``kind`` (translation/rigid/affine/bspline)."""
    k = kind.lower()
    if k not in _TRANSFORM_EXTRAS:
        raise ValueError(
            f"Unknown elastix preset {kind!r}; use one of {sorted(_TRANSFORM_EXTRAS)}."
        )
    pm = ParameterMap()
    pm.update(_COMMON)
    pm.update(_TRANSFORM_EXTRAS[k])
    return pm


def translation(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """Translation-only registration (single parameter map)."""
    return Elastix(fixed, moving, out_dir, parameter_map("translation"), **kwargs)


def rigid(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """Rigid (Euler) registration with automatic center initialization."""
    return Elastix(fixed, moving, out_dir, parameter_map("rigid"), **kwargs)


def affine(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """Affine registration."""
    return Elastix(fixed, moving, out_dir, parameter_map("affine"), **kwargs)


def bspline(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """B-spline (deformable) registration."""
    return Elastix(fixed, moving, out_dir, parameter_map("bspline"), **kwargs)


def affine_bspline(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """Two-stage affine -> B-spline (two parameter maps, run sequentially)."""
    return Elastix(
        fixed, moving, out_dir,
        [parameter_map("affine"), parameter_map("bspline")],
        **kwargs,
    )


__all__ = [
    "parameter_map",
    "translation",
    "rigid",
    "affine",
    "bspline",
    "affine_bspline",
]
