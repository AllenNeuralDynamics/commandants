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
        raise ValueError(f"Unknown elastix preset {kind!r}; use one of {sorted(_TRANSFORM_EXTRAS)}.")
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
        fixed,
        moving,
        out_dir,
        [parameter_map("affine"), parameter_map("bspline")],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Allen CCF / STPT template-construction schedule
# ---------------------------------------------------------------------------
# Mimics the AllenInstitute/stpt_registration pipeline (Oh 2014 / Kuan 2015; the
# method used to build the Allen Mouse CCF average template): a *global* stage of
# rigid + affine Mattes mutual information (64 histogram bins, center-of-gravity
# init, coarse-to-fine from shrink factor 8), followed by a *local* 3rd-order
# B-spline driven by normalized cross-correlation over a 4-level coarse-to-fine
# grid (~30 -> 4 voxel spacing) with random sampling.
#
# CAVEAT: the original deformable step uses a discrete MRF / graph-labeling
# optimizer run symmetrically (forward + backward, composed -> invertible). elastix
# uses continuous stochastic gradient descent and is not symmetric by construction,
# so the B-spline stage matches the original *in spirit* (transform order, metric,
# grid schedule, sampling), not in optimizer mechanism. Every value is overridable.

_CCF_COMMON = {
    "Registration": "MultiResolutionRegistration",
    "FixedImagePyramid": "FixedSmoothingImagePyramid",
    "MovingImagePyramid": "MovingSmoothingImagePyramid",
    "Interpolator": "BSplineInterpolator",
    "BSplineInterpolationOrder": 1,  # linear during optimization
    "ResampleInterpolator": "FinalBSplineInterpolator",
    "FinalBSplineInterpolationOrder": 3,
    "Resampler": "DefaultResampler",
    "Optimizer": "AdaptiveStochasticGradientDescent",  # original used ITK RegularStepGradientDescent
    "ImageSampler": "RandomCoordinate",
    "NewSamplesEveryIteration": True,
    "FixedInternalImagePixelType": "float",
    "MovingInternalImagePixelType": "float",
    "WriteResultImage": True,
    "ResultImagePixelType": "float",
    "ResultImageFormat": "nii",
}

# Coarse-to-fine pyramid for the linear stages: factor 8 -> 1 over 4 levels, per axis.
_CCF_LINEAR_SCHEDULE = [8, 8, 8, 4, 4, 4, 2, 2, 2, 1, 1, 1]


def _ccf_linear_map(transform: str) -> ParameterMap:
    pm = ParameterMap()
    pm.update(_CCF_COMMON)
    pm.update(
        {
            "Transform": transform,
            "AutomaticScalesEstimation": True,
            "Metric": "AdvancedMattesMutualInformation",
            "NumberOfHistogramBins": 64,  # matches the STPT pipeline
            "NumberOfResolutions": 4,
            "ImagePyramidSchedule": _CCF_LINEAR_SCHEDULE,
            # NOTE: the original ITK metric sampled 250000 points once; elastix re-samples
            # every iteration, so a few thousand is equivalent -- do NOT use 250000 here.
            "NumberOfSpatialSamples": 3000,
            "MaximumNumberOfIterations": 500,
            "HowToCombineTransforms": "Compose",
        }
    )
    if transform == "EulerTransform":
        pm.update(
            {
                "AutomaticTransformInitialization": True,
                "AutomaticTransformInitializationMethod": "CenterOfGravity",  # original MomentsOn()
            }
        )
    return pm


def _ccf_bspline_map() -> ParameterMap:
    pm = ParameterMap()
    pm.update(_CCF_COMMON)
    pm.update(
        {
            "Transform": "BSplineTransform",
            "BSplineTransformSplineOrder": 3,  # original B-spline order 3
            "Metric": "AdvancedNormalizedCorrelation",  # original cross-correlation
            "NumberOfResolutions": 4,  # 4-level coarse-to-fine
            "FinalGridSpacingInVoxels": 4,  # finest grid ~4 voxels
            "GridSpacingSchedule": [7.5, 3.75, 2.0, 1.0],  # ~30, 15, 8, 4 voxel grids
            "NumberOfSpatialSamples": 1024,  # original samplenum
            "MaximumNumberOfIterations": 1000,
            "HowToCombineTransforms": "Compose",
        }
    )
    return pm


def ccf_parameter_maps() -> list[ParameterMap]:
    """The three STPT/CCF stages (rigid, affine, B-spline) as editable ParameterMaps."""
    return [
        _ccf_linear_map("EulerTransform"),
        _ccf_linear_map("AffineTransform"),
        _ccf_bspline_map(),
    ]


def ccf_global(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """STPT/CCF *global* alignment only: rigid -> affine Mattes MI (no deformable)."""
    maps = ccf_parameter_maps()[:2]
    return Elastix(fixed, moving, out_dir, maps, **kwargs)


def ccf_stpt(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> Elastix:
    """Full STPT/CCF schedule: rigid -> affine (MI) -> B-spline (NCC).

    Mimics AllenInstitute/stpt_registration template construction. See the module
    notes above for what transfers exactly and the deformable-optimizer caveat.
    """
    return Elastix(fixed, moving, out_dir, ccf_parameter_maps(), **kwargs)


__all__ = [
    "parameter_map",
    "translation",
    "rigid",
    "affine",
    "bspline",
    "affine_bspline",
    "ccf_parameter_maps",
    "ccf_global",
    "ccf_stpt",
]
