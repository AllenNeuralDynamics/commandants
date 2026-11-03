"""Preset builders for the Allen STPT/CCF driver.

Each returns a ready, un-run :class:`~commandants.stpt.StptRegistration`, following the
same convention as the ANTs and elastix presets: ``preset(fixed, moving, destination,
**kwargs)``. As with elastix, the third positional argument is an output *directory*.

``ccf_global`` and ``ccf_stpt`` are named to match their ANTs and elastix counterparts, so
``benchmark.compare(preset="ccf_stpt")`` resolves on all three backends.
"""

from __future__ import annotations

from typing import Any

from .registration import StptRegistration


def rigid(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> StptRegistration:
    """Rigid (versor) stage only, driven by mutual information."""
    return StptRegistration(fixed, moving, out_dir, stages=("rigid",), **kwargs)


def affine(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> StptRegistration:
    """Rigid then affine, both driven by mutual information."""
    return StptRegistration(fixed, moving, out_dir, stages=("rigid", "affine"), **kwargs)


def ccf_global(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> StptRegistration:
    """The upstream *global* alignment: rigid + affine mutual information."""
    return affine(fixed, moving, out_dir, **kwargs)


def ccf_stpt(fixed: Any, moving: Any, out_dir: Any, **kwargs) -> StptRegistration:
    """Full Allen schedule: rigid -> affine (MI) -> discrete symmetric MRF deformable.

    This is the real algorithm, as opposed to the ANTs and elastix ``ccf_stpt`` presets,
    which approximate the deformable stage with continuous optimizers.
    """
    return StptRegistration(fixed, moving, out_dir, stages=("rigid", "affine", "deformable"), **kwargs)


__all__ = ["rigid", "affine", "ccf_global", "ccf_stpt"]
