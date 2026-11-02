"""Wrappers for the Allen STPT/CCF registration binaries (the third backend).

This is the backend that runs the **real** Allen algorithm, as opposed to the ANTs and
elastix ``ccf_stpt`` presets, which only approximate it (their deformable stages use
continuous optimizers, while the Allen local step is a discrete MRF solved symmetrically).

Status: **scaffolding only**. Binary resolution (``$STPTPATH``), the shared command base,
and the benchmark hook are in place; the command wrappers land once we settle whether to
drive the three ``idp*`` pipeline modules directly or a thin two-volume driver built from
the same registration classes.
"""

from ._base import StptToolCommand

__all__ = ["StptToolCommand"]
