"""Wrappers for the Allen STPT/CCF registration algorithm (the third backend).

This runs the **real** Allen algorithm, as opposed to the ANTs and elastix ``ccf_stpt``
presets, which only approximate it: their deformable stages use continuous optimizers,
while the Allen local step is a discrete MRF solved symmetrically (forward + backward,
composed into an invertible field)::

    from commandants import stpt

    reg = stpt.presets.ccf_stpt("fixed.nii.gz", "moving.nii.gz", "out_dir")
    print(reg.to_shell())
    result = reg.run()
    warped = result.load("warped")     # -> SimpleITK.Image

It drives ``stpt_register``, a minimal ITK-only driver built from a separate,
Allen-licensed repository (the upstream code is BSD-2 **plus a non-commercial clause**,
so it is not vendored into this MIT package -- commandants only invokes the binary).
Resolve it with ``$STPTPATH`` or ``PATH``; there is no download, so no ``install-stpt``.

Why a driver instead of the upstream ``idpProjection*`` modules: those cannot register two
3D volumes at all. They assemble a volume slice-by-slice from one 2D file per ``<sub-image>``
and pin the z-extraction to slice 0, so a 3D input would have to be exploded into N 2D
slices -- on top of requiring a ``<specimen>`` XML root (an unguarded null dereference
otherwise), a structured ``model_directory``, and JPEG2000/Kakadu.
"""

from . import presets
from ._base import StptToolCommand
from .registration import STAGES, StptRegistration

__all__ = ["StptRegistration", "StptToolCommand", "STAGES", "presets"]
