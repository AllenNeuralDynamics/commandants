"""commandants -- a thin, transparent Python wrapper for the ANTs command line.

commandANTs wraps the ANTs *binaries* directly (via subprocess) and exposes the
full CLI surface. Nothing is hidden: every builder has a ``.build_command()`` for
inspection, a ``.extra_args(...)`` escape hatch for arbitrary flags, and models
the real ``antsRegistration`` grammar -- including multivariate and point-set
metrics (``PSE``/``ICP``/``JHCT``) that let you *constrain a warp with points*.

Quickstart
----------
>>> from commandants import AntsRegistration
>>> from commandants import Rigid, SyN, MI, CC, PSE, Convergence
>>> reg = AntsRegistration(3, output="out_", verbose=True)
>>> _ = reg.initialize_from_images("fixed.nii.gz", "moving.nii.gz")
>>> _ = reg.add_stage(Rigid(0.1), MI("fixed.nii.gz", "moving.nii.gz", bins=32),
...                   Convergence([1000, 500, 250, 100]), [8, 4, 2, 1], [3, 2, 1, 0])
>>> print(reg.to_shell())   # inspect the exact command, no ANTs required
"""

from __future__ import annotations

from .core import (
    AntsCommand,
    AntsNotFoundError,
    AntsRuntimeError,
    CommandantsError,
    CompletedAnts,
    is_available,
    resolve_binary,
    version,
)
from .estimate import ResourceEstimate, estimate_registration
from .exit_codes import ExitCodeExplanation, explain_exit_code
from .install import (
    install_ants,
    installed_versions,
    managed_bin_dir,
    uninstall_ants,
)
from . import presets
from .io import TempWorkspace, is_sitk_image, read_points, write_points
from .presets import affine, rigid, similarity, syn, syn_only, synonly, translation
from .preprocessing import (
    ImageMath,
    N4BiasFieldCorrection,
    ResampleImage,
    ThresholdImage,
)
from .registration import (
    CC,
    GC,
    ICP,
    JHCT,
    MI,
    PSE,
    Affine,
    AntsApplyTransforms,
    AntsApplyTransformsToPoints,
    AntsRegistration,
    BSpline,
    BSplineSyN,
    CompositeAffine,
    Convergence,
    Demons,
    Exponential,
    GaussianDisplacementField,
    Mattes,
    MeanSquares,
    Metric,
    RawTransform,
    Rigid,
    Similarity,
    Stage,
    SyN,
    TimeVaryingVelocityField,
    Transform,
    Translation,
    is_point_set_metric,
)

# Version is static and bumped by commitizen (see [tool.commitizen] in
# pyproject.toml, version_files points here). Keep the assignment on one line.
__version__ = "0.2.0"

__all__ = [
    "__version__",
    # core
    "AntsCommand",
    "CompletedAnts",
    "CommandantsError",
    "AntsNotFoundError",
    "AntsRuntimeError",
    "resolve_binary",
    "is_available",
    "version",
    # ANTs binary provisioning
    "install_ants",
    "uninstall_ants",
    "managed_bin_dir",
    "installed_versions",
    # resource estimation
    "estimate_registration",
    "ResourceEstimate",
    # exit-code interpretation
    "explain_exit_code",
    "ExitCodeExplanation",
    # preset registration builders (ANTsPyX-style)
    "presets",
    "translation",
    "rigid",
    "similarity",
    "affine",
    "syn",
    "syn_only",
    "synonly",
    # in-memory images + point I/O
    "TempWorkspace",
    "is_sitk_image",
    "read_points",
    "write_points",
    # registration builders
    "AntsRegistration",
    "AntsApplyTransforms",
    "AntsApplyTransformsToPoints",
    "Stage",
    "Convergence",
    # metrics
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
    "is_point_set_metric",
    # transforms
    "Transform",
    "Translation",
    "Rigid",
    "Similarity",
    "Affine",
    "CompositeAffine",
    "SyN",
    "GaussianDisplacementField",
    "BSplineSyN",
    "BSpline",
    "TimeVaryingVelocityField",
    "Exponential",
    "RawTransform",
    # preprocessing
    "N4BiasFieldCorrection",
    "ThresholdImage",
    "ImageMath",
    "ResampleImage",
]
