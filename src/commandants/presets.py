"""Pre-baked registration builders, in the spirit of ANTsPyX's ``type_of_transform``.

Each function returns a ready-to-run :class:`AntsRegistration` (it does **not**
run it) using antsRegistrationSyN-standard schedules, so you keep the
inspect/estimate/run workflow::

    reg = presets.syn("fixed.nii.gz", "moving.nii.gz", "out_")
    print(reg.to_shell())          # inspect
    reg.estimate_resources(...)    # size it
    result = reg.run(stream=True)  # go

Unlike ANTsPyX these hide nothing: every schedule/metric/step is an overridable
argument, and the exact stages are documented per function. The linear presets
(and ``syn``) include a **center-of-mass initialization** by default -- the step
whose absence yields an identity transform. ``syn_only`` does **not** initialize
(matching ANTsPyX ``SyNOnly``); pass ``init=...`` or ``initial_transform=...`` if
your images aren't already aligned.

Defaults chosen to mirror ANTsPyX: single precision (``use_float=True``), Mattes
mutual information for the metric, light intensity winsorization.
"""

from __future__ import annotations

from typing import Union

from .registration import (
    CC,
    MI,
    Affine,
    AntsRegistration,
    BSplineSyN,
    Convergence,
    Mattes,
    MeanSquares,
    Rigid,
    Similarity,
    SyN,
    Translation,
)

PathLike = Union[str, "os.PathLike[str]"]  # noqa: F821

# Linear defaults mirror ANTsPyX's pure Rigid/Affine command (its
# aff_iterations / aff_shrink_factors / aff_smoothing_sigmas and Rigid[0.25]).
# These give the optimizer enough "reach" (step x iterations, down to full
# resolution) to close a large translation offset -- a weaker schedule undershoots
# and lands the result off to the side.
_LINEAR_GRAD_STEP = 0.25
_LINEAR_SAMPLING_PCT = 0.2
_LINEAR_ITERATIONS = (2100, 1200, 1200, 10)
_LINEAR_SHRINK = (6, 4, 2, 1)
_LINEAR_SMOOTH = (3, 2, 1, 0)
# SyN (deformable) stage defaults.
_SYN_ITERATIONS = (100, 70, 50, 20)
_SYN_SHRINK = (8, 4, 2, 1)
_SYN_SMOOTH = (3, 2, 1, 0)


def _make_metric(kind, fixed, moving, *, bins=32, radius=4, sampling="Regular", sampling_pct=0.25, weight=1.0):
    k = str(kind).lower()
    if k in ("mattes",):
        return Mattes(fixed, moving, weight=weight, bins=bins, sampling=sampling, sampling_pct=sampling_pct)
    if k in ("mi", "mutualinformation"):
        return MI(fixed, moving, weight=weight, bins=bins, sampling=sampling, sampling_pct=sampling_pct)
    if k in ("cc", "neighborhoodcc"):
        return CC(fixed, moving, weight=weight, radius=radius, sampling=sampling, sampling_pct=sampling_pct)
    if k in ("meansquares", "ms", "msq"):
        return MeanSquares(fixed, moving, weight=weight, sampling=sampling, sampling_pct=sampling_pct)
    raise ValueError(f"Unknown metric {kind!r}; use 'mattes', 'mi', 'cc', or 'meansquares'.")


def _new_reg(
    dim,
    output,
    warped_output,
    inverse_warped_output,
    use_float,
    winsorize,
    use_histogram_matching,
    collapse_output_transforms,
    write_composite_transform,
    random_seed,
    verbose,
    ants_path,
):
    return AntsRegistration(
        dimensionality=dim,
        output=output,
        warped_output=warped_output,
        inverse_warped_output=inverse_warped_output,
        use_float=use_float,
        winsorize=winsorize,
        use_histogram_matching=use_histogram_matching,
        collapse_output_transforms=collapse_output_transforms,
        write_composite_transform=write_composite_transform,
        random_seed=random_seed,
        verbose=verbose,
        ants_path=ants_path,
    )


def _linear_preset(
    fixed,
    moving,
    output,
    transforms,
    *,
    dim=3,
    warped_output=None,
    inverse_warped_output=None,
    init="center-of-mass",
    initial_transform=None,
    metric="mattes",
    bins=32,
    sampling="Regular",
    sampling_pct=_LINEAR_SAMPLING_PCT,
    grad_step=_LINEAR_GRAD_STEP,
    iterations=_LINEAR_ITERATIONS,
    shrink_factors=_LINEAR_SHRINK,
    smoothing_sigmas=_LINEAR_SMOOTH,
    smoothing_units="vox",
    use_float=True,
    winsorize=(0.005, 0.995),
    use_histogram_matching=False,
    mask=None,
    moving_mask=None,
    collapse_output_transforms=None,
    write_composite_transform=False,
    random_seed=None,
    verbose=False,
    ants_path=None,
) -> AntsRegistration:
    reg = _new_reg(
        dim,
        output,
        warped_output,
        inverse_warped_output,
        use_float,
        winsorize,
        use_histogram_matching,
        collapse_output_transforms,
        write_composite_transform,
        random_seed,
        verbose,
        ants_path,
    )
    # An explicit initial transform takes precedence over center-of-mass init.
    if initial_transform is not None:
        reg.add_initial_moving_transform(initial_transform)
    elif init is not None:
        reg.initialize_from_images(fixed, moving, init=init)
    if mask is not None or moving_mask is not None:
        reg.set_masks(mask, moving_mask)
    for transform_cls in transforms:
        reg.add_stage(
            transform=transform_cls(gradient_step=grad_step),
            metrics=_make_metric(metric, fixed, moving, bins=bins, sampling=sampling, sampling_pct=sampling_pct),
            convergence=Convergence(iterations),
            shrink_factors=list(shrink_factors),
            smoothing_sigmas=list(smoothing_sigmas),
            smoothing_units=smoothing_units,
        )
    return reg


def translation(fixed, moving, output, **kwargs) -> AntsRegistration:
    """Init + a single Translation stage."""
    return _linear_preset(fixed, moving, output, [Translation], **kwargs)


def rigid(fixed, moving, output, **kwargs) -> AntsRegistration:
    """Init + a single Rigid stage."""
    return _linear_preset(fixed, moving, output, [Rigid], **kwargs)


def similarity(fixed, moving, output, **kwargs) -> AntsRegistration:
    """Init + Rigid + Similarity stages."""
    return _linear_preset(fixed, moving, output, [Rigid, Similarity], **kwargs)


def affine(fixed, moving, output, **kwargs) -> AntsRegistration:
    """Init + Rigid + Affine stages."""
    return _linear_preset(fixed, moving, output, [Rigid, Affine], **kwargs)


def syn(
    fixed,
    moving,
    output,
    *,
    dim=3,
    warped_output=None,
    inverse_warped_output=None,
    init="center-of-mass",
    initial_transform=None,
    # linear part (Rigid + Affine)
    aff_metric="mattes",
    aff_bins=32,
    aff_sampling="Regular",
    aff_sampling_pct=_LINEAR_SAMPLING_PCT,
    aff_grad_step=_LINEAR_GRAD_STEP,
    aff_iterations=_LINEAR_ITERATIONS,
    aff_shrink_factors=_LINEAR_SHRINK,
    aff_smoothing_sigmas=_LINEAR_SMOOTH,
    # SyN part
    syn_metric="mattes",
    syn_bins=32,
    syn_radius=4,
    syn_sampling=None,
    syn_sampling_pct=None,
    grad_step=0.1,
    flow_sigma=3.0,
    total_sigma=0.0,
    syn_iterations=_SYN_ITERATIONS,
    syn_shrink_factors=_SYN_SHRINK,
    syn_smoothing_sigmas=_SYN_SMOOTH,
    smoothing_units="vox",
    use_float=True,
    winsorize=(0.005, 0.995),
    use_histogram_matching=False,
    mask=None,
    moving_mask=None,
    collapse_output_transforms=None,
    write_composite_transform=False,
    random_seed=None,
    verbose=False,
    ants_path=None,
) -> AntsRegistration:
    """Full pipeline: init + Rigid + Affine + SyN (== antsRegistrationSyN / ANTsPyX SyNRA)."""
    reg = _new_reg(
        dim,
        output,
        warped_output,
        inverse_warped_output,
        use_float,
        winsorize,
        use_histogram_matching,
        collapse_output_transforms,
        write_composite_transform,
        random_seed,
        verbose,
        ants_path,
    )
    if initial_transform is not None:
        reg.add_initial_moving_transform(initial_transform)
    elif init is not None:
        reg.initialize_from_images(fixed, moving, init=init)
    if mask is not None or moving_mask is not None:
        reg.set_masks(mask, moving_mask)
    for transform_cls in (Rigid, Affine):
        reg.add_stage(
            transform=transform_cls(gradient_step=aff_grad_step),
            metrics=_make_metric(
                aff_metric, fixed, moving, bins=aff_bins, sampling=aff_sampling, sampling_pct=aff_sampling_pct
            ),
            convergence=Convergence(aff_iterations),
            shrink_factors=list(aff_shrink_factors),
            smoothing_sigmas=list(aff_smoothing_sigmas),
            smoothing_units=smoothing_units,
        )
    _add_syn_stage(
        reg,
        fixed,
        moving,
        syn_metric,
        syn_bins,
        syn_radius,
        syn_sampling,
        syn_sampling_pct,
        grad_step,
        flow_sigma,
        total_sigma,
        syn_iterations,
        syn_shrink_factors,
        syn_smoothing_sigmas,
        smoothing_units,
    )
    return reg


def syn_only(
    fixed,
    moving,
    output,
    *,
    dim=3,
    warped_output=None,
    inverse_warped_output=None,
    init=None,
    initial_transform=None,
    syn_metric="mattes",
    syn_bins=32,
    syn_radius=4,
    syn_sampling=None,
    syn_sampling_pct=None,
    grad_step=0.1,
    flow_sigma=3.0,
    total_sigma=0.0,
    syn_iterations=_SYN_ITERATIONS,
    syn_shrink_factors=_SYN_SHRINK,
    syn_smoothing_sigmas=_SYN_SMOOTH,
    smoothing_units="vox",
    use_float=True,
    winsorize=(0.005, 0.995),
    use_histogram_matching=False,
    mask=None,
    moving_mask=None,
    collapse_output_transforms=None,
    write_composite_transform=False,
    random_seed=None,
    verbose=False,
    ants_path=None,
) -> AntsRegistration:
    """SyN only -- no linear stages.

    Like ANTsPyX ``SyNOnly``, this does **not** center-of-mass initialize by
    default: it assumes the images are already aligned (e.g. you ran ``affine``
    first). Pass ``initial_transform=<path>`` to prepend an existing linear
    transform, or ``init="center-of-mass"`` to align centers first.
    """
    reg = _new_reg(
        dim,
        output,
        warped_output,
        inverse_warped_output,
        use_float,
        winsorize,
        use_histogram_matching,
        collapse_output_transforms,
        write_composite_transform,
        random_seed,
        verbose,
        ants_path,
    )
    if initial_transform is not None:
        reg.add_initial_moving_transform(initial_transform)
    if init is not None:
        reg.initialize_from_images(fixed, moving, init=init)
    if mask is not None or moving_mask is not None:
        reg.set_masks(mask, moving_mask)
    _add_syn_stage(
        reg,
        fixed,
        moving,
        syn_metric,
        syn_bins,
        syn_radius,
        syn_sampling,
        syn_sampling_pct,
        grad_step,
        flow_sigma,
        total_sigma,
        syn_iterations,
        syn_shrink_factors,
        syn_smoothing_sigmas,
        smoothing_units,
    )
    return reg


def _add_syn_stage(
    reg,
    fixed,
    moving,
    metric,
    bins,
    radius,
    sampling,
    sampling_pct,
    grad_step,
    flow_sigma,
    total_sigma,
    iterations,
    shrink_factors,
    smoothing_sigmas,
    smoothing_units,
):
    reg.add_stage(
        transform=SyN(gradient_step=grad_step, update_field_variance=flow_sigma, total_field_variance=total_sigma),
        metrics=_make_metric(
            metric, fixed, moving, bins=bins, radius=radius, sampling=sampling, sampling_pct=sampling_pct
        ),
        convergence=Convergence(iterations),
        shrink_factors=list(shrink_factors),
        smoothing_sigmas=list(smoothing_sigmas),
        smoothing_units=smoothing_units,
    )


# ---------------------------------------------------------------------------
# Allen CCF / STPT template-construction analogue (ANTs)
# ---------------------------------------------------------------------------
# The closest ANTs parameter analogue of the AllenInstitute/stpt_registration
# pipeline (Oh 2014 / Kuan 2015), the elastix `ccf_stpt` preset's ANTs sibling:
#   - global: Rigid + Affine, Mattes mutual information with 64 histogram bins
#     (matches the STPT metric), center-of-gravity init, coarse-to-fine from
#     shrink factor 8;
#   - deformable: BSplineSyN driven by neighborhood cross-correlation (CC).
#
# Why BSplineSyN + CC: the STPT local step is a 3rd-order B-spline optimized
# symmetrically (forward + backward, composed -> invertible) under cross-correlation.
# ANTs' BSplineSyN is a B-spline-regularized SYMMETRIC diffeomorphic transform, so it
# captures both the B-spline character AND the symmetric/invertible property -- a
# closer match than elastix's (asymmetric) B-spline. CC is the direct NCC analogue.
# The original deformable optimizer is a discrete MRF, so this remains an analogue,
# not a reproduction (use `commandants.stpt` to drive the real binaries).

_CCF_BINS = 64
_CCF_LINEAR_SHRINK = (8, 4, 2, 1)  # coarse start at factor 8 (STPT StartingFactors(8))
_CCF_LINEAR_SMOOTH = (3, 2, 1, 0)
_CCF_LINEAR_ITERS = (1000, 500, 250, 100)
_CCF_DEF_SHRINK = (8, 4, 2, 1)
_CCF_DEF_SMOOTH = (3, 2, 1, 0)
_CCF_DEF_ITERS = (100, 70, 50, 20)


def ccf_global(fixed, moving, output, *, bins=_CCF_BINS, **kwargs) -> AntsRegistration:
    """STPT/CCF *global* analogue: Rigid + Affine, Mattes MI (64 bins), COM init, coarse-to-fine from shrink 8."""
    kwargs.setdefault("iterations", _CCF_LINEAR_ITERS)
    kwargs.setdefault("shrink_factors", _CCF_LINEAR_SHRINK)
    kwargs.setdefault("smoothing_sigmas", _CCF_LINEAR_SMOOTH)
    return _linear_preset(fixed, moving, output, [Rigid, Affine], metric="mattes", bins=bins, **kwargs)


def ccf_stpt(
    fixed,
    moving,
    output,
    *,
    dim=3,
    warped_output=None,
    inverse_warped_output=None,
    init="center-of-mass",
    initial_transform=None,
    # linear (Rigid + Affine): Mattes MI, 64 bins
    aff_bins=_CCF_BINS,
    aff_sampling="Regular",
    aff_sampling_pct=_LINEAR_SAMPLING_PCT,
    aff_grad_step=_LINEAR_GRAD_STEP,
    aff_iterations=_CCF_LINEAR_ITERS,
    aff_shrink_factors=_CCF_LINEAR_SHRINK,
    aff_smoothing_sigmas=_CCF_LINEAR_SMOOTH,
    # deformable: BSplineSyN + cross-correlation
    grad_step=0.1,
    update_mesh_size=26,
    total_mesh_size=0,
    syn_radius=4,
    def_iterations=_CCF_DEF_ITERS,
    def_shrink_factors=_CCF_DEF_SHRINK,
    def_smoothing_sigmas=_CCF_DEF_SMOOTH,
    smoothing_units="vox",
    use_float=True,
    winsorize=(0.005, 0.995),
    use_histogram_matching=False,
    mask=None,
    moving_mask=None,
    collapse_output_transforms=None,
    write_composite_transform=False,
    random_seed=None,
    verbose=False,
    ants_path=None,
) -> AntsRegistration:
    """STPT/CCF analogue (ANTs): Rigid + Affine (Mattes MI, 64 bins) -> BSplineSyN (cross-correlation).

    Mirrors AllenInstitute/stpt_registration template construction. See the module notes
    above for what transfers and why BSplineSyN + CC is the closest ANTs match to the
    original symmetric B-spline deformable step.
    """
    reg = _new_reg(
        dim,
        output,
        warped_output,
        inverse_warped_output,
        use_float,
        winsorize,
        use_histogram_matching,
        collapse_output_transforms,
        write_composite_transform,
        random_seed,
        verbose,
        ants_path,
    )
    if initial_transform is not None:
        reg.add_initial_moving_transform(initial_transform)
    elif init is not None:
        reg.initialize_from_images(fixed, moving, init=init)
    if mask is not None or moving_mask is not None:
        reg.set_masks(mask, moving_mask)
    for transform_cls in (Rigid, Affine):
        reg.add_stage(
            transform=transform_cls(gradient_step=aff_grad_step),
            metrics=_make_metric(
                "mattes", fixed, moving, bins=aff_bins, sampling=aff_sampling, sampling_pct=aff_sampling_pct
            ),
            convergence=Convergence(aff_iterations),
            shrink_factors=list(aff_shrink_factors),
            smoothing_sigmas=list(aff_smoothing_sigmas),
            smoothing_units=smoothing_units,
        )
    reg.add_stage(
        transform=BSplineSyN(
            gradient_step=grad_step, update_mesh_size=update_mesh_size, total_mesh_size=total_mesh_size
        ),
        metrics=_make_metric("cc", fixed, moving, radius=syn_radius),
        convergence=Convergence(def_iterations),
        shrink_factors=list(def_shrink_factors),
        smoothing_sigmas=list(def_smoothing_sigmas),
        smoothing_units=smoothing_units,
    )
    return reg


#: Alias for users who write it without the underscore.
synonly = syn_only

__all__ = [
    "translation",
    "rigid",
    "similarity",
    "affine",
    "syn",
    "syn_only",
    "synonly",
    "ccf_global",
    "ccf_stpt",
]
