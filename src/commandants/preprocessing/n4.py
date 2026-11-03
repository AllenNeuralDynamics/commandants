"""``N4BiasFieldCorrection`` wrapper (flag-based tool)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from ..core.params import bracket, xjoin
from ..core.runner import AntsCommand

PathLike = Union[str, Path]


class N4BiasFieldCorrection(AntsCommand):
    """N4 inhomogeneity (bias field) correction.

    Parameters
    ----------
    dimensionality:
        Image dimensionality (2, 3, or 4).
    input_image:
        Image to correct.
    output:
        Path for the corrected image.
    bias_output:
        Optional path to also write the estimated bias field; when given the
        ``--output`` flag is emitted as ``[corrected,biasField]``.
    mask_image, weight_image:
        Optional mask / weight images.
    shrink_factor:
        Resampling factor to speed up fitting (ANTs default 4).
    convergence_iterations, convergence_threshold:
        Per-level iteration schedule and convergence threshold ->
        ``-c [i1xi2x...,threshold]``.
    bspline_distance, bspline_order:
        B-spline fitting mesh distance and (optional) order -> ``-b [dist,order]``.
    rescale_intensities:
        Rescale output intensities to the input range.
    histogram_sharpening:
        ``(fwhm, wiener_noise, n_bins)`` -> ``-t [fwhm,noise,bins]``.
    verbose:
        Verbose output.
    """

    binary_name = "N4BiasFieldCorrection"

    def __init__(
        self,
        dimensionality: int,
        input_image: PathLike,
        output: PathLike,
        *,
        bias_output: Optional[PathLike] = None,
        mask_image: Optional[PathLike] = None,
        weight_image: Optional[PathLike] = None,
        shrink_factor: Optional[int] = None,
        convergence_iterations: Optional[Sequence[int]] = None,
        convergence_threshold: Optional[float] = None,
        bspline_distance: Optional[float] = None,
        bspline_order: Optional[int] = None,
        rescale_intensities: Optional[bool] = None,
        histogram_sharpening: Optional[Sequence[float]] = None,
        verbose: bool = False,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.input_image = input_image
        self.output = output
        self.bias_output = bias_output
        self.mask_image = mask_image
        self.weight_image = weight_image
        self.shrink_factor = shrink_factor
        self.convergence_iterations = convergence_iterations
        self.convergence_threshold = convergence_threshold
        self.bspline_distance = bspline_distance
        self.bspline_order = bspline_order
        self.rescale_intensities = rescale_intensities
        self.histogram_sharpening = histogram_sharpening
        self.verbose = verbose

    def _output_arg(self) -> str:
        if self.bias_output is not None:
            return bracket(self.output, self.bias_output)
        return str(self.output)

    def _build_args(self) -> List[str]:
        args: List[str] = [
            "--image-dimensionality",
            str(self.dimensionality),
            "--input-image",
            self._resolve(self.input_image, "input", must_exist=True),
        ]
        if self.mask_image is not None:
            args += ["--mask-image", self._resolve(self.mask_image, "mask", must_exist=True)]
        if self.weight_image is not None:
            args += ["--weight-image", self._resolve(self.weight_image, "weight", must_exist=True)]
        if self.shrink_factor is not None:
            args += ["--shrink-factor", str(self.shrink_factor)]
        if self.convergence_iterations is not None:
            args += [
                "--convergence",
                bracket(xjoin(self.convergence_iterations), self.convergence_threshold),
            ]
        if self.bspline_distance is not None:
            args += ["--bspline-fitting", bracket(self.bspline_distance, self.bspline_order)]
        if self.rescale_intensities is not None:
            args += ["--rescale-intensities", "1" if self.rescale_intensities else "0"]
        if self.histogram_sharpening is not None:
            args += ["--histogram-sharpening", bracket(*self.histogram_sharpening)]
        args += ["--output", self._output_arg()]
        args += ["--verbose", "1" if self.verbose else "0"]
        return args

    def declared_outputs(self) -> dict:
        outputs = {"output": str(self.output)}
        if self.bias_output is not None:
            outputs["bias_field"] = str(self.bias_output)
        return outputs


__all__ = ["N4BiasFieldCorrection"]
