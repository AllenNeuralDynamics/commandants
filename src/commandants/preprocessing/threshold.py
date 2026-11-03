"""``ThresholdImage`` wrapper (positional-argument tool).

ANTs usage::

    ThresholdImage Dim in.nii out.nii threshlo threshhi <inside=1> <outside=0>
    ThresholdImage Dim in.nii out.nii Otsu   numberOfThresholds
    ThresholdImage Dim in.nii out.nii Kmeans numberOfThresholds <segImage>
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from ..core.params import fmt_value
from ..core.runner import AntsCommand

PathLike = Union[str, Path]


class ThresholdImage(AntsCommand):
    """Intensity thresholding (and Otsu/Kmeans via classmethods).

    The primary constructor performs range thresholding: voxels with intensity in
    ``[lower, upper]`` are set to ``inside`` (default 1), others to ``outside``
    (default 0).
    """

    binary_name = "ThresholdImage"

    def __init__(
        self,
        dimensionality: int,
        input_image: PathLike,
        output_image: PathLike,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        inside: Optional[float] = None,
        outside: Optional[float] = None,
        *,
        _positional_tail: Optional[List] = None,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.input_image = input_image
        self.output_image = output_image
        self.lower = lower
        self.upper = upper
        self.inside = inside
        self.outside = outside
        # Used by the Otsu/Kmeans classmethods to supply their own trailing args.
        self._positional_tail = _positional_tail

    @classmethod
    def otsu(
        cls,
        dimensionality: int,
        input_image: PathLike,
        output_image: PathLike,
        n_thresholds: int = 1,
        ants_path: Optional[PathLike] = None,
    ) -> "ThresholdImage":
        """Otsu multi-threshold segmentation."""
        return cls(
            dimensionality,
            input_image,
            output_image,
            _positional_tail=["Otsu", n_thresholds],
            ants_path=ants_path,
        )

    @classmethod
    def kmeans(
        cls,
        dimensionality: int,
        input_image: PathLike,
        output_image: PathLike,
        n_thresholds: int = 1,
        ants_path: Optional[PathLike] = None,
    ) -> "ThresholdImage":
        """K-means multi-threshold segmentation."""
        return cls(
            dimensionality,
            input_image,
            output_image,
            _positional_tail=["Kmeans", n_thresholds],
            ants_path=ants_path,
        )

    def _build_args(self) -> List[str]:
        args = [
            str(self.dimensionality),
            self._resolve(self.input_image, "input", must_exist=True),
            str(self.output_image),
        ]
        if self._positional_tail is not None:
            args += [fmt_value(v) for v in self._positional_tail]
            return args

        if self.lower is None or self.upper is None:
            raise ValueError(
                "Range thresholding requires both lower and upper (or use the "
                "ThresholdImage.otsu / .kmeans classmethods)."
            )
        # Emit lower, upper, then trailing inside/outside dropping unset tail.
        tail = [self.lower, self.upper, self.inside, self.outside]
        while tail and tail[-1] is None:
            tail.pop()
        args += [fmt_value(v) for v in tail]
        return args

    def declared_outputs(self) -> dict:
        return {"output": str(self.output_image)}


__all__ = ["ThresholdImage"]
