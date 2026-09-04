"""``antsApplyTransforms`` and ``antsApplyTransformsToPoints`` wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..core.params import bracket
from ..core.runner import AntsCommand

PathLike = Union[str, Path]

# Valid values for antsApplyTransforms --output-data-type.
_OUTPUT_DATA_TYPES = {"char", "uchar", "short", "int", "float", "double", "default"}

# Accept friendly names or the raw integer ANTs expects for --input-image-type.
_IMAGE_TYPES = {
    "scalar": 0,
    "vector": 1,
    "tensor": 2,
    "time-series": 3,
    "timeseries": 3,
}


def _image_type_code(value: Union[int, str]) -> int:
    if isinstance(value, int):
        return value
    key = value.lower()
    if key not in _IMAGE_TYPES:
        raise ValueError(f"image_type must be one of {sorted(_IMAGE_TYPES)} or 0-3; got {value!r}")
    return _IMAGE_TYPES[key]


class AntsApplyTransforms(AntsCommand):
    """Apply an ordered list of transforms to resample an image.

    Transforms are applied in the order ANTs expects (the *last* transform added
    is applied *first* to the moving image). Each transform may be inverted --
    handy for reusing a forward affine in the inverse direction.

    Computations default to single precision (``use_float=True``); pass
    ``use_float=False`` for double or ``None`` to omit the flag. ``output_data_type``
    (``char``/``uchar``/``short``/``int``/``float``/``double``/``default``) sets the
    *stored* pixel type of the output image, independent of the computation
    precision.
    """

    binary_name = "antsApplyTransforms"

    def __init__(
        self,
        dimensionality: int,
        input_image: PathLike,
        reference_image: PathLike,
        output: PathLike,
        *,
        interpolation: Optional[str] = None,
        image_type: Union[int, str, None] = None,
        default_value: Optional[float] = None,
        use_float: Optional[bool] = True,  # single precision by default (memory-friendly)
        output_data_type: Optional[str] = None,
        verbose: bool = False,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.input_image = input_image
        self.reference_image = reference_image
        self.output = output
        self.interpolation = interpolation
        self.image_type = image_type
        self.default_value = default_value
        self.use_float = use_float
        if output_data_type is not None and output_data_type not in _OUTPUT_DATA_TYPES:
            raise ValueError(f"output_data_type must be one of {sorted(_OUTPUT_DATA_TYPES)}; got {output_data_type!r}")
        self.output_data_type = output_data_type
        self.verbose = verbose
        self._transforms: List[Tuple[PathLike, bool]] = []

    def add_transform(self, transform: PathLike, invert: bool = False) -> "AntsApplyTransforms":
        """Append a transform to the list (optionally inverted)."""
        self._transforms.append((transform, invert))
        return self

    def add_transforms(self, *transforms: PathLike) -> "AntsApplyTransforms":
        """Append several non-inverted transforms in order."""
        for tx in transforms:
            self._transforms.append((tx, False))
        return self

    def _build_args(self) -> List[str]:
        args: List[str] = [
            "--dimensionality",
            str(self.dimensionality),
            "--input",
            self._resolve(self.input_image, "input"),
            "--reference-image",
            self._resolve(self.reference_image, "reference"),
            "--output",
            str(self.output),
        ]
        if self.interpolation is not None:
            args += ["--interpolation", self.interpolation]
        if self.image_type is not None:
            args += ["--input-image-type", str(_image_type_code(self.image_type))]
        if self.default_value is not None:
            args += ["--default-value", str(self.default_value)]
        if self.use_float is not None:
            args += ["--float", "1" if self.use_float else "0"]
        if self.output_data_type is not None:
            args += ["--output-data-type", self.output_data_type]
        for i, (tx, invert) in enumerate(self._transforms):
            path = self._resolve(tx, f"transform{i}")
            args += ["--transform", bracket(path, True) if invert else path]
        args += ["--verbose", "1" if self.verbose else "0"]
        return args

    def declared_outputs(self) -> dict:
        return {"output": str(self.output)}


class AntsApplyTransformsToPoints(AntsCommand):
    """Apply transforms to a point set (CSV of physical-space coordinates).

    Note: point mapping goes the *opposite* direction to image mapping (an ANTs
    convention). To move points from moving space into fixed space you typically
    use the inverse of the transform used to warp the image. See
    :mod:`commandants.io.points` for reading/writing the CSV format.
    """

    binary_name = "antsApplyTransformsToPoints"

    def __init__(
        self,
        dimensionality: int,
        input_csv: PathLike,
        output_csv: PathLike,
        *,
        precision: Optional[int] = None,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.precision = precision
        self._transforms: List[Tuple[PathLike, bool]] = []

    def add_transform(self, transform: PathLike, invert: bool = False) -> "AntsApplyTransformsToPoints":
        """Append a transform to the list (optionally inverted)."""
        self._transforms.append((transform, invert))
        return self

    def _build_args(self) -> List[str]:
        args: List[str] = [
            "--dimensionality",
            str(self.dimensionality),
            "--input",
            str(self.input_csv),
            "--output",
            str(self.output_csv),
        ]
        for i, (tx, invert) in enumerate(self._transforms):
            path = self._resolve(tx, f"transform{i}")
            args += ["--transform", bracket(path, True) if invert else path]
        if self.precision is not None:
            args += ["--precision", str(self.precision)]
        return args

    def declared_outputs(self) -> dict:
        return {"output": str(self.output_csv)}


__all__ = ["AntsApplyTransforms", "AntsApplyTransformsToPoints"]
