"""``ResampleImage`` wrapper (positional-argument tool).

ANTs usage::

    ResampleImage Dim in.nii out.nii MxNxO[xP] [size=1,spacing=0] [interp] [pixeltype]

The 4th argument is the target size or spacing vector; the 5th flag says which:
``1`` = interpret the vector as the output **size** (voxel counts), ``0`` =
interpret it as the output **spacing** (physical units).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from ..core.params import xjoin
from ..core.runner import AntsCommand

PathLike = Union[str, Path]


class ResampleImage(AntsCommand):
    """Resample an image to a target size or spacing.

    Parameters
    ----------
    dimensionality:
        Image dimensionality.
    input_image, output_image:
        Input and output paths.
    dims:
        The MxNxO vector (as a sequence, e.g. ``[128, 128, 64]`` or
        ``[1.0, 1.0, 1.0]``).
    interpret:
        ``"size"`` (voxel counts) or ``"spacing"`` (physical units). Controls the
        ``size=1,spacing=0`` flag. ``None`` omits the flag (ANTs default).
    interpolation:
        Integer interpolation code (0=linear, 1=nearest, 2=gaussian,
        3=windowedSinc, 4=bspline). ``None`` omits it.
    pixeltype:
        Optional output pixel type code.
    """

    binary_name = "ResampleImage"

    def __init__(
        self,
        dimensionality: int,
        input_image: PathLike,
        output_image: PathLike,
        dims: Sequence[float],
        *,
        interpret: Optional[str] = None,
        interpolation: Optional[int] = None,
        pixeltype: Optional[int] = None,
        ants_path: Optional[PathLike] = None,
    ) -> None:
        super().__init__(ants_path=ants_path)
        self.dimensionality = dimensionality
        self.input_image = input_image
        self.output_image = output_image
        self.dims = dims
        if interpret not in (None, "size", "spacing"):
            raise ValueError("interpret must be 'size', 'spacing', or None.")
        self.interpret = interpret
        self.interpolation = interpolation
        self.pixeltype = pixeltype

    def _build_args(self) -> List[str]:
        args = [
            str(self.dimensionality),
            self._resolve(self.input_image, "input", must_exist=True),
            str(self.output_image),
            xjoin(self.dims),
        ]
        # The trailing positional args are order-sensitive; only emit a later one
        # if it (or an earlier one it depends on) is set.
        if self.interpret is not None or self.interpolation is not None or self.pixeltype is not None:
            size_flag = "1" if self.interpret == "size" else "0"
            args.append(size_flag)
        if self.interpolation is not None or self.pixeltype is not None:
            args.append(str(self.interpolation if self.interpolation is not None else 0))
        if self.pixeltype is not None:
            args.append(str(self.pixeltype))
        return args

    def declared_outputs(self) -> dict:
        return {"output": str(self.output_image)}


__all__ = ["ResampleImage"]
