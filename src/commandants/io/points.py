"""Read/write ANTs point-set CSV files.

``antsApplyTransformsToPoints`` consumes and produces CSV files with a header row
naming the coordinate columns (``x,y,z,t``) plus optional ``label`` and
``comment`` columns. Coordinates are in **physical space** (LPS/RAS per the image
header), not voxel indices.

These helpers use only the standard library for I/O; returning coordinates as a
NumPy array (the default) lazily imports NumPy, which ships with the optional
``[io]`` extra.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

_COORD_NAMES = ["x", "y", "z", "t"]


def write_points(
    path: PathLike,
    coords: Sequence[Sequence[float]],
    labels: Optional[Sequence[Any]] = None,
    comments: Optional[Sequence[Any]] = None,
    dim: Optional[int] = None,
) -> str:
    """Write points to an ANTs-compatible CSV.

    Parameters
    ----------
    path:
        Destination ``.csv`` path.
    coords:
        Sequence of coordinate rows (each of length 2, 3, or 4) or a 2D NumPy
        array.
    labels, comments:
        Optional per-point label / comment columns.
    dim:
        Coordinate dimensionality; inferred from the first row if omitted.

    Returns
    -------
    str
        The written path (as a string).
    """
    rows = [list(row) for row in coords]
    if not rows:
        raise ValueError("No points to write.")
    if dim is None:
        dim = len(rows[0])
    if dim < 2 or dim > 4:
        raise ValueError(f"dim must be 2, 3, or 4; got {dim}.")

    header = _COORD_NAMES[:dim]
    if labels is not None:
        header.append("label")
    if comments is not None:
        header.append("comment")

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, row in enumerate(rows):
            if len(row) < dim:
                raise ValueError(f"Point {i} has {len(row)} coordinates but dim={dim}.")
            out = list(row[:dim])
            if labels is not None:
                out.append(labels[i])
            if comments is not None:
                out.append(comments[i])
            writer.writerow(out)
    return str(path)


def read_points(
    path: PathLike,
    as_array: bool = True,
) -> Tuple[Any, dict]:
    """Read points from an ANTs CSV.

    Returns a ``(coords, extra)`` tuple where ``coords`` is an ``(N, dim)`` NumPy
    array (or a list of lists when ``as_array=False``) and ``extra`` is a dict
    that may contain ``"label"`` and ``"comment"`` lists when those columns are
    present.
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row]

    coord_cols = [i for i, name in enumerate(header) if name.strip().lower() in _COORD_NAMES]
    label_idx = _find_col(header, "label")
    comment_idx = _find_col(header, "comment")

    coords = [[float(row[i]) for i in coord_cols] for row in rows]
    extra: dict = {}
    if label_idx is not None:
        extra["label"] = [row[label_idx] for row in rows]
    if comment_idx is not None:
        extra["comment"] = [row[comment_idx] for row in rows]

    if as_array:
        try:
            import numpy as np  # noqa: PLC0415 (intentional lazy import)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "read_points(as_array=True) requires NumPy. Install the [io] extra: "
                "pip install 'commandants[io]' -- or pass as_array=False."
            ) from exc
        return np.asarray(coords, dtype=float), extra
    return coords, extra


def _find_col(header: List[str], name: str) -> Optional[int]:
    for i, col in enumerate(header):
        if col.strip().lower() == name:
            return i
    return None


__all__ = ["write_points", "read_points"]
