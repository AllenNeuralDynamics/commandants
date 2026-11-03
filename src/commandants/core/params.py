"""Formatting helpers for ANTs command-line arguments.

ANTs uses two recurring argument idioms that these helpers centralize so their
quirks live in exactly one place:

* **Bracketed parameter lists** -- e.g. ``MI[fixed.nii,moving.nii,1,32]`` or
  ``[corrected.nii,bias.nii]``. Trailing unset values are dropped so the emitted
  string matches how ANTs itself is normally invoked.
* **``x``-joined multi-resolution vectors** -- e.g. ``1000x500x250x0`` for
  iterations, shrink factors, and smoothing sigmas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence


def fmt_value(value: Any) -> str:
    """Render a single argument value as ANTs expects it.

    * ``bool`` -> ``"1"``/``"0"`` (ANTs uses integer flags, not ``True``/``False``).
    * ``Path`` -> ``str``.
    * everything else -> ``str``.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def bracket(*parts: Any, trim_trailing: bool = True) -> str:
    """Build a bracketed parameter list like ``[a,b,c]``.

    ``None`` entries become empty positions. When ``trim_trailing`` is True
    (the default) trailing ``None`` entries are dropped entirely, so
    ``bracket("a", 1, None, None)`` -> ``"[a,1]"`` while
    ``bracket("a", None, 3)`` -> ``"[a,,3]"`` (an interior gap is preserved
    because position is significant to ANTs).
    """
    rendered: list[str | None] = [None if p is None else fmt_value(p) for p in parts]

    if trim_trailing:
        while rendered and rendered[-1] is None:
            rendered.pop()

    body = ",".join("" if r is None else r for r in rendered)
    return f"[{body}]"


def xjoin(values: Iterable[Any]) -> str:
    """Join a multi-resolution vector with ``x`` -> ``"8x4x2x1"``."""
    return "x".join(fmt_value(v) for v in values)


def str_resolve(value: Any, name: str | None = None, *, must_exist: bool = False) -> str:
    """Default image resolver: just stringify.

    Command builders pass a smarter resolver (one that writes in-memory SimpleITK
    images to temp files) during a real run; this identity-like fallback is used
    when a metric/transform renders on its own or when no in-memory images are
    involved. The ``name`` and ``must_exist`` arguments are accepted (and ignored)
    so it shares the resolver call signature -- existence is only ever checked on a
    real run, which goes through the materializing resolver.
    """
    return fmt_value(value)


def as_sequence(value: Any) -> Sequence[Any]:
    """Coerce a scalar-or-sequence into a list; leave existing sequences alone.

    Strings and ``Path`` objects are treated as scalars, not iterables.
    """
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
