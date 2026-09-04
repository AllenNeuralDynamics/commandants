"""elastix parameter maps.

An elastix registration is configured by one or more **parameter files** -- plain
text, one ``(Key value ...)`` per line -- rather than CLI flags. :class:`ParameterMap`
is a thin, ordered, dict-like model of that file: build one in Python, load an existing
``.txt``, tweak it, and hand it to :class:`~commandants.elastix.Elastix`. Nothing is
hidden -- it round-trips the exact text.

Value formatting follows elastix conventions: strings are double-quoted
(``(Metric "AdvancedMattesMutualInformation")``), numbers are bare
(``(NumberOfResolutions 4)``), and Python booleans become quoted ``"true"``/``"false"``.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, "os.PathLike[str]"]

_LINE = re.compile(r"^\s*\(\s*(\w+)\s*(.*?)\s*\)\s*$")
_TOKENS = re.compile(r'"[^"]*"|\S+')


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return '"true"' if v else '"false"'
    if isinstance(v, (int, float)):
        return str(v)
    return f'"{v}"'


class ParameterMap:
    """Ordered, dict-like model of an elastix parameter file."""

    def __init__(self, mapping: Optional[Mapping[str, Any]] = None) -> None:
        self._d: "OrderedDict[str, tuple]" = OrderedDict()
        if mapping is not None:
            self.update(mapping)

    # -- mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> tuple:
        return self._d[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._d[key] = tuple(value) if isinstance(value, (list, tuple)) else (value,)

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def get(self, key: str, default: Any = None):
        return self._d.get(key, default)

    # -- fluent building ------------------------------------------------------
    def set(self, key: str, *values: Any) -> "ParameterMap":
        """Set ``key`` to one or more values; returns self."""
        self._d[key] = tuple(values)
        return self

    def update(self, mapping: Mapping[str, Any]) -> "ParameterMap":
        for k, v in mapping.items():
            self[k] = v
        return self

    def copy(self) -> "ParameterMap":
        new = ParameterMap()
        new._d = OrderedDict(self._d)
        return new

    # -- text I/O -------------------------------------------------------------
    def to_text(self) -> str:
        lines = []
        for key, vals in self._d.items():
            body = " ".join(_fmt(v) for v in vals)
            lines.append(f"({key} {body})" if body else f"({key})")
        return "\n".join(lines) + "\n"

    def write(self, path: PathLike) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_text())
        return str(path)

    @classmethod
    def parse(cls, text: str) -> "ParameterMap":
        pm = cls()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            key, rest = m.group(1), m.group(2)
            vals = []
            for tok in _TOKENS.findall(rest):
                if len(tok) >= 2 and tok.startswith('"') and tok.endswith('"'):
                    vals.append(tok[1:-1])
                else:
                    try:
                        vals.append(int(tok))
                    except ValueError:
                        try:
                            vals.append(float(tok))
                        except ValueError:
                            vals.append(tok)
            pm._d[key] = tuple(vals)
        return pm

    @classmethod
    def from_file(cls, path: PathLike) -> "ParameterMap":
        with open(path, encoding="utf-8") as fh:
            return cls.parse(fh.read())

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ParameterMap({len(self._d)} keys: {list(self._d)[:4]}...)"


def is_param_map(obj: Any) -> bool:
    """True if ``obj`` is an in-memory parameter map (ParameterMap or dict)."""
    return isinstance(obj, (ParameterMap, dict))


def to_parameter_map(obj: Any) -> ParameterMap:
    """Coerce a ParameterMap or dict into a ParameterMap (copy)."""
    if isinstance(obj, ParameterMap):
        return obj.copy()
    if isinstance(obj, dict):
        return ParameterMap(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__} to a ParameterMap.")


__all__ = ["ParameterMap", "is_param_map", "to_parameter_map"]
