"""elastix support for commandants -- a transparent CLI wrapper around the
``elastix``/``transformix`` binaries, reusing the shared commandants core.

Quickstart
----------
>>> from commandants.elastix import presets, Transformix
>>> reg = presets.affine("fixed.nii.gz", "moving.nii.gz", "out_dir")
>>> print(reg.to_shell())          # inspect the elastix command (no binary needed)
>>> # result = reg.run(stream=True)  # requires elastix (commandants install-elastix)

Configuration is via parameter maps (``ParameterMap`` / dict / .txt path), not flags.
"""

from __future__ import annotations

from . import presets
from .apply import Transformix
from .parameters import ParameterMap, is_param_map, to_parameter_map
from .registration import Elastix

__all__ = [
    "Elastix",
    "Transformix",
    "ParameterMap",
    "is_param_map",
    "to_parameter_map",
    "presets",
]
