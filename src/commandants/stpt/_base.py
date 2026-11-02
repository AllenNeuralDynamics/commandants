"""Shared base for the Allen STPT/CCF (``idp*``) command wrappers.

The Allen registration binaries reuse the binary-agnostic
:class:`~commandants.core.runner.AntsCommand` core, exactly like the ANTs and elastix
wrappers do; this base adds the one thing they all share -- creating the output
directory before the run, since the modules expect it to exist.

Unlike ANTs and elastix, these binaries are **not downloadable**: they are built from
https://github.com/AllenInstitute/stpt_registration by the user, so they resolve from
``$STPTPATH`` or ``PATH`` only (see :data:`commandants.install.STPT_SPEC`). "idp" is
Allen's namespace tag for its Informatics Data Processing pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from ..core.runner import AntsCommand


class StptToolCommand(AntsCommand):
    """Base for the STPT/CCF command wrappers; each subclass sets ``self.out_dir``."""

    out_dir: Any

    def _prepare_execution(self, env: Optional[Mapping[str, str]]) -> Optional[Mapping[str, str]]:
        """Create the output directory the Allen modules write into."""
        os.makedirs(str(self.out_dir), exist_ok=True)
        return env


__all__ = ["StptToolCommand"]
