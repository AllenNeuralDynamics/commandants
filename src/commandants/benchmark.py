"""Head-to-head registration benchmarking across backends.

Register the **same image pair** with several backends (ANTs, elastix, ...) and collect
comparable timings plus normalized output paths::

    from commandants import benchmark

    report = benchmark.compare("fixed.nii.gz", "moving.nii.gz", "bench_out", threads=8)
    print(report.to_table())
    report.to_csv("bench.csv")

Why this module exists: the backends are genuinely different tools, so their builders and
results do not line up. ANTs takes a transform *prefix* and an opt-in ``warped_output``, and
only exposes transform filenames through its own ``expected_transforms()``; elastix takes an
output *directory*, always writes a result image, and declares everything in
``declared_outputs()``. A :class:`Backend` adapter hides that difference behind a uniform
``build()`` / ``normalize()`` pair, so a third backend is a one-line registration.

Timing is already apples-to-apples in the core: every backend subclasses ``AntsCommand`` and
shares one ``run()`` path, which times only the subprocess (image materialization and output
bookkeeping happen outside the timed window). What this module adds is **fairness around**
that measurement:

* streaming is forced off (it would bias whichever tool is chattier -- ANTs, typically);
* threads are pinned identically for every backend;
* the warped-image format is equalized, so compression cost inside the timed window matches;
* ``check=False`` plus per-backend error capture, so one tool failing does not discard the
  others' numbers;
* optional ``warmup`` and ``repeats`` to absorb page-cache and binary-load effects.

Accuracy (Dice, landmark error, ...) is deliberately out of scope -- the harness gives you
each backend's warped image and transforms at predictable paths so you can score them yourself.
"""

from __future__ import annotations

import csv
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .core.executable import is_available
from .core.runner import AntsCommand, CompletedAnts

#: Extension used for every backend's warped output, so gzip cost is equal across
#: backends inside the timed region. ".nii" matches the elastix presets'
#: ``ResultImageFormat`` default; change both together if you override it.
DEFAULT_WARPED_EXT = ".nii"


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """Adapter making one registration tool benchmarkable.

    Parameters
    ----------
    name:
        Short key used in reports and in ``compare(backends=...)``.
    binary:
        Executable probed with :func:`commandants.core.executable.is_available`.
    build:
        ``(fixed, moving, out_dir, **opts) -> AntsCommand`` -- an un-run builder.
    normalize:
        ``(command, result) -> {"warped": str|None, "transforms": [str], "log": str|None}``.
    """

    name: str
    binary: str
    build: Callable[..., AntsCommand]
    normalize: Callable[[AntsCommand, CompletedAnts], dict]

    def available(self) -> bool:
        """True if this backend's binary can be resolved."""
        return is_available(self.binary)


def _build_ants(
    fixed: Any,
    moving: Any,
    out_dir: str,
    *,
    preset: str = "affine",
    fixed_mask: Any = None,
    moving_mask: Any = None,
    threads: Optional[int] = None,  # noqa: ARG001 - pinned via env, not a preset kwarg
    warped_ext: str = DEFAULT_WARPED_EXT,
    **kwargs: Any,
) -> AntsCommand:
    """Build an ANTs registration from ``commandants.presets``."""
    from . import presets as ants_presets

    os.makedirs(out_dir, exist_ok=True)
    builder = getattr(ants_presets, preset)
    return builder(
        fixed,
        moving,
        os.path.join(out_dir, "ants_"),
        warped_output=os.path.join(out_dir, f"ants_warped{warped_ext}"),
        mask=fixed_mask,  # ANTs spells the fixed-side mask `mask=`
        moving_mask=moving_mask,
        **kwargs,
    )


def _build_elastix(
    fixed: Any,
    moving: Any,
    out_dir: str,
    *,
    preset: str = "affine",
    fixed_mask: Any = None,
    moving_mask: Any = None,
    threads: Optional[int] = None,
    warped_ext: str = DEFAULT_WARPED_EXT,  # noqa: ARG001 - set by the parameter map
    **kwargs: Any,
) -> AntsCommand:
    """Build an elastix registration from ``commandants.elastix.presets``."""
    from .elastix import presets as elastix_presets

    builder = getattr(elastix_presets, preset)
    return builder(
        fixed,
        moving,
        out_dir,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        threads=threads,
        **kwargs,
    )


def _build_stpt(
    fixed: Any,
    moving: Any,
    out_dir: str,
    *,
    preset: str = "affine",
    fixed_mask: Any = None,
    moving_mask: Any = None,
    threads: Optional[int] = None,
    warped_ext: str = DEFAULT_WARPED_EXT,
    **kwargs: Any,
) -> AntsCommand:
    """Build an Allen STPT/CCF registration from ``commandants.stpt.presets``."""
    from .stpt import presets as stpt_presets

    builder = getattr(stpt_presets, preset)
    return builder(
        fixed,
        moving,
        out_dir,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        threads=threads,
        warped_ext=warped_ext,
        **kwargs,
    )


def _normalize_ants(command: AntsCommand, result: CompletedAnts) -> dict:
    """ANTs: warped comes from declared outputs, transforms from expected_transforms()."""
    info = command.expected_transforms()
    root = info.get("output_dir", "")
    transforms = [os.path.join(root, t) for t in info.get("forward", [])]
    return {
        "warped": result.outputs.get("warped") or info.get("warped"),
        "transforms": transforms,
        "log": None,
    }


def _normalize_elastix(command: AntsCommand, result: CompletedAnts) -> dict:  # noqa: ARG001
    """elastix: everything is declared. The final TransformParameters chains earlier stages."""
    outputs = result.outputs
    final = outputs.get("transform")
    return {
        "warped": outputs.get("warped"),
        "transforms": [final] if final else [],
        "log": outputs.get("log"),
    }


def _normalize_stpt(command: AntsCommand, result: CompletedAnts) -> dict:  # noqa: ARG001
    """STPT declares everything. Transforms are listed deformable-first, as ANTs does."""
    outputs = result.outputs
    transforms = [p for p in (outputs.get("warp"), outputs.get("affine")) if p]
    return {
        "warped": outputs.get("warped"),
        "transforms": transforms,
        "log": outputs.get("log"),
    }


#: Registry of known backends. Add one with :func:`register_backend`.
BACKENDS: dict[str, Backend] = {
    "ants": Backend("ants", "antsRegistration", _build_ants, _normalize_ants),
    "elastix": Backend("elastix", "elastix", _build_elastix, _normalize_elastix),
    "stpt": Backend("stpt", "stpt_register", _build_stpt, _normalize_stpt),
}


def register_backend(backend: Backend) -> None:
    """Add (or replace) a backend in the registry."""
    BACKENDS[backend.name] = backend


def available_backends() -> list[str]:
    """Names of registered backends whose binary can be resolved right now."""
    return [name for name, backend in BACKENDS.items() if backend.available()]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One backend's single registration run."""

    backend: str
    trial: int
    warmup: bool = False
    ok: bool = False
    returncode: Optional[int] = None
    duration_seconds: Optional[float] = None
    warped: Optional[str] = None
    transforms: list[str] = field(default_factory=list)
    log: Optional[str] = None
    error: Optional[str] = None
    argv: list[str] = field(default_factory=list)


_CSV_FIELDS = (
    "backend",
    "trial",
    "warmup",
    "ok",
    "returncode",
    "duration_seconds",
    "warped",
    "transforms",
    "log",
    "error",
)


@dataclass
class BenchmarkReport:
    """All trials from a :func:`compare` call, plus summarization helpers."""

    trials: list[Trial] = field(default_factory=list)

    def timed(self) -> list[Trial]:
        """Successful, non-warmup trials -- the ones that count."""
        return [t for t in self.trials if t.ok and not t.warmup and t.duration_seconds is not None]

    def summary(self) -> dict[str, dict]:
        """Per-backend ``{runs, best, median, mean}`` over timed trials."""
        out: dict[str, dict] = {}
        for name in BACKENDS:
            durations = [t.duration_seconds for t in self.timed() if t.backend == name]
            if durations:
                out[name] = {
                    "runs": len(durations),
                    "best": min(durations),
                    "median": statistics.median(durations),
                    "mean": statistics.fmean(durations),
                }
        return out

    def to_table(self) -> str:
        """Render a human-readable summary + per-trial detail."""
        lines = [f"{'backend':<12} {'runs':>4} {'best(s)':>9} {'median(s)':>10} {'mean(s)':>9}"]
        lines.append("-" * 48)
        for name, stats in self.summary().items():
            lines.append(
                f"{name:<12} {stats['runs']:>4} {stats['best']:>9.2f} {stats['median']:>10.2f} {stats['mean']:>9.2f}"
            )
        failures = [t for t in self.trials if not t.ok]
        if failures:
            lines.append("")
            lines.append("failures:")
            for t in failures:
                lines.append(f"  {t.backend} trial {t.trial}: rc={t.returncode} {t.error or ''}".rstrip())
        return "\n".join(lines)

    def to_csv(self, path: str | os.PathLike[str]) -> str:
        """Write every trial as a CSV row; returns the path."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for t in self.trials:
                row = {k: getattr(t, k) for k in _CSV_FIELDS}
                row["transforms"] = os.pathsep.join(t.transforms)
                writer.writerow(row)
        return str(path)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _thread_env(threads: Optional[int]) -> Optional[dict]:
    """A full environment copy with the ITK thread cap pinned (run(env=) replaces os.environ)."""
    if threads is None:
        return None
    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    return env


def compare(
    fixed: Any,
    moving: Any,
    out_dir: str | os.PathLike[str],
    *,
    backends: Sequence[str] = ("ants", "elastix", "stpt"),
    preset: str = "affine",
    fixed_mask: Any = None,
    moving_mask: Any = None,
    threads: Optional[int] = None,
    repeats: int = 1,
    warmup: int = 0,
    warped_ext: str = DEFAULT_WARPED_EXT,
    skip_unavailable: bool = True,
    build_kwargs: Optional[dict[str, dict]] = None,
) -> BenchmarkReport:
    """Register the same pair with each backend and time it.

    Parameters
    ----------
    fixed, moving:
        Paths or in-memory SimpleITK images (materialized once per trial, outside timing).
    out_dir:
        Root output directory; each trial gets its own ``<backend>_trial<N>`` subfolder so
        runs cannot contaminate one another.
    backends:
        Registry names to run (see :data:`BACKENDS`).
    preset:
        Preset name looked up on each backend's ``presets`` module. Use a name both
        backends define (e.g. ``"affine"``, ``"rigid"``, ``"ccf_stpt"``).
    threads:
        Pinned for every backend -- elastix via ``-threads``, ANTs via
        ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS``. Leave None to use each tool's default
        (not recommended for a comparison).
    repeats, warmup:
        Timed runs per backend, and discarded runs before them.
    build_kwargs:
        Per-backend extra builder kwargs, e.g. ``{"ants": {"use_float": False}}``.

    Returns
    -------
    BenchmarkReport
        Every trial, with ``summary()`` / ``to_table()`` / ``to_csv()`` helpers.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    build_kwargs = build_kwargs or {}
    env = _thread_env(threads)
    report = BenchmarkReport()

    for name in backends:
        backend = BACKENDS.get(name)
        if backend is None:
            raise ValueError(f"Unknown backend {name!r}; known: {sorted(BACKENDS)}")
        if skip_unavailable and not backend.available():
            report.trials.append(Trial(backend=name, trial=0, error=f"{backend.binary} not found; skipped"))
            continue

        for index in range(warmup + repeats):
            is_warmup = index < warmup
            trial = Trial(backend=name, trial=index, warmup=is_warmup)
            trial_dir = os.path.join(out_dir, f"{name}_trial{index}")
            try:
                command = backend.build(
                    fixed,
                    moving,
                    trial_dir,
                    preset=preset,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    threads=threads,
                    warped_ext=warped_ext,
                    **build_kwargs.get(name, {}),
                )
                # Timed runs never stream: line-by-line pumping happens inside the timed
                # window and would penalize the chattier tool.
                result = command.run(check=False, env=env, stream=False, on_line=None, log_file=None)
                trial.argv = list(result.argv)
                trial.returncode = result.returncode
                trial.duration_seconds = result.duration_seconds
                trial.ok = result.returncode == 0
                if trial.ok:
                    normalized = backend.normalize(command, result)
                    trial.warped = normalized.get("warped")
                    trial.transforms = list(normalized.get("transforms") or [])
                    trial.log = normalized.get("log")
                else:
                    trial.error = (result.stderr or result.stdout or "").strip()[-500:] or None
            except Exception as exc:  # one backend failing must not lose the others
                trial.error = f"{type(exc).__name__}: {exc}"
            report.trials.append(trial)

    return report


__all__ = [
    "Backend",
    "BenchmarkReport",
    "Trial",
    "BACKENDS",
    "compare",
    "register_backend",
    "available_backends",
    "DEFAULT_WARPED_EXT",
]
