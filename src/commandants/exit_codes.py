"""Translate a process exit code into a human explanation.

Reality check: ANTs (like most ITK/C++ tools) does **not** publish a table of
numbered error codes. Its binaries exit ``0`` on success and ``1`` on failure
(the real detail is in stderr). Everything else you see is one of:

* a **POSIX signal** -- Python's ``subprocess`` reports these as a *negative*
  return code (``-9`` = killed by signal 9 = SIGKILL); shells report them as
  ``128 + N`` (``137`` = 128 + 9 = SIGKILL);
* a **Windows exception code** (NTSTATUS), e.g. ``3221225477`` = access violation.

The single most common one for registration -- ``-9`` / ``137`` (SIGKILL) -- is
almost always the out-of-memory killer or a cluster memory limit. Estimate first
with :meth:`AntsRegistration.estimate_resources`.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import List, Optional

_MEMORY_TIPS = [
    "Estimate first: reg.estimate_resources(shape=...) to size the job.",
    "Cut memory: use_float=True (halves the real-type size), coarser/fewer "
    "resolution levels, downsample the images, or restrict with a mask.",
    "Give the job more RAM, or reduce ITK threads "
    "(ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS) -- more threads can raise peak memory.",
    "Check the evidence: `dmesg -T | grep -i oom` (Linux OOM killer) or your "
    "scheduler logs (e.g. SLURM `sacct -j <id> --format=State,MaxRSS,ReqMem`).",
]

# signal number -> (name, summary, likely_causes, suggestions)
_SIGNALS = {
    1: (
        "SIGHUP",
        "Hangup -- the controlling terminal closed.",
        ["The terminal/session that launched ANTs exited."],
        ["Run under `nohup`, `tmux`/`screen`, or as a proper batch job."],
    ),
    2: ("SIGINT", "Interrupted (Ctrl-C).", ["You (or a parent process) pressed Ctrl-C."], []),
    4: (
        "SIGILL",
        "Illegal CPU instruction.",
        [
            "The binary was built for a newer instruction set (e.g. AVX/AVX-512) "
            "than this CPU supports -- common with prebuilt binaries on older/VM CPUs."
        ],
        [
            "If you used `commandants install-ants`, the prebuilt build may target a "
            "newer CPU; try a different release asset or build ANTs locally.",
            "Check `/proc/cpuinfo` flags against the build's requirements.",
        ],
    ),
    6: (
        "SIGABRT",
        "Aborted -- an uncaught C++ exception or failed assertion.",
        ["std::bad_alloc (out of memory!)", "an ITK exception", "an assertion failure"],
        [
            "Read stderr for the ITK/`terminate called after throwing` message.",
            "If it mentions bad_alloc/allocate, treat it as OUT OF MEMORY (see below).",
        ]
        + _MEMORY_TIPS,
    ),
    7: (
        "SIGBUS",
        "Bus error -- bad memory or file access.",
        ["A truncated/corrupt input image", "a memory-mapped file on a flaky/full network or disk"],
        ["Verify the input files open cleanly and the disk isn't full."],
    ),
    8: (
        "SIGFPE",
        "Arithmetic exception (e.g. divide-by-zero).",
        ["Degenerate image geometry (zero spacing) or all-constant/empty inputs."],
        ["Check the image headers (spacing/direction) and that inputs aren't empty."],
    ),
    9: (
        "SIGKILL",
        "Forcibly killed -- this signal cannot be caught or handled.",
        [
            "The OS out-of-memory (OOM) killer reclaimed memory (most common)",
            "a cluster/cgroup memory limit was exceeded (SLURM, Docker --memory, k8s)",
            "a wall-clock/time limit, or a manual `kill -9`",
        ],
        ["This is almost always OUT OF MEMORY."] + _MEMORY_TIPS,
    ),
    11: (
        "SIGSEGV",
        "Segmentation fault -- invalid memory access.",
        ["A corrupt or malformed input image", "an incompatible ANTs/ITK build", "an ANTs bug on this input"],
        [
            "Confirm inputs open in SimpleITK/ITK-SNAP and have sane headers.",
            "Try a different ANTs version; minimize to a small reproducer.",
        ],
    ),
    13: ("SIGPIPE", "Broken pipe.", ["A downstream process reading ANTs' output closed early."], []),
    15: (
        "SIGTERM",
        "Terminated -- a graceful kill request.",
        ["A scheduler hit a time/resource limit", "someone/something ran `kill`"],
        ["Check scheduler logs for a time or memory limit; raise the limit."],
    ),
    24: (
        "SIGXCPU",
        "CPU time limit exceeded.",
        ["A scheduler/ulimit CPU-time cap was hit."],
        ["Raise the CPU-time limit or speed up the job (fewer iterations/levels)."],
    ),
    25: (
        "SIGXFSZ",
        "File size limit exceeded.",
        ["An output exceeded a filesystem/ulimit file-size cap."],
        ["Raise the file-size limit; check free disk space."],
    ),
}

# NTSTATUS (Windows) unsigned value -> (name, summary, causes, suggestions)
_WINDOWS = {
    0xC0000005: (
        "STATUS_ACCESS_VIOLATION",
        "Access violation -- invalid memory access (the Windows analogue of a segfault).",
        ["A corrupt/malformed input image", "an incompatible build", "a bug"],
        ["Verify inputs open in SimpleITK/ITK-SNAP; try another ANTs build."],
    ),
    0xC00000FD: ("STATUS_STACK_OVERFLOW", "Stack overflow.", ["Runaway recursion or an extreme input."], []),
    0xC0000017: ("STATUS_NO_MEMORY", "Out of memory.", ["The process could not allocate memory."], _MEMORY_TIPS),
    0xC000013A: ("STATUS_CONTROL_C_EXIT", "Terminated by Ctrl-C.", ["The console received Ctrl-C."], []),
    0xC000001D: (
        "STATUS_ILLEGAL_INSTRUCTION",
        "Illegal instruction -- possibly a CPU-mismatch build.",
        ["The binary targets instructions this CPU lacks."],
        ["Try a different ANTs build for your CPU."],
    ),
    0xC0000094: (
        "STATUS_INTEGER_DIVIDE_BY_ZERO",
        "Integer divide by zero.",
        ["Degenerate geometry or empty inputs."],
        [],
    ),
    0xC0000374: ("STATUS_HEAP_CORRUPTION", "Heap corruption.", ["A memory bug or incompatible libraries."], []),
    0xC0000409: (
        "STATUS_STACK_BUFFER_OVERRUN",
        "Security check / buffer overrun.",
        ["A memory-safety check failed."],
        [],
    ),
}


@dataclass
class ExitCodeExplanation:
    """A decoded explanation of a process exit/return code."""

    code: int
    category: str  # success | failure | signal | windows-exception | usage | unknown
    name: str
    summary: str
    likely_causes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"exit code {self.code}: {self.name} -- {self.summary}"]
        if self.likely_causes:
            lines.append("  likely causes:")
            lines += [f"    - {c}" for c in self.likely_causes]
        if self.suggestions:
            lines.append("  what to try:")
            lines += [f"    - {s}" for s in self.suggestions]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text()


def _signal_explanation(signum: int, code: int, shell: bool = False) -> ExitCodeExplanation:
    entry = _SIGNALS.get(signum)
    origin = f" (shell reported as 128+{signum})" if shell else ""
    if entry is None:
        return ExitCodeExplanation(
            code=code,
            category="signal",
            name=f"signal {signum}",
            summary=f"Terminated by signal {signum}{origin}.",
            likely_causes=["An OS signal terminated the process."],
            suggestions=["Check system/scheduler logs for why it was signalled."],
        )
    name, summary, causes, suggestions = entry
    return ExitCodeExplanation(
        code=code,
        category="signal",
        name=name,
        summary=summary + origin,
        likely_causes=list(causes),
        suggestions=list(suggestions),
    )


def _windows_explanation(value: int, code: int) -> ExitCodeExplanation:
    name, summary, causes, suggestions = _WINDOWS[value]
    return ExitCodeExplanation(
        code=code,
        category="windows-exception",
        name=f"{name} (0x{value:08X})",
        summary=summary,
        likely_causes=list(causes),
        suggestions=list(suggestions),
    )


def explain_exit_code(code: int, system: Optional[str] = None) -> ExitCodeExplanation:
    """Explain a process exit/return code (see the module docstring for the model)."""
    system = system or platform.system()
    unsigned32 = code & 0xFFFFFFFF

    if code == 0:
        return ExitCodeExplanation(
            code=0,
            category="success",
            name="EXIT_SUCCESS",
            summary="Success -- ANTs completed normally.",
        )

    # Windows NTSTATUS exception codes (match signed or unsigned form).
    if code in _WINDOWS:
        return _windows_explanation(code, code)
    if unsigned32 in _WINDOWS:
        return _windows_explanation(unsigned32, code)

    # POSIX signal via Python's negative-return convention.
    if code < 0:
        return _signal_explanation(-code, code)

    # POSIX shell convention: 128 + signal number.
    if 128 < code < 128 + 65:
        return _signal_explanation(code - 128, code, shell=True)

    if code == 1:
        return ExitCodeExplanation(
            code=1,
            category="failure",
            name="EXIT_FAILURE",
            summary="Generic failure -- ANTs raised an error and exited.",
            likely_causes=[
                "An ITK exception (bad/missing input, header/dimension mismatch)",
                "Invalid arguments or an unreadable file/transform",
            ],
            suggestions=[
                "Read stderr -- ANTs prints the actual error there.",
                "Re-run with verbose=True and check the assembled command (reg.to_shell()).",
            ],
        )
    if code == 2:
        return ExitCodeExplanation(
            code=2,
            category="usage",
            name="usage/argument error",
            summary="Usually an argument or usage error.",
            likely_causes=["Malformed/missing arguments."],
            suggestions=["Inspect reg.to_shell(); check flags and file paths."],
        )

    return ExitCodeExplanation(
        code=code,
        category="unknown",
        name=f"exit {code}",
        summary=f"Undocumented exit code {code}.",
        likely_causes=["ANTs does not define a numbered error-code table beyond 0 (success) / 1 (failure)."],
        suggestions=["Read stderr for the underlying message."],
    )


__all__ = ["explain_exit_code", "ExitCodeExplanation"]
