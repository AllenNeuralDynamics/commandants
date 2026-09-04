"""Tests for live output streaming in run(stream=...) / run(on_line=...).

Uses the Python interpreter as a stand-in subprocess so no ANTs is needed.
"""

from __future__ import annotations

import sys

from commandants.core.runner import AntsCommand


class _PyCommand(AntsCommand):
    """Runs the current Python interpreter; args supplied via extra_args."""

    binary_name = sys.executable

    def _build_args(self):
        return []


def test_stream_captures_and_echoes(capsys):
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('iter 1'); print('iter 2')")
    result = cmd.run(stream=True)
    assert result.returncode == 0
    # Captured for later use...
    assert "iter 1" in result.stdout and "iter 2" in result.stdout
    # ...and echoed live to stdout.
    assert "iter 1" in capsys.readouterr().out


def test_on_line_callback_receives_lines():
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('a'); print('b'); print('c')")
    lines: list[str] = []
    result = cmd.run(on_line=lines.append)  # implies streaming
    assert result.returncode == 0
    joined = "".join(lines)
    assert "a" in joined and "b" in joined and "c" in joined


def test_stream_merges_stderr_into_stdout():
    cmd = _PyCommand()
    cmd.extra_args("-c", "import sys; print('out'); print('err', file=sys.stderr)")
    result = cmd.run(stream=True)
    assert "out" in result.stdout and "err" in result.stdout
    assert result.stderr is None  # merged into stdout when streaming


def test_log_file_by_path(tmp_path):
    log = tmp_path / "run.log"
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('hello'); print('world')")
    result = cmd.run(log_file=str(log))
    assert result.returncode == 0
    contents = log.read_text()
    assert "hello" in contents and "world" in contents


def test_log_file_open_handle_left_open(tmp_path):
    log = tmp_path / "run2.log"
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('kept open')")
    with open(log, "w", encoding="utf-8") as fh:
        cmd.run(log_file=fh)
        assert not fh.closed  # a caller-owned handle stays open
    assert "kept open" in log.read_text()


def test_tee_console_and_file(tmp_path, capsys):
    log = tmp_path / "run3.log"
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('teed')")
    cmd.run(stream=True, log_file=str(log))
    assert "teed" in capsys.readouterr().out  # console
    assert "teed" in log.read_text()  # and file


def test_stream_writes_to_current_sys_stdout(monkeypatch):
    """Jupyter/ipykernel replaces sys.stdout with its own write/flush object.

    We must write to whatever sys.stdout *is at call time* (and flush), not a
    file descriptor captured at import -- otherwise output wouldn't appear in a
    notebook cell. This simulates that redirected stream.
    """
    import sys

    class FakeOutStream:  # stand-in for ipykernel's OutStream
        def __init__(self):
            self.buf = []
            self.flushes = 0

        def write(self, s):
            self.buf.append(s)

        def flush(self):
            self.flushes += 1

    fake = FakeOutStream()
    monkeypatch.setattr(sys, "stdout", fake)

    cmd = _PyCommand()
    cmd.extra_args("-c", "print('notebook-cell-line')")
    cmd.run(stream=True)

    assert "notebook-cell-line" in "".join(fake.buf)
    assert fake.flushes > 0  # flushed per line -> appears live in the cell


def test_bad_callback_does_not_kill_run():
    cmd = _PyCommand()
    cmd.extra_args("-c", "print('x')")

    def boom(_line):
        raise RuntimeError("callback error")

    result = cmd.run(on_line=boom)
    assert result.returncode == 0
