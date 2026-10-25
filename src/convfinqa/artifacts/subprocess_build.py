"""Running one artifact build in a child process, so memory is freed and a crash cannot take the caller down.

The child is `python -m convfinqa.artifacts _run <name>`. It must register the same builders as the
caller, so the command repeats the caller's directories and the modules it imports. The pieces are
separate functions: the command and the environment are plain data, and `run` does the process work.

A build can take hours and has no timeout. Once the child has opened its scratch directory it sends its
own output to `build.log` there; until then (and when it fails to get that far) its error output is
captured here, so a failure always shows its cause.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .errors import BuildError

LOG_TAIL_LINES = 30
"""How many lines of a failed build's log are included in the error message."""

PYTHONHASHSEED_RANGE = 2**32
"""`PYTHONHASHSEED` must be below this, or the interpreter will not start."""


def child_command(
    name: str, *, artifacts_dir: Path, root: Path, chapters_dir: Path | None,
    modules: Sequence[str], force: bool,
) -> list[str]:
    """The command line that builds `name` in this Python environment.

    It runs the hidden `_run` command of `python -m convfinqa.artifacts` with the caller's artifacts
    directory, project root and chapters directory, and one `--import` per builder module, so the
    child registers the same builders (`--import=` with an empty value when there are none, so that
    the child does not fall back to the default modules). `force` is passed on to `_run`.
    """
    cmd = [
        sys.executable, "-m", "convfinqa.artifacts",
        "--artifacts-dir", str(artifacts_dir), "--root", str(root),
    ]
    # The `=` form: argparse takes a separate empty value for the next argument, a command.
    cmd.append(f"--chapters-dir={chapters_dir or ''}")
    for module in modules:
        cmd += ["--import", module]
    if not modules:
        cmd.append("--import=")  # none, explicitly: otherwise the child would fall back to the default modules
    cmd += ["_run", name]
    if force:
        cmd.append("--force")
    return cmd


def child_environment(seed: int, extra_paths: Sequence[str] = ()) -> dict[str, str]:
    """The environment for the child: ours, plus the settings that make its random draws repeatable.

    `PYTHONHASHSEED` fixes string hashing (and so set and dict iteration order). Python accepts only
    0 to 2**32 - 1 for it, so the seed is reduced modulo 2**32. `CUBLAS_WORKSPACE_CONFIG` is the
    setting torch requires for deterministic cuBLAS, and replaces any value already set.
    `extra_paths` go in front of any `PYTHONPATH`.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed % PYTHONHASHSEED_RANGE)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if extra_paths:
        # an empty entry would put the current directory on the child's import path
        env["PYTHONPATH"] = os.pathsep.join([*extra_paths, *filter(None, [env.get("PYTHONPATH")])])
    return env


def _how_it_ended(returncode: int) -> str:
    """Describe a non-zero return code: an exit code, or the signal that killed the process."""
    if returncode >= 0:
        return f"failed (exit code {returncode})"
    try:
        signal_name = signal.Signals(-returncode).name
    except ValueError:
        signal_name = f"signal {-returncode}"
    hint = " (often the out-of-memory killer)" if -returncode == signal.SIGKILL else ""
    return f"was killed by {signal_name}{hint}"


def run(name: str, cmd: list[str], env: dict[str, str], *, cwd: Path, log: Path) -> None:
    """Run the build command and wait for it; on failure raise `BuildError` saying why.

    The child writes its output to `log` itself (see `Registry.execute`), so the caller's terminal
    only sees two progress lines. What the child printed to stderr before that point is captured and
    shown with the error, or echoed after a successful build. The message says whether the child
    exited with a code or was killed by a signal, and ends with the last `LOG_TAIL_LINES` lines of the
    log when there is one.
    """
    print(f"[artifacts] building {name} in a subprocess (log: {log})", file=sys.stderr, flush=True)
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=cwd, stderr=subprocess.PIPE, text=True, errors="replace")
    early = proc.stderr.strip()
    if proc.returncode != 0:
        message = f"building {name!r} {_how_it_ended(proc.returncode)}; log: {log}"
        if log.exists():
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])
            if tail:
                message += f"\n--- end of the log ---\n{tail}"
        if early:
            message += f"\n--- error output before the log started ---\n{early}"
        raise BuildError(message)
    if early:
        print(early, file=sys.stderr, flush=True)
    print(f"[artifacts] built {name} in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
