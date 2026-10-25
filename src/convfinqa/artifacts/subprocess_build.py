"""Running one artifact build in a child process, so memory is freed and a crash cannot take the caller down.

The child is `python -m convfinqa.artifacts _run <name>`. It must register the same builders as the
caller, so the command repeats the caller's directories and the modules it imports. The pieces are
separate functions: the command and the environment are plain data, and `run` does the process work.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .errors import BuildError

LOG_TAIL_LINES = 30
"""How many lines of a failed build's log are included in the error message."""


def child_command(
    name: str, *, artifacts_dir: Path, root: Path, chapters_dir: Path | None, modules: Sequence[str], force: bool,
) -> list[str]:
    """The command line that builds `name` in this Python environment."""
    cmd = [sys.executable, "-m", "convfinqa.artifacts", "--artifacts-dir", str(artifacts_dir), "--root", str(root)]
    cmd.append(f"--chapters-dir={chapters_dir or ''}")  # `=` form: argparse treats a separate empty value as a command
    for module in modules:
        cmd += ["--import", module]
    cmd += ["_run", name]
    if force:
        cmd.append("--force")
    return cmd


def child_environment(seed: int, extra_paths: Sequence[str] = ()) -> dict[str, str]:
    """The environment for the child: ours, plus the settings that make its random draws repeatable.

    `PYTHONHASHSEED` fixes string hashing (and so set and dict iteration order); `CUBLAS_WORKSPACE_CONFIG`
    is the setting torch requires for deterministic cuBLAS. `extra_paths` go in front of `PYTHONPATH`.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if extra_paths:
        env["PYTHONPATH"] = os.pathsep.join([*extra_paths, env.get("PYTHONPATH", "")])
    return env


def run(name: str, cmd: list[str], env: dict[str, str], *, cwd: Path, log: Path) -> None:
    """Run the build command; on failure raise `BuildError` with the end of the child's log.

    The child writes its output to `log` itself (see `Registry.execute`), so the caller's terminal
    only sees the two progress lines.
    """
    print(f"[artifacts] building {name} in a subprocess (log: {log})", file=sys.stderr, flush=True)
    started = time.time()
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    if proc.returncode != 0:
        tail = ""
        if log.exists():
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:])
        raise BuildError(f"building {name!r} failed (exit code {proc.returncode}); log: {log}\n{tail}")
    print(f"[artifacts] built {name} in {time.time() - started:.0f}s", file=sys.stderr, flush=True)
