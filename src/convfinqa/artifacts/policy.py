"""When `get` may build an artifact or load a stale one, decided from the environment.

Two switches, both read from environment variables when a `Policy` is made:

- `RUN_HEAVY=1`: a heavy artifact that is stale or missing is built (in a subprocess), not refused.
- `ALLOW_STALE=1`: a heavy artifact that exists but is stale is loaded anyway, with a visible warning.

Light artifacts are always built when stale. A missing heavy artifact is never loaded.
"""

from __future__ import annotations

import enum
import os
import sys
from dataclasses import dataclass

from .handles import Status


class Action(enum.Enum):
    """What `get` should do for an artifact."""

    USE = "use"  # fresh: load it
    BUILD = "build"  # make it fresh, then load it
    USE_STALE = "use_stale"  # load the old result, with a warning
    REFUSE = "refuse"  # raise, naming the build command


@dataclass(frozen=True)
class Policy:
    run_heavy: bool = False
    allow_stale: bool = False

    @classmethod
    def from_env(cls) -> Policy:
        """The policy the environment variables `RUN_HEAVY` and `ALLOW_STALE` ask for (each must be `"1"`)."""
        return cls(
            run_heavy=os.environ.get("RUN_HEAVY") == "1",
            allow_stale=os.environ.get("ALLOW_STALE") == "1",
        )

    def decide(self, *, heavy: bool, status: Status) -> Action:
        """What to do with an artifact that is `heavy` or light and has the given `status`."""
        if status.fresh:
            return Action.USE
        if not heavy or self.run_heavy:
            return Action.BUILD
        if status.kind == "stale" and self.allow_stale:
            return Action.USE_STALE
        return Action.REFUSE


def stale_message(name: str, status: Status) -> str:
    """The error text for a heavy artifact that is missing or stale and may not be built."""
    detail = "is missing" if status.kind == "missing" else f"is stale ({status.reason})"
    message = (
        f"Heavy artifact {name!r} {detail}. Build it with:\n"
        f"    uv run python -m convfinqa.artifacts build {name}\n"
        "(or run ./build.sh, or render with RUN_HEAVY=1)."
    )
    if status.kind == "stale":
        message += " ALLOW_STALE=1 would load the old result with a warning."
    return message


def show_stale_warning(name: str, reason: str) -> None:
    """Show a warning for a stale artifact that is being used anyway.

    Uses `IPython.display` if available to show a styled warning message in notebooks. Otherwise,
    prints to `stderr`.
    """
    msg = f"STALE artifact {name!r} ({reason}). Showing the old result; rebuild with ./build.sh."
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            display(HTML(
                '<div style="border:2px solid #c0392b;background:#fdecea;color:#7b241c;padding:6px 10px;'
                f'margin:4px 0;font-weight:600">&#9888; {msg}</div>'
            ))
            return
    except ImportError:
        pass
    print(f"WARNING: {msg}", file=sys.stderr)
