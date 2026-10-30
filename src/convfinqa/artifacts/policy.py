"""When `get` may build an artifact or load a stale one.

Two switches, which `Policy.from_env()` reads from environment variables (and `get` calls when it is
not given a policy):

- `RUN_HEAVY=1`: a heavy artifact that is stale or missing is built (in a subprocess), not refused.
- `ALLOW_STALE=1`: a heavy artifact that exists but is stale is loaded anyway, with a visible warning.

A light artifact that is stale or missing is built in this process, after the light artifacts it is
built from. If that needs a heavy build that is not allowed (a heavy input is stale and `RUN_HEAVY` is
not set), `ALLOW_STALE` builds it for this use only, from the stale input, with a visible warning and
nothing saved; if that cannot be done either, it loads the light artifact's old result (which must
exist), and otherwise `get` refuses. A missing heavy artifact is never loaded, and when both switches
are set `RUN_HEAVY` wins: the artifact is rebuilt.
"""

from __future__ import annotations

import enum
import html
import os
import sys
from dataclasses import dataclass

from .handles import Status


class Action(enum.Enum):
    """What `get` should do for an artifact."""

    USE = "use"
    """It is fresh: load it."""

    BUILD = "build"
    """Make it fresh (in this process if light, in a subprocess if heavy), then load it."""

    USE_STALE = "use_stale"
    """Load the old result and show a warning."""

    REFUSE = "refuse"
    """Raise `StaleArtifactError`, naming the build command."""


@dataclass(frozen=True)
class Policy:
    """The rules for when `get` may build or load stale. The default, `Policy()`, is the strict one."""

    run_heavy: bool = False
    """Build a stale or missing heavy artifact instead of refusing."""

    allow_stale: bool = False
    """Load a stale heavy artifact (one that exists) instead of refusing."""

    @classmethod
    def from_env(cls) -> Policy:
        """The policy the environment variables `RUN_HEAVY` and `ALLOW_STALE` ask for.

        A switch is on only when its variable is exactly `"1"`.
        """
        return cls(
            run_heavy=os.environ.get("RUN_HEAVY") == "1",
            allow_stale=os.environ.get("ALLOW_STALE") == "1",
        )

    def decide(self, *, heavy: bool, status: Status) -> Action:
        """What to do with an artifact of the given weight and status. In order:

        1. fresh: `USE`;
        2. light, or `run_heavy` is set: `BUILD`;
        3. stale and `allow_stale` is set: `USE_STALE`;
        4. anything else (a heavy artifact that is stale or missing): `REFUSE`.
        """
        if status.fresh:
            return Action.USE
        if not heavy or self.run_heavy:
            return Action.BUILD
        if status.kind == "stale" and self.allow_stale:
            return Action.USE_STALE
        return Action.REFUSE


def refusal_message(name: str, status: Status) -> str:
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
                f'margin:4px 0;font-weight:600">&#9888; {html.escape(msg)}</div>'
            ))
            return
    except ImportError:
        pass
    print(f"WARNING: {msg}", file=sys.stderr)
