"""Command line: `python -m convfinqa.artifacts build|status|refresh`.

    python -m convfinqa.artifacts status                 # what is fresh, stale or missing, and why
    python -m convfinqa.artifacts build --all            # build everything the chapters ask for
    python -m convfinqa.artifacts build sft_dsl_0        # build one artifact, and what it needs
    python -m convfinqa.artifacts refresh                # mark chapters that used outdated artifacts

`build` is the one place heavy artifacts are built on request: it allows subprocess builds, unlike a
chapter's `get`. The options before the command (`--artifacts-dir`, `--root`, `--chapters-dir`,
`--import`) describe the project and are meant for tests and the build subprocesses; by default they
describe this checkout. A `_run <name>` command, hidden from the help, builds one artifact in the
calling process: it is what a heavy build's subprocess runs.

Exit codes: 0 on success, 1 when an `ArtifactError` is reported (printed as `error: ...`), 2 for a
mistake in the command line. `status --check` exits 1 if any artifact it shows is not fresh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chapters import scan_chapters
from .errors import ArtifactError
from .registry import ARTIFACTS, DEFAULT_MODULES, ROOT, Registry


def status_tree(reg: Registry, names: list[str]) -> list[str]:
    """Lines showing each named artifact's status, with what it is built from indented beneath it.

    Each line is `name  [kind: reason]`, followed by `note:` lines for environment drift. An artifact
    that several others use is shown under each of them.
    """
    lines: list[str] = []

    def show(name: str, depth: int) -> None:
        st = reg.status(name)
        label = st.kind + (f": {st.reason}" if st.reason else "")
        lines.append(f"{'  ' * depth}{name}  [{label}]")
        for note in st.notes:
            lines.append(f"{'  ' * depth}  note: {note}")
        for dep in reg.catalog.dependencies(name):
            show(dep, depth + 1)

    for name in names:
        show(name, 0)
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the command line (see the module docstring) and return its exit code.

    `argv` defaults to `sys.argv[1:]`. A mistake in the command line makes `argparse` raise
    `SystemExit(2)`; an `ArtifactError` is printed as `error: ...` and gives exit code 1.
    """
    p = argparse.ArgumentParser(
        prog="python -m convfinqa.artifacts",
        description="Build, inspect and refresh cached artifacts."
    )

    p.add_argument(
        "--artifacts-dir", type=Path, default=ARTIFACTS,
        help="The directory to store artifacts in (default read from registry module)",
    )
    p.add_argument(
        "--root", type=Path, default=ROOT,
        help="The root directory of the project (default read from registry module)",
    )
    p.add_argument(
        "--chapters-dir", default=None,
        help="Where the chapters' .qmd files are (default: ROOT/docs; give an empty value for none)",
    )
    p.add_argument(
        "--import", dest="imports", action="append", default=[], metavar="MODULE",
        help="Modules to import to register builders, repeatable (default: convfinqa.builders; "
             "give an empty value for none)",
    )

    # The metavar lists the commands by hand, because argparse would put `_run` in the usage line.
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{build,status,refresh}")

    # Build command
    b = sub.add_parser(
        "build",
        help="Build artifacts (heavy ones in subprocesses) and anything stale they need"
    )
    b.add_argument(
        "names", nargs="*",
        help="The artifact names to build"
    )
    b.add_argument(
        "--all", action="store_true",
        help="Build every artifact the chapters ask for with get(\"name\")"
    )
    b.add_argument(
        "--force", action="store_true",
        help="Rebuild the named artifacts even if fresh"
    )

    # Status command
    s = sub.add_parser("status", help="Show a fresh/stale/missing tree")
    s.add_argument(
        "names", nargs="*",
        help="The artifact names to check status for (default: every target)"
    )
    s.add_argument(
        "--check", action="store_true",
        help="Exit with code 1 if any artifact shown is not fresh (for scripts)"
    )

    # Refresh command
    sub.add_parser(
        "refresh",
        help="Delete the freeze folders of chapters that used outdated artifacts"
    )

    # Hidden command for internal use: run a single artifact in this process
    # No `help=`: that is what keeps a command out of the list
    # (`help=SUPPRESS` would list it, with the text "==SUPPRESS==").
    r = sub.add_parser("_run")
    r.add_argument("name")
    r.add_argument("--force", action="store_true")

    # Parse command-line arguments
    args = p.parse_args(argv)

    # Initialize the registry with the specified artifacts directory, root, and imported modules
    chapters = (
        args.root / "docs" if args.chapters_dir is None
        else (Path(args.chapters_dir) if args.chapters_dir else None)
    )
    reg = Registry(
        args.artifacts_dir,
        root=args.root,
        modules=[m for m in args.imports if m] if args.imports else DEFAULT_MODULES,
        chapters_dir=chapters
    )

    # Interact with the registry based on the command-line arguments
    try:
        if args.cmd == "_run":
            reg.execute(args.name, force=args.force, redirect_output=True)
        elif args.cmd == "build":
            if args.all and args.names:
                p.error("give artifact names or --all, not both")
            names = reg.catalog.targets() if args.all else args.names
            if not names and not args.all:
                p.error("give artifact names or --all")
            reg.catalog.validate(names)
            for n in names:
                reg.ensure(n, allow_heavy=True, force=args.force)
            print(f"[artifacts] {len(names)} target(s) fresh")
        elif args.cmd == "status":
            names = args.names or reg.catalog.targets()
            print("\n".join(status_tree(reg, names)) or "(no targets)")
            if args.check and not all(reg.status(n).fresh for n in names):
                return 1
        elif args.cmd == "refresh":
            # Quarto keeps a chapter's saved results in docs/_freeze/<chapter>/
            freeze = args.root / "docs" / "_freeze"
            # A catalog that does not load (a chapter mid-edit) must stop here: `status` would raise for
            # every name, and every chapter would count as outdated and lose its saved results.
            reg.catalog.validate(reg.catalog.targets())
            removed = reg.records.refresh(freeze, reg.status)
            if chapters:
                scan = scan_chapters(chapters)
                asking = sorted({Path(where.rsplit(":", 1)[0]).stem for _, where in scan.gets}
                                | {b.chapter for b in scan.builders})
                removed += reg.records.drop_unrecorded(freeze, asking)
            if removed:
                print(f"[artifacts] chapters marked for re-execution: {', '.join(removed)}")
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
