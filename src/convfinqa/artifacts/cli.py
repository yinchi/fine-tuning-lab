"""Command line: `python -m convfinqa.artifacts build|status|refresh`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import ArtifactError
from .registry import ARTIFACTS, DEFAULT_MODULES, ROOT, Registry


def status_tree(reg: Registry, names: list[str]) -> list[str]:
    """Lines showing each artifact's status, with what it is built from indented beneath it."""
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
    p = argparse.ArgumentParser(prog="python -m convfinqa.artifacts", description="Build, inspect and refresh cached artifacts.")
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
        help="Modules to import (can be specified multiple times)"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

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

    # Refresh command
    sub.add_parser(
        "refresh",
        help="delete the freeze folders of chapters that used outdated artifacts"
    )

    # Hidden command for internal use: run a single artifact in this process
    r = sub.add_parser(
        "_run", help=argparse.SUPPRESS
    )  # build one artifact in this process
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
        modules=args.imports or DEFAULT_MODULES,
        chapters_dir=chapters
    )

    # Interact with the registry based on the command-line arguments
    try:
        if args.cmd == "_run":
            reg.execute(args.name, force=args.force, redirect_output=True)
        elif args.cmd == "build":
            names = reg.catalog.targets() if args.all else args.names
            if not names and not args.all:
                p.error("give artifact names or --all")
            reg.catalog.validate(names)
            for n in names:
                reg.ensure(n, allow_heavy=True, force=args.force)
            print(f"[artifacts] {len(names)} target(s) fresh")
        elif args.cmd == "status":
            print("\n".join(status_tree(reg, args.names or reg.catalog.targets())) or "(no targets)")
        elif args.cmd == "refresh":
            removed = reg.records.refresh(args.root / "docs" / "_freeze", reg.status)
            if removed:
                print(f"[artifacts] chapters marked for re-execution: {', '.join(removed)}")
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
