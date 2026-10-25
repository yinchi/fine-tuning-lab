"""The registry: the coordinator. It joins the catalog (what exists), the store (what is on disk) and the
records (what chapters used), and decides freshness, builds (in process or in a subprocess), and `get`."""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

from . import subprocess_build
from .catalog import Catalog, active_catalog
from .errors import BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .hashing import is_framework, sha256_file
from .policy import Action, Policy, show_stale_warning, stale_message
from .provenance import drift_notes, provenance
from .records import ChapterRecords
from .seeding import chdir, preserved_rng_state, seed_everything
from .store import Store

ROOT = Path(os.environ.get("CONVFINQA_ROOT") or Path(__file__).resolve().parents[3])
"""The project root.

Read from the environment variable `CONVFINQA_ROOT` if set, otherwise defaults to the
great-grandparent of the current file's directory.
"""

ARTIFACTS = Path(os.environ.get("CONVFINQA_ARTIFACTS") or ROOT / "artifacts")
"""The directory holding built artifacts.

Read from `CONVFINQA_ARTIFACTS` if set, otherwise `ROOT / "artifacts"`.
"""

DEFAULT_MODULES = ("convfinqa.builders",)
"""Modules the registry imports to register builders.

Importing these modules runs their `@artifact` decorators.
"""

def artifact(template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
    """Register a builder with the catalog that is loading (the default one outside `Catalog.load`)."""
    return (active_catalog() or DEFAULT.catalog).artifact(
        template, deps=deps, files=files, heavy=heavy, version=version, seed=seed
    )


class Registry:
    """Joins the catalog (what exists), the store (what is on disk) and the chapter records.

    It decides whether an artifact is fresh, builds the ones that are not (in this process or in a
    subprocess), and answers `get` for chapters. The catalog is `registry.catalog`; ask it directly
    to resolve names, list targets or validate the graph.
    """

    def __init__(
        self, artifacts_dir: Path = ARTIFACTS, *, root: Path = ROOT, modules=(), extra_paths=(),
        chapters_dir: Path | None = None,
    ):
        self.dir = Path(artifacts_dir)
        self.root = Path(root)
        self.extra_paths = [str(p) for p in extra_paths]
        self.catalog = Catalog(modules, chapters_dir)
        self.store = Store(self.dir)
        self.records = ChapterRecords(self.dir / "_chapters")

    # freshness ..................................................................................

    def status(self, name: str) -> Status:
        """Whether `name` is fresh, stale (and why), or missing; dependencies are checked recursively."""
        spec, params = self.catalog.resolve(name)
        if (node := self.store.read_node(name)) is None:
            return Status("missing", "not built")
        if node.get("version") != spec.version:
            return Status("stale", f"version changed ({node.get('version')} -> {spec.version})")

        config = spec.config(params, self.root)
        if any(h is None for h in config["files"].values()):
            missing = [f for f, h in config["files"].items() if h is None]
            return Status("stale", f"input file missing: {missing}")
        if node.get("config") != config:
            changed = [k for k in config if node.get("config", {}).get(k) != config[k]]
            return Status("stale", f"config changed ({', '.join(changed)})")
        if node.get("code_hash") != spec.code_hash(node.get("dynamic_modules", [])):
            return Status("stale", "code changed")

        deps = spec.dep_names(params)
        if sorted(node.get("deps", {})) != sorted(deps):
            return Status("stale", "dependencies changed")
        for d in deps:
            ds = self.status(d)
            if not ds.fresh:
                return Status("stale", f"dependency {d} is {ds.kind}")
            if ds.output_hash != node["deps"][d]:
                return Status("stale", f"dependency {d} was rebuilt")

        return Status("fresh", output_hash=node["output_hash"], notes=tuple(drift_notes(node)))

    # building ...................................................................................

    def ensure(self, name: str, *, allow_heavy: bool, force: bool = False) -> Handle:
        """Make `name` fresh (building it and its stale dependencies) and return its handle."""
        spec, params = self.catalog.resolve(name)
        for dep in spec.dep_names(params):
            self.ensure(dep, allow_heavy=allow_heavy)
        st = self.status(name)
        if st.fresh and not force:
            return self.store.handle(name)
        if spec.heavy:
            if not allow_heavy:
                raise StaleArtifactError(stale_message(name, st))
            self._spawn(name, force=force)
        else:
            self.execute(name, force=force)
        return self.store.handle(name)

    def _spawn(self, name: str, *, force: bool) -> None:
        spec, params = self.catalog.resolve(name)
        cmd = subprocess_build.child_command(
            name, artifacts_dir=self.dir, root=self.root, chapters_dir=self.catalog.chapters_dir,
            modules=self.catalog.modules, force=force,
        )
        env = subprocess_build.child_environment(spec.build_seed(params), self.extra_paths)
        subprocess_build.run(name, cmd, env, cwd=self.root, log=self.store.log_path(name))

    def execute(self, name: str, *, force: bool = False, redirect_output: bool = False) -> None:
        """Run the builder in this process. Takes the lock, then re-checks freshness."""
        spec, params = self.catalog.resolve(name)
        with self.store.lock(name):
            if not force and self.status(name).fresh:
                return  # another process built it while we waited
            dep_handles = {}
            for d in spec.dep_names(params):
                if not self.status(d).fresh:
                    raise BuildError(f"cannot build {name!r}: dependency {d!r} is not fresh")
                dep_handles[d] = self.store.handle(d)
            out = self.store.scratch(name)
            if redirect_output:
                fd = os.open(self.store.log_path(name), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
                os.dup2(fd, 1)
                os.dup2(fd, 2)
            files = Files({
                rel: InputFile(self.root / rel, sha256_file(self.root / rel)) for rel in spec.files
            })
            args = [out, Deps(dep_handles)] + ([files] if spec.files else [])
            config = spec.config(params, self.root)
            before = set(sys.modules)
            started = time.time()
            with preserved_rng_state(), chdir(out):
                seed_everything(config["seed"])
                spec.fn(*args, **params)
            dynamic = sorted(
                m for m in set(sys.modules) - before
                if (m == "convfinqa" or m.startswith("convfinqa.")) and not is_framework(m)
            )
            self.store.commit(name, {
                "name": name,
                "version": spec.version,
                "config": config,
                "deps": {d: h.hash for d, h in dep_handles.items()},
                "code_hash": spec.code_hash(dynamic),
                "dynamic_modules": dynamic,
                **provenance(self.root),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "duration_s": round(time.time() - started, 1),
            })

    # what chapters call .........................................................................

    def get(self, name: str, policy: Policy | None = None) -> Handle:
        """Return the artifact `name`, building it if the policy allows (see `policy.py`).

        `policy` defaults to the one the environment asks for (`RUN_HEAVY`, `ALLOW_STALE`).
        """
        policy = policy or Policy.from_env()
        spec, _ = self.catalog.resolve(name)
        st = self.status(name)
        match policy.decide(heavy=spec.heavy, status=st):
            case Action.USE:
                for note in st.notes:
                    warnings.warn(f"artifact {name}: {note}", stacklevel=2)
                h = self.store.handle(name)
            case Action.BUILD:
                h = self.ensure(name, allow_heavy=policy.run_heavy)
            case Action.USE_STALE:
                h = self.store.handle(name, stale=True)
                show_stale_warning(name, st.reason)
            case Action.REFUSE:
                raise StaleArtifactError(stale_message(name, st))
        self.records.note_use(h)
        return h


DEFAULT = Registry(modules=DEFAULT_MODULES, chapters_dir=ROOT / "docs")
"""The default registry instance used by the `get` function."""


def get(name: str) -> Handle:
    """Get a handle to an artifact by name.
    
    Uses the default registry instance. If the rules allow, also builds the artifact if it is stale
    or missing.  If the artifact cannot be found or built, an `ArtifactError` is raised.
    """
    return DEFAULT.get(name)
