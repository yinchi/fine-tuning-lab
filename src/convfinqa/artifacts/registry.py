"""The registry: joins catalog, store and chapter records to check freshness, build, and answer `get`.

The catalog knows what artifacts exist, the store what is on disk, and the records what each chapter
used. The registry decides whether an artifact is fresh (`status`), makes it fresh (`ensure`, which
builds in this process with `execute` or in a subprocess with `_spawn`), and serves chapters (`get`).

Module-level `artifact` and `get` work on the default registry, `DEFAULT`.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

from . import subprocess_build
from .catalog import Catalog, active_catalog
from .errors import BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .hashing import hash_outputs, is_framework, sha256_json
from .policy import Action, Policy, refusal_message, show_stale_warning
from .provenance import drift_notes, provenance
from .records import ChapterRecords
from .seeding import preserved_rng_state, seed_everything
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
    """Return the decorator that registers a builder, with the catalog that is loading builders.

    Outside a load (a chapter cell running in a kernel) that is the default registry's catalog.
    """
    return (active_catalog() or DEFAULT.catalog).artifact(
        template, deps=deps, files=files, heavy=heavy, version=version, seed=seed
    )


class Registry:
    """Joins the catalog (what exists), the store (what is on disk) and the chapter records.

    It decides whether an artifact is fresh, builds the ones that are not (in this process or in a
    subprocess), and answers `get` for chapters. Ask the parts directly for anything that is theirs:
    `registry.catalog` to resolve names, list targets or validate the graph, `registry.store` for
    handles and paths, `registry.records` for what chapters used.
    """

    def __init__(
        self, artifacts_dir: Path = ARTIFACTS, *, root: Path = ROOT, modules=(), extra_paths=(),
        chapters_dir: Path | None = None,
    ):
        """Initialize the registry with the given directories and modules.

        Args:
            artifacts_dir: Where built artifacts are stored.
            root: The project root; declared input files are resolved against it.
            modules: Modules to import to register builders (see `Catalog`).
            extra_paths: Directories put on `PYTHONPATH` of build subprocesses, so that they can import
                modules that are not installed (used by tests).
            chapters_dir: Where the chapters' `.qmd` files are scanned for builders, if anywhere.
        """
        self.dir = Path(artifacts_dir)
        """The artifacts directory."""

        self.root = Path(root)
        """The project root."""

        self.extra_paths = [str(p) for p in extra_paths]
        """Extra `PYTHONPATH` entries for build subprocesses."""

        self.catalog = Catalog(modules, chapters_dir)
        """What artifacts exist."""

        self.store = Store(self.dir)
        """What is on disk."""

        self.records = ChapterRecords(self.dir / "_chapters")
        """What chapters used."""

    def status(self, name: str) -> Status:
        """Whether `name` is fresh, stale (and why), or missing.

        The checks run in this order, and the first to fail decides: never built (missing); the
        builder's `version` changed; a declared input file is missing; the config differs (name
        fields, seed, input file contents); the code hash differs; the list of dependencies differs;
        then each dependency in turn, which must itself be fresh and have the same output hash as when
        this artifact was built.

        Dependencies are checked recursively, so a stale artifact deep in the
        graph makes everything built from it stale. A fresh result carries `notes` about drift in the
        environment (see `provenance.drift_notes`), which never make it stale.
        """
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

    def ensure(self, name: str, *, allow_heavy: bool, force: bool = False) -> Handle:
        """Make `name` fresh and return its handle, building whatever is stale or missing.

        The dependencies are ensured first, in dependency order, so one that is rebuilt with identical
        output leaves `name` fresh (early cutoff). A light artifact is built in this process
        (`execute`), a heavy one in a subprocess (`_spawn`) and only if `allow_heavy` is set;
        otherwise `StaleArtifactError` is raised, also when the stale heavy artifact is a dependency.
        `force` rebuilds `name` itself even if it is fresh; it is not passed on to its dependencies.
        """
        spec, params = self.catalog.resolve(name)
        for dep in spec.dep_names(params):
            self.ensure(dep, allow_heavy=allow_heavy)
        st = self.status(name)
        if st.fresh and not force:
            return self.store.handle(name)
        if spec.heavy:
            if not allow_heavy:
                raise StaleArtifactError(refusal_message(name, st))
            self._spawn(name, force=force)
        else:
            self.execute(name, force=force)
        return self.store.handle(name)

    def _spawn(self, name: str, *, force: bool) -> None:
        """Build `name` in a child process (`python -m convfinqa.artifacts _run <name>`) and wait for it.

        The child gets this registry's directories, builder modules and chapters directory, so it
        registers the same builders, and an environment that pins its random seed
        (`subprocess_build.child_environment`). Its output goes to the build log in the artifact's
        scratch directory; this process only prints progress. The child takes the artifact's lock itself.
        Raises `BuildError`, with the end of the log, if the child fails.
        """
        spec, params = self.catalog.resolve(name)
        cmd = subprocess_build.child_command(
            name, artifacts_dir=self.dir, root=self.root, chapters_dir=self.catalog.chapters_dir,
            modules=self.catalog.modules, force=force,
        )
        env = subprocess_build.child_environment(spec.build_seed(params), self.extra_paths)
        subprocess_build.run(name, cmd, env, cwd=self.root, log=self.store.log_path(name))

    def execute(self, name: str, *, force: bool = False, redirect_output: bool = False) -> None:
        """Build `name` in this process: run its builder and store the result.

        Normally reached through `ensure` for light artifacts, and by the build subprocess for heavy
        ones. It holds the artifact's lock for the whole build and, once it has the lock, checks again
        that the artifact needs building (another process may have built it while this one waited),
        unless `force`.

        The steps: collect the handles of the dependencies (all must be fresh, else `BuildError`);
        open a fresh scratch directory; check that the declared input files exist (else `BuildError`);
        call the builder as `fn(out, deps, [files], **fields)` with the random state seeded from the
        build seed (and restored afterwards) and the working directory set to the scratch directory;
        then commit `node.json` with the config, the dependencies' output hashes, the code hash, the
        provenance and the timing. `dynamic_modules` lists `convfinqa` modules the builder loaded that
        were not imported before; it adds to the static import analysis, which already covers imports
        written in the builder.

        A builder that raises leaves no `node.json`, so the artifact still counts as missing or stale,
        and its scratch directory, with the log, is kept for inspection.

        `redirect_output` points this process's stdout and stderr at the build log for good, so use it
        only in a process dedicated to the build (the build subprocess does).
        """
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
            code_before = spec.code_hash()
            config, dynamic, started = self._call_builder(name, spec, params, out, dep_handles)
            if spec.code_hash() != code_before:
                # The hash recorded below is read from the files now; a result built from the old code
                # would be recorded as fresh for the new code.
                raise BuildError(
                    f"the code of {name!r} (the builder or a `convfinqa` module it imports) changed while it "
                    "was building; the result is not recorded, so build it again"
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

    def _call_builder(self, name: str, spec, params: dict, out: Path, dep_handles: dict) -> tuple[dict, list, float]:
        """Run the builder of `name` into `out`, as `execute` and `_provisional` both do.

        Checks the declared input files, calls the builder as `fn(out, deps, [files], **fields)` with the
        random state seeded from the build seed (and restored afterwards) and the working directory set
        to `out`, and finds the `convfinqa` modules it loaded beyond the static import analysis.

        Returns:
            The config, the sorted dynamic modules, and the start time.
        """
        config = spec.config(params, self.root)
        if missing := [rel for rel, h in config["files"].items() if h is None]:
            raise BuildError(f"cannot build {name!r}: declared input file(s) missing: {missing}")
        # The builder sees the very hashes recorded in the config, read once.
        files = Files({rel: InputFile(self.root / rel, h) for rel, h in config["files"].items()})
        args = [out, Deps(dep_handles)] + ([files] if spec.files else [])
        before = set(sys.modules)
        started = time.time()
        with preserved_rng_state(), contextlib.chdir(out):
            seed_everything(config["seed"])
            spec.fn(*args, **params)
        dynamic = sorted(
            m for m in set(sys.modules) - before
            if (m == "convfinqa" or m.startswith("convfinqa.")) and not is_framework(m)
        )
        return config, dynamic, started

    def _provisional(self, name: str) -> Handle:
        """Build the light artifact `name` for this use only, from inputs that may be stale (`ALLOW_STALE`).

        Nothing is recorded: no `node.json` is written and the artifact on disk is left as it is, so it
        stays stale or missing and is rebuilt properly once its inputs are. The result is in a temporary
        directory, removed when the process ends, and the handle is marked `stale`. A heavy input that
        is stale is used as it is; one that was never built, or a light input that cannot be made
        this way, raises `StaleArtifactError`.
        """
        spec, params = self.catalog.resolve(name)
        dep_handles = {}
        for d in spec.dep_names(params):
            dep_spec, _ = self.catalog.resolve(d)
            dep_status = self.status(d)
            if dep_status.fresh:
                dep_handles[d] = self.store.handle(d)
            elif not dep_spec.heavy:
                dep_handles[d] = self._provisional(d)
            elif dep_status.kind == "stale":
                dep_handles[d] = self.store.handle(d, stale=True)
            else:
                raise StaleArtifactError(refusal_message(d, dep_status))
        out = Path(tempfile.mkdtemp(prefix=f"{name}.provisional-"))
        atexit.register(shutil.rmtree, out, ignore_errors=True)
        self._call_builder(name, spec, params, out, dep_handles)
        outputs = hash_outputs(out)
        return Handle(name, out, sha256_json(outputs), True, frozenset(outputs))

    def _refresh_light_inputs(self, name: str) -> None:
        """Rebuild, in this process, the stale or missing light artifacts that `name` is built from.

        This is done before judging `name`, so that it is judged against up-to-date inputs. A light
        input that is rebuilt with identical output leaves `name` fresh (early cutoff), and a heavy
        artifact is not refused over an input that rebuilding would have settled. Anything whose
        rebuild needs a heavy build that is not allowed is left as it is.
        """
        spec, params = self.catalog.resolve(name)
        for dep in spec.dep_names(params):
            self._refresh_light_inputs(dep)  # its own inputs first
            if not self.catalog.resolve(dep)[0].heavy and not self.status(dep).fresh:
                try:
                    self.ensure(dep, allow_heavy=False)
                except StaleArtifactError:
                    pass  # blocked by a stale heavy input of its own

    def get(self, name: str, policy: Policy | None = None) -> Handle:
        """Return a handle to the artifact `name`, building it if the policy allows (see `policy.py`).

        `policy` defaults to the one the environment asks for (`RUN_HEAVY`, `ALLOW_STALE`). The light
        artifacts `name` is built from are brought up to date first. If `name` is light and cannot be
        rebuilt because a heavy input is stale, `allow_stale` builds it for this use from the stale input
        (`_provisional`, nothing is saved); failing that, it loads the artifact's old result, like a heavy
        one. Either handle has `stale=True`.

        The use is recorded for the running chapter (`records`), so that the chapter is executed again
        when the artifact changes. Raises `StaleArtifactError` if the policy refuses, `BuildError` if
        a build fails, and `ArtifactError` if `name` is not a registered artifact.
        """
        policy = policy or Policy.from_env()
        spec, _ = self.catalog.resolve(name)
        self._refresh_light_inputs(name)
        st = self.status(name)
        match policy.decide(heavy=spec.heavy, status=st):
            case Action.USE:
                for note in st.notes:
                    warnings.warn(f"artifact {name}: {note}", stacklevel=2)
                h = self.store.handle(name)
            case Action.BUILD:
                try:
                    h = self.ensure(name, allow_heavy=policy.run_heavy)
                except StaleArtifactError:
                    # Only a light artifact gets here: its rebuild needs a heavy input that is not allowed.
                    if not policy.allow_stale:
                        raise
                    try:
                        h = self._provisional(name)
                        show_stale_warning(name, "built for this use from stale inputs, and not saved")
                    except StaleArtifactError:
                        if st.kind != "stale":
                            raise
                        h = self.store.handle(name, stale=True)
                        show_stale_warning(name, st.reason)
            case Action.USE_STALE:
                h = self.store.handle(name, stale=True)
                show_stale_warning(name, st.reason)
            case Action.REFUSE:
                raise StaleArtifactError(refusal_message(name, st))
        self.records.note_use(h)
        return h


DEFAULT = Registry(modules=DEFAULT_MODULES, chapters_dir=ROOT / "docs")
"""The default registry, with the project's directories. The module-level `get` and `artifact` use it."""


def get(name: str) -> Handle:
    """Return a handle to the artifact `name` from the default registry: what chapters call.

    The environment decides what may be built (`RUN_HEAVY`) or loaded stale (`ALLOW_STALE`); see
    `Registry.get` for the rules and the errors.
    """
    return DEFAULT.get(name)
