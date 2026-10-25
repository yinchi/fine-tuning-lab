"""The registry: builder registration, freshness, building (in process or in a subprocess), and `get`."""

from __future__ import annotations

import contextlib
import enum
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .chapters import check_pure, compile_builder, scan_chapters
from .errors import ArtifactError, BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .hashing import LOG_NAME, hash_outputs, is_framework, library_versions, sha256_file, sha256_json
from .records import ChapterRecords
from . import subprocess_build
from .seeding import chdir, preserved_rng_state, seed_everything
from .spec import NAME, SAMPLE, Spec, make_spec

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

_active: Registry | None = None


class LoadState(enum.Enum):
    """Whether a registry has read its builders (from modules and chapters) yet."""

    UNLOADED = "unloaded"  # not loaded, or a load failed (so the next use tries again)
    LOADING = "loading"  # builders are being registered; `@artifact` must register, not look up
    LOADED = "loaded"


def artifact(template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
    """Register a builder with the active registry (the default one outside `Registry.load`)."""
    return (_active or DEFAULT).artifact(
        template, deps=deps, files=files, heavy=heavy, version=version, seed=seed
    )


class Registry:
    def __init__(
        self, artifacts_dir: Path = ARTIFACTS, *, root: Path = ROOT, modules=(), extra_paths=(),
        chapters_dir: Path | None = None,
    ):
        self.dir = Path(artifacts_dir)
        self.root = Path(root)
        self.modules = list(modules)
        self.extra_paths = [str(p) for p in extra_paths]
        self.chapters_dir = Path(chapters_dir) if chapters_dir else None
        self.specs: dict[str, Spec] = {}
        self.target_locations: dict[str, str] = {}  # literal get("name") calls found in chapters
        self.unparsed_cells: list[str] = []
        self.load_state = LoadState.UNLOADED
        self._scan_signature: tuple = ()  # (name, mtime, size) of the chapter files last scanned
        self.records = ChapterRecords(self.dir / "_chapters")

    # registration ...............................................................................

    def artifact(self, template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
        """Decorator registering a builder.
        
        In a chapter cell that the scan already registered, it is a no-op.
        """
        if self.load_state is not LoadState.LOADING:
            self._ensure_loaded()
            existing = self.specs.get(template)
            if existing is not None and existing.chapter is not None:
                return lambda fn: fn  # the chapter's scanned definition is the registered one

        def decorate(fn: Callable) -> Callable:
            self._register(
                template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed
            )
            return fn

        return decorate

    def _register(
        self, template: str, fn: Callable, *, deps, files, heavy, version, seed,
        chapter: str | None = None, source: str | None = None, location: str = "",
    ) -> None:
        spec = make_spec(
            template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed,
            chapter=chapter, source=source, location=location,
        )
        old = self.specs.get(template)
        if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
            where = f" (also at {old.location})" if old.location else ""
            raise ArtifactError(f"artifact template {template!r} is registered twice{where}")
        self.specs[template] = spec

    @contextlib.contextmanager
    def _loading(self):
        """Make this the registry that `@artifact` registers into, while builders are being loaded.

        On success the state becomes `LOADED`. If anything raises, the previous state is restored: a
        failed first load stays `UNLOADED` (the next use tries again, and raises the same error until
        it is fixed), and a failed re-scan leaves a registry that was loaded still loaded.
        """
        global _active
        previous_active, previous_state = _active, self.load_state
        _active, self.load_state = self, LoadState.LOADING
        try:
            yield
        except BaseException:
            self.load_state = previous_state
            raise
        else:
            self.load_state = LoadState.LOADED
        finally:
            _active = previous_active

    def load(self, modules=None) -> Registry:
        """Register builders from the modules and the chapters' code cells, then validate the graph."""
        with self._loading():
            for m in self.modules if modules is None else modules:
                if m in sys.modules:
                    importlib.reload(sys.modules[m])
                else:
                    importlib.import_module(m)
            if self.chapters_dir is not None:
                self._load_chapters()
            self.validate()
        return self

    def _chapters_signature(self) -> tuple:
        if self.chapters_dir is None:
            return ()
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(self.chapters_dir.glob("*.qmd"))
        )

    def _load_chapters(self) -> None:
        # Forget what an earlier scan registered, so builders removed from a chapter disappear.
        self.specs = {t: sp for t, sp in self.specs.items() if sp.chapter is None}
        self.target_locations = {}
        signature = self._chapters_signature()  # taken before scanning: a change during the scan is seen next time
        scan = scan_chapters(self.chapters_dir)
        self.unparsed_cells = scan.unparsed
        for name, where in scan.gets:
            self.target_locations.setdefault(name, where)
        for b in scan.builders:
            check_pure(b)
            namespace = {
                "__name__": f"convfinqa_chapters.{b.chapter}",
                "Path": Path,
                "Deps": Deps,
                "Files": Files
            }
            fn = compile_builder(b, namespace)
            self._register(
                b.template, fn, deps=b.options.get("deps", ()), files=b.options.get("files", ()),
                heavy=b.options.get("heavy", False), version=b.options.get("version", 1),
                seed=b.options.get("seed", 0), chapter=b.chapter, source=b.source,
                location=b.location,
            )
        self._scan_signature = signature  # only after success, so a failed scan is retried

    def targets(self) -> list[str]:
        """The artifact names chapters ask for with `get("literal")`: what `build --all` builds."""
        self._ensure_loaded()
        return sorted(self.target_locations)

    def _ensure_loaded(self) -> None:
        """Load on first use. A long-lived process (a Jupyter kernel that Quarto keeps alive between
        renders) also re-reads the chapters when their files change, so an edited builder is never
        served stale."""
        if self.load_state is LoadState.LOADING:
            return
        if self.load_state is LoadState.UNLOADED:
            self.load()
        elif self.chapters_dir is not None and self._scan_signature != self._chapters_signature():
            with self._loading():
                self._load_chapters()
                self.validate()

    # names ......................................................................................

    def resolve(self, name: str) -> tuple[Spec, dict[str, Any]]:
        self._ensure_loaded()
        if not NAME.fullmatch(name):
            raise ArtifactError(f"{name!r} is not a valid artifact name (letters, digits and underscores only)")
        matches = []
        for spec in self.specs.values():
            if (m := spec.pattern.fullmatch(name)):
                try:
                    params = {f: spec.fields[f](v) for f, v in m.groupdict().items()}
                except ValueError:
                    continue
                matches.append((spec, params))
        if not matches:
            raise ArtifactError(f"no artifact is registered for the name {name!r}")
        if len(matches) > 1:
            raise ArtifactError(f"the name {name!r} matches several templates: {[s.template for s, _ in matches]}")
        return matches[0]

    def dependencies(self, name: str) -> list[str]:
        """The concrete names of the artifacts that the artifact `name` is built from."""
        spec, params = self.resolve(name)
        return spec.dep_names(params)

    def validate(self, targets=()) -> None:
        """Every dependency template resolves, there are no cycles; `targets` are resolved fully."""
        graph: dict[str, list[str]] = {}
        for tpl, spec in self.specs.items():
            sample = {f: SAMPLE[t] for f, t in spec.fields.items()}
            sample = {f: spec.fields[f](v) for f, v in sample.items()}
            edges = []
            for d in spec.deps:
                try:
                    dep_spec, _ = self.resolve(d.format(**sample))
                except (KeyError, ArtifactError) as e:
                    raise ArtifactError(f"{tpl!r} depends on {d!r}, which does not resolve: {e}") from None
                edges.append(dep_spec.template)
            graph[tpl] = edges
        state: dict[str, int] = {}

        def visit(node: str, trail: list[str]) -> None:
            if state.get(node) == 1:
                raise ArtifactError("dependency cycle: " + " -> ".join(trail + [node]))
            if state.get(node) == 2:
                return
            state[node] = 1
            for nxt in graph[node]:
                visit(nxt, trail + [node])
            state[node] = 2

        for node in graph:
            visit(node, [])
        for name in targets:
            try:
                self._walk(name)
            except ArtifactError as e:
                where = self.target_locations.get(name)
                raise ArtifactError(f"{where}: {e}" if where else str(e)) from None

    def _walk(self, name: str) -> list[str]:
        """`name` and everything it depends on, dependencies first (concrete names)."""
        order: list[str] = []

        def visit(n: str) -> None:
            if n in order:
                return
            for d in self.dependencies(n):
                visit(d)
            order.append(n)

        visit(name)
        return order

    # freshness ..................................................................................

    def _read_node(self, name: str) -> dict | None:
        try:
            node_path = self.dir / name / "node.json"
            return json.loads(node_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _seed(self, spec: Spec, params: dict[str, Any]) -> int:
        return int(params.get("seed", spec.seed))

    def _config(self, spec: Spec, params: dict[str, Any]) -> dict:
        input_hashes = {
            rel: sha256_file(p) if (p := self.root / rel).is_file() else None
            for rel in spec.files
        }
        return {"params": params, "seed": self._seed(spec, params), "files": input_hashes}

    def status(self, name: str) -> Status:
        spec, params = self.resolve(name)
        if (node := self._read_node(name)) is None:
            return Status("missing", "not built")
        if node.get("version") != spec.version:
            return Status("stale", f"version changed ({node.get('version')} -> {spec.version})")

        config = self._config(spec, params)
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

        notes = []
        built = node.get("library_versions", {})
        for lib, ver in library_versions().items():
            if built.get(lib) != ver:
                notes.append(f"{lib} was {built.get(lib)} when built, now {ver}")
        if node.get("python") != sys.version.split()[0].rpartition(".")[0]:
            notes.append(f"built with Python {node.get('python')}")
        return Status("fresh", output_hash=node["output_hash"], notes=tuple(notes))

    def handle(self, name: str, *, stale: bool = False) -> Handle:
        if (node := self._read_node(name)) is None:
            raise ArtifactError(f"artifact {name!r} has not been built")
        return Handle(name, self.dir / name, node["output_hash"], stale, frozenset(node["outputs"]))

    # building ...................................................................................

    def ensure(self, name: str, *, allow_heavy: bool, force: bool = False) -> Handle:
        """Make `name` fresh (building it and its stale dependencies) and return its handle."""
        spec, params = self.resolve(name)
        for dep in spec.dep_names(params):
            self.ensure(dep, allow_heavy=allow_heavy)
        st = self.status(name)
        if st.fresh and not force:
            return self.handle(name)
        if spec.heavy:
            if not allow_heavy:
                raise StaleArtifactError(self._stale_message(name, st))
            self._spawn(name, force=force)
        else:
            self.execute(name, force=force)
        return self.handle(name)

    @staticmethod
    def _stale_message(name: str, st: Status) -> str:
        detail = "is missing" if st.kind == "missing" else f"is stale ({st.reason})"
        msg = (f"Heavy artifact {name!r} {detail}. Build it with:\n    uv run python -m convfinqa.artifacts build {name}\n"
               "(or run ./build.sh, or render with RUN_HEAVY=1).")
        if st.kind == "stale":
            msg += " ALLOW_STALE=1 would load the old result with a warning."
        return msg

    def _spawn(self, name: str, *, force: bool) -> None:
        spec, params = self.resolve(name)
        cmd = subprocess_build.child_command(
            name, artifacts_dir=self.dir, root=self.root, chapters_dir=self.chapters_dir,
            modules=self.modules, force=force,
        )
        env = subprocess_build.child_environment(self._seed(spec, params), self.extra_paths)
        subprocess_build.run(
            name, cmd, env, cwd=self.root, log=subprocess_build.log_path(self.dir, name)
        )

    def execute(self, name: str, *, force: bool = False, redirect_output: bool = False) -> None:
        """Run the builder in this process. Takes the lock, then re-checks freshness."""
        spec, params = self.resolve(name)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / ".locks").mkdir(exist_ok=True)
        with (self.dir / ".locks" / f"{name}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not force and self.status(name).fresh:
                return  # another process built it while we waited
            dep_handles = {}
            for d in spec.dep_names(params):
                if not self.status(d).fresh:
                    raise BuildError(f"cannot build {name!r}: dependency {d!r} is not fresh")
                dep_handles[d] = self.handle(d)
            tmp, final, old = (self.dir / f"{name}{s}" for s in (".tmp", "", ".old"))
            for p in (tmp, old):
                shutil.rmtree(p, ignore_errors=True)
            tmp.mkdir(parents=True)
            if redirect_output:
                fd = os.open(tmp / LOG_NAME, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
                os.dup2(fd, 1)
                os.dup2(fd, 2)
            files = Files({
                rel: InputFile(self.root / rel, sha256_file(self.root / rel)) for rel in spec.files
            })
            args = [tmp, Deps(dep_handles)] + ([files] if spec.files else [])
            config = self._config(spec, params)
            before = set(sys.modules)
            t0 = time.time()
            with preserved_rng_state(), chdir(tmp):
                seed_everything(config["seed"])
                spec.fn(*args, **params)
            dynamic = sorted(
                m for m in set(sys.modules) - before
                if (m == "convfinqa" or m.startswith("convfinqa.")) and not is_framework(m)
            )
            outputs = hash_outputs(tmp)
            node = {
                "name": name,
                "version": spec.version,
                "config": config,
                "deps": {d: h.hash for d, h in dep_handles.items()},
                "outputs": outputs,
                "output_hash": sha256_json(outputs),
                "code_hash": spec.code_hash(dynamic),
                "dynamic_modules": dynamic,
                "git_commit": _git(self.root, "rev-parse", "HEAD"),
                "git_dirty": _git_dirty(self.root),
                "library_versions": library_versions(),
                "python": sys.version.split()[0].rpartition(".")[0],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "duration_s": round(time.time() - t0, 1),
            }
            tmp_node = tmp / "node.json.part"
            tmp_node.write_text(json.dumps(node, indent=2))
            os.replace(tmp_node, tmp / "node.json")  # written last, atomically
            if final.exists():
                os.replace(final, old)
            os.replace(tmp, final)
            shutil.rmtree(old, ignore_errors=True)

    # what chapters call .........................................................................

    def get(self, name: str) -> Handle:
        """Return the artifact `name`, building it if the rules allow (see the module docstring)."""
        spec, _ = self.resolve(name)
        st = self.status(name)
        if st.fresh:
            for note in st.notes:
                warnings.warn(f"artifact {name}: {note}", stacklevel=2)
            h = self.handle(name)
        elif not spec.heavy or os.environ.get("RUN_HEAVY") == "1":
            h = self.ensure(name, allow_heavy=os.environ.get("RUN_HEAVY") == "1")
        elif st.kind == "stale" and os.environ.get("ALLOW_STALE") == "1":
            h = self.handle(name, stale=True)
            _show_stale_warning(name, st.reason)
        else:
            raise StaleArtifactError(self._stale_message(name, st))
        self.records.note_use(h)
        return h

    # chapter records ............................................................................

    def refresh(self, freeze_dir: Path) -> list[str]:
        """Delete the freeze folder of every chapter that used an artifact that is no longer current."""
        return self.records.refresh(freeze_dir, self.status)


def _git(root: Path, *args: str) -> str | None:
    """Run a git command in the given root directory and return its output as a string.

    Used to obtain the current commit hash when updating an artifact result.
    """
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty(root: Path) -> bool | None:
    """Whether the working tree has uncommitted changes, or `None` if that cannot be determined."""
    status = _git(root, "status", "--porcelain")
    return None if status is None else bool(status)


def _show_stale_warning(name: str, reason: str) -> None:
    """Show a warning for a stale artifact.
    
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


DEFAULT = Registry(modules=DEFAULT_MODULES, chapters_dir=ROOT / "docs")
"""The default registry instance used by the `get` function."""


def get(name: str) -> Handle:
    """Get a handle to an artifact by name.
    
    Uses the default registry instance. If the rules allow, also builds the artifact if it is stale
    or missing.  If the artifact cannot be found or built, an `ArtifactError` is raised.
    """
    return DEFAULT.get(name)
