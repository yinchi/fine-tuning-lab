"""The registry: builder registration, freshness, building (in process or in a subprocess), and `get`."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import re
import shutil
import string
import subprocess
import sys
import time
import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chapters import ANNOTATION_NAMES, check_pure, compile_builder, scan_chapters
from .errors import ArtifactError, BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .hashing import (
    LOG_NAME, code_hash, hash_outputs, inline_code_hash, is_framework, library_versions, module_closure,
    module_path, sha256_file, sha256_json,
)
from .seeding import chdir, preserved_rng_state, seed_everything

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


@dataclass
class Spec:
    template: str
    fn: Callable
    deps: list[str]
    files: list[str]
    heavy: bool
    version: int
    seed: int
    fields: dict[str, type]
    pattern: re.Pattern
    source: str | None = None  # the function's source, when it is defined in a chapter cell
    chapter: str | None = None  # the chapter that defines it, if any
    location: str = ""  # file:line of the definition, for error messages

    @property
    def module(self) -> str:
        return self.fn.__module__


_SAMPLE = {str: "x", int: "0"}
_NAME = re.compile(r"[A-Za-z0-9_]+")
_LITERAL = re.compile(r"[A-Za-z0-9_]*")
# What each kind of name field matches: str is letters and digits, int is canonical digits. Fields are
# separated by literal text such as "_", and neither kind can contain "_", so matching is unambiguous.
_FIELD_REGEX = {str: "[A-Za-z0-9]+", int: "0|[1-9][0-9]*"}
_active: Registry | None = None


def _template_parts(template: str) -> list[tuple[str, str | None]]:
    """Split a name template into (literal, field) pairs, rejecting anything but simple fields."""
    parts = []
    for lit, field, spec, conv in string.Formatter().parse(template):
        if spec or conv or not _LITERAL.fullmatch(lit) or (field is not None and not field.isidentifier()):
            raise ArtifactError(
                f"bad name template {template!r}: names use letters, digits and underscores, "
                "with fields written as {name}"
            )
        parts.append((lit, field))
    for (_, a), (lit, b) in zip(parts, parts[1:]):
        if a is not None and b is not None and lit == "":
            raise ArtifactError(f"bad name template {template!r}: two fields need text between them")
    return parts


def artifact(template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
    """Register a builder with the active registry (the default one outside `Registry.load`)."""
    return (_active or DEFAULT).artifact(template, deps=deps, files=files, heavy=heavy, version=version, seed=seed)


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
        self._loaded = False
        self._loading = False
        self._scan_signature: tuple = ()  # (name, mtime, size) of the chapter files last scanned
        self._chapter_seen: dict[str, str] = {}
        self._chapter_stale: set[str] = set()

    # registration ...............................................................................

    def artifact(self, template: str, *, deps=(), files=(), heavy=False, version=1, seed=0):
        """Decorator registering a builder. In a chapter cell that the scan already registered, it is a no-op."""
        if not self._loading:
            self._ensure_loaded()
            existing = self.specs.get(template)
            if existing is not None and existing.chapter is not None:
                return lambda fn: fn  # the chapter's scanned definition is the registered one

        def decorate(fn: Callable) -> Callable:
            self._register(template, fn, deps=deps, files=files, heavy=heavy, version=version, seed=seed)
            return fn

        return decorate

    def _register(
        self, template: str, fn: Callable, *, deps, files, heavy, version, seed,
        chapter: str | None = None, source: str | None = None, location: str = "",
    ) -> None:
        parts = _template_parts(template)
        fields = [f for _, f in parts if f is not None]
        try:
            hints = typing.get_type_hints(fn)
        except NameError as e:
            raise ArtifactError(
                f"builder for {template!r}: annotation uses an unknown name ({e}). "
                f"Annotations may use only {', '.join(ANNOTATION_NAMES)}, builtins and the module's own imports."
            ) from None
        spec_fields = {f: hints.get(f) for f in fields}
        self._check_signature(template, fn, spec_fields, bool(files))
        if chapter is None and module_path(fn.__module__) is None:
            raise ArtifactError(
                f"builder for {template!r} must live in an importable module or in a chapter code cell "
                "(a function defined elsewhere, such as an interactive session, cannot be rebuilt in a subprocess)"
            )
        for d in deps:
            unknown = {f for _, f in _template_parts(d) if f is not None} - set(fields)
            if unknown:
                raise ArtifactError(f"{template!r}: dependency {d!r} uses fields {sorted(unknown)} the name lacks")
        regex = "".join(
            re.escape(lit) + (f"(?P<{f}>{_FIELD_REGEX[spec_fields[f]]})" if f else "") for lit, f in parts
        )
        spec = Spec(
            template, fn, list(deps), list(files), heavy, version, seed, spec_fields, re.compile(regex),
            source, chapter, location,
        )
        old = self.specs.get(template)
        if old and (old.fn.__module__, old.fn.__qualname__) != (fn.__module__, fn.__qualname__):
            where = f" (also at {old.location})" if old.location else ""
            raise ArtifactError(f"artifact template {template!r} is registered twice{where}")
        self.specs[template] = spec

    @staticmethod
    def _check_signature(template, fn, fields, has_files):
        import inspect

        params = list(inspect.signature(fn).parameters.values())
        expected = ["out", "deps"] + (["files"] if has_files else [])
        names = [p.name for p in params]
        pos = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        kw = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
        if [p.name for p in pos] != expected or kw != set(fields):
            raise ArtifactError(
                f"builder for {template!r} must have the signature "
                f"({', '.join(expected)}, *, {', '.join(sorted(fields))}); got ({', '.join(names)}). "
                "Builders take only out, deps, files and the fields of their own name."
            )
        for f, t in fields.items():
            if t not in _SAMPLE:
                raise ArtifactError(f"builder for {template!r}: field {f!r} needs a str or int annotation")

    def load(self, modules=None) -> Registry:
        """Register builders from the modules and the chapters' code cells, then validate the graph."""
        global _active
        previous, _active = _active, self
        self._loading = True
        try:
            for m in self.modules if modules is None else modules:
                if m in sys.modules:
                    importlib.reload(sys.modules[m])
                else:
                    importlib.import_module(m)
            if self.chapters_dir is not None:
                self._load_chapters()
        finally:
            _active = previous
            self._loading = False
        self._loaded = True
        self.validate()
        return self

    def _chapters_signature(self) -> tuple:
        if self.chapters_dir is None:
            return ()
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(self.chapters_dir.glob("*.qmd"))
        )

    def _load_chapters(self) -> None:
        # Forget what an earlier scan registered, so builders removed from a chapter disappear.
        self.specs = {t: sp for t, sp in self.specs.items() if sp.chapter is None}
        self.target_locations = {}
        self._scan_signature = self._chapters_signature()
        scan = scan_chapters(self.chapters_dir)
        self.unparsed_cells = scan.unparsed
        for name, where in scan.gets:
            self.target_locations.setdefault(name, where)
        for b in scan.builders:
            check_pure(b)
            namespace = {"__name__": f"convfinqa_chapters.{b.chapter}", "Path": Path, "Deps": Deps, "Files": Files}
            fn = compile_builder(b, namespace)
            self._register(
                b.template, fn, deps=b.options.get("deps", ()), files=b.options.get("files", ()),
                heavy=b.options.get("heavy", False), version=b.options.get("version", 1),
                seed=b.options.get("seed", 0), chapter=b.chapter, source=b.source, location=b.location,
            )

    def targets(self) -> list[str]:
        """The artifact names chapters ask for with `get("literal")`: what `build --all` builds."""
        self._ensure_loaded()
        return sorted(self.target_locations)

    def _ensure_loaded(self) -> None:
        """Load on first use. A long-lived process (a Jupyter kernel that Quarto keeps alive between renders)
        also re-reads the chapters when their files change, so an edited builder is never served stale."""
        if not self._loaded:
            self.load()
        elif self.chapters_dir is not None and self._scan_signature != self._chapters_signature():
            global _active
            previous, _active = _active, self
            self._loading = True
            try:
                self._load_chapters()
            finally:
                _active = previous
                self._loading = False
            self.validate()

    # names ......................................................................................

    def resolve(self, name: str) -> tuple[Spec, dict[str, Any]]:
        self._ensure_loaded()
        if not _NAME.fullmatch(name):
            raise ArtifactError(f"{name!r} is not a valid artifact name (letters, digits and underscores only)")
        matches = []
        for spec in self.specs.values():
            m = spec.pattern.fullmatch(name)
            if m:
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

    def dep_names(self, spec: Spec, params: dict[str, Any]) -> list[str]:
        return [d.format(**params) for d in spec.deps]

    def validate(self, targets=()) -> None:
        """Every dependency template resolves, there are no cycles; `targets` are resolved fully."""
        graph: dict[str, list[str]] = {}
        for tpl, spec in self.specs.items():
            sample = {f: _SAMPLE[t] for f, t in spec.fields.items()}
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
            spec, params = self.resolve(n)
            for d in self.dep_names(spec, params):
                visit(d)
            order.append(n)

        visit(name)
        return order

    # freshness ..................................................................................

    def _node_path(self, name: str) -> Path:
        return self.dir / name / "node.json"

    def _read_node(self, name: str) -> dict | None:
        try:
            return json.loads(self._node_path(name).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _seed(self, spec: Spec, params: dict[str, Any]) -> int:
        return int(params.get("seed", spec.seed))

    def _input_hashes(self, spec: Spec) -> dict[str, str | None]:
        out = {}
        for rel in spec.files:
            p = self.root / rel
            out[rel] = sha256_file(p) if p.is_file() else None
        return out

    def _config(self, spec: Spec, params: dict[str, Any]) -> dict:
        return {"params": params, "seed": self._seed(spec, params), "files": self._input_hashes(spec)}

    @staticmethod
    def _code_hash(spec: Spec, dynamic=()) -> str:
        if spec.source is not None:
            return inline_code_hash(spec.source, dynamic)
        return code_hash(module_closure(spec.module) | set(dynamic))

    def status(self, name: str) -> Status:
        spec, params = self.resolve(name)
        node = self._read_node(name)
        if node is None:
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
        if node.get("code_hash") != self._code_hash(spec, node.get("dynamic_modules", [])):
            return Status("stale", "code changed")
        deps = self.dep_names(spec, params)
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
        node = self._read_node(name)
        if node is None:
            raise ArtifactError(f"artifact {name!r} has not been built")
        return Handle(name, self.dir / name, node["output_hash"], stale, frozenset(node["outputs"]))

    # building ...................................................................................

    @staticmethod
    def build_command(name: str) -> str:
        return f"uv run python -m convfinqa.artifacts build {name}"

    def ensure(self, name: str, *, allow_heavy: bool, force: bool = False) -> Handle:
        """Make `name` fresh (building it and its stale dependencies) and return its handle."""
        spec, params = self.resolve(name)
        for dep in self.dep_names(spec, params):
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
        msg = (f"Heavy artifact {name!r} {detail}. Build it with:\n    {Registry.build_command(name)}\n"
               "(or run ./build.sh, or render with RUN_HEAVY=1).")
        if st.kind == "stale":
            msg += " ALLOW_STALE=1 would load the old result with a warning."
        return msg

    def _spawn(self, name: str, *, force: bool) -> None:
        spec, params = self.resolve(name)
        cmd = [sys.executable, "-m", "convfinqa.artifacts", "--artifacts-dir", str(self.dir), "--root", str(self.root)]
        cmd += [f"--chapters-dir={self.chapters_dir or ''}"]
        for m in self.modules:
            cmd += ["--import", m]
        cmd += ["_run", name] + (["--force"] if force else [])
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(self._seed(spec, params))
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if self.extra_paths:
            env["PYTHONPATH"] = os.pathsep.join(self.extra_paths + [env.get("PYTHONPATH", "")])
        log = self.dir / f"{name}.tmp" / LOG_NAME
        print(f"[artifacts] building {name} in a subprocess (log: {log})", file=sys.stderr, flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, cwd=self.root)
        if proc.returncode != 0:
            tail = ""
            if log.exists():
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
            raise BuildError(f"building {name!r} failed (exit code {proc.returncode}); log: {log}\n{tail}")
        print(f"[artifacts] built {name} in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

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
            for d in self.dep_names(spec, params):
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
                "code_hash": self._code_hash(spec, dynamic),
                "dynamic_modules": dynamic,
                "git_commit": _git(self.root, "rev-parse", "HEAD"),
                "git_dirty": bool(_git(self.root, "status", "--porcelain")),
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
        self._record_use(h)
        return h

    def _record_use(self, h: Handle) -> None:
        document = os.environ.get("QUARTO_DOCUMENT_FILE")
        if not document:
            return
        self._chapter_seen[h.name] = h.hash
        if h.stale:
            self._chapter_stale.add(h.name)
        path = self.dir / "_chapters" / f"{Path(document).stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"artifacts": self._chapter_seen, "stale": sorted(self._chapter_stale)}, indent=2))

    # CLI actions ................................................................................

    def tree(self, names: list[str]) -> list[str]:
        lines: list[str] = []

        def show(n: str, depth: int) -> None:
            st = self.status(n)
            label = st.kind + (f": {st.reason}" if st.reason else "")
            lines.append(f"{'  ' * depth}{n}  [{label}]")
            for note in st.notes:
                lines.append(f"{'  ' * depth}  note: {note}")
            spec, params = self.resolve(n)
            for d in self.dep_names(spec, params):
                show(d, depth + 1)

        for n in names:
            show(n, 0)
        return lines

    def refresh(self, freeze_dir: Path) -> list[str]:
        """Delete the freeze folder of every chapter that used an artifact that is no longer current."""
        removed = []
        for rec in sorted((self.dir / "_chapters").glob("*.json")):
            data = json.loads(rec.read_text())
            outdated = bool(data.get("stale"))
            for name, h in data["artifacts"].items():
                try:
                    st = self.status(name)
                except ArtifactError:
                    outdated = True
                    continue
                outdated |= not st.fresh or st.output_hash != h
            if outdated:
                shutil.rmtree(freeze_dir / rec.stem, ignore_errors=True)
                rec.unlink()  # the chapter re-records its usage when it re-executes
                removed.append(rec.stem)
        return removed


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _show_stale_warning(name: str, reason: str) -> None:
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


def get(name: str) -> Handle:
    """Chapter entry point: `artifacts.get("split-v1")`."""
    return DEFAULT.get(name)
