"""What a builder declaration is: the `Spec` record and the rules a declaration must follow.

The rules cover the name template (`template_parts`), the options given to `@artifact`
(`check_options`), the builder's signature (`check_signature`) and its body (`check_pure`). `make_spec`
applies them all and returns the `Spec`; the catalog calls it whenever a builder is registered, and uses
`NAME` and `SAMPLE` when resolving and validating names.

Pure validation and data; nothing here touches the artifacts on disk.
"""

from __future__ import annotations

import ast
import inspect
import re
import string
import textwrap
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .chapters import ANNOTATION_NAMES, outside_names
from .errors import ArtifactError
from .hashing import inline_code_hash, module_path, sha256_file


FIELD_TYPES = (str, int)
"""The types a name field may have. `SAMPLE` and `_FIELD_REGEX` have an entry for each."""

SAMPLE = {str: "x", int: "0"}
"""A valid example value for each field type, used to check dependency templates without real values."""

_NAME_CHARS = "[A-Za-z0-9_]"
"""The characters an artifact name may contain: letters, digits and underscores."""

NAME = re.compile(f"{_NAME_CHARS}+")
"""Regular expression for valid artifact names, which must not be empty."""

_LITERAL = re.compile(f"{_NAME_CHARS}*")
"""Regular expression for valid literal parts of name templates, which may be empty.

See `template_parts` for how templates are split into parts.
"""

_FIELD_REGEX = {str: "[A-Za-z0-9]+", int: "0|[1-9][0-9]*"}
"""Regular expressions for each allowed field value, keyed by the field type."""


@dataclass
class Spec:
    """One registered builder: what its `@artifact(...)` declared, validated and ready to use.

    It stands for a whole family of artifacts, the names that match `template`. `Catalog.resolve`
    turns a concrete name such as `sft_dsl_0` into the `Spec` it matches and the parameters parsed from
    the name (`{"variant": "dsl", "seed": 0}`).
    """

    template: str
    """The template string for the artifact names, with placeholders for variable parts.

    For example, a template like "foo_{variant}_{seed}" has two variable parts: "variant" and
    "seed". A `seed` field is the build seed (see `build_seed`); without one the declared default is used.
    """

    fn: Callable
    """The builder function, called as `fn(out, deps, [files], **fields)`: the scratch directory, the
    dependency handles, the input files (only if any are declared), and the template's fields as
    keyword arguments."""

    deps: list[str]
    """Template strings for the dependencies of this artifact. Allowed placeholders are
    those defined in `template`."""

    files: list[str]
    """Declared static input files, as literal paths relative to the project root (no placeholders).
    Their content hashes are part of the artifact's config."""

    heavy: bool
    """Whether the build runs in a subprocess and needs RUN_HEAVY=1 or the build command."""

    version: int
    """Bump it to invalidate existing results when something the hashes cannot see has changed, such as
    a library upgrade or a changed external resource."""

    seed: int
    """A default seed value for this artifact builder. Used to generate reproducible artifacts
    when `seed` is not provided in the template."""

    fields: dict[str, type]
    """Each field of the template with its type, taken from the builder's annotations.

    The only allowed types are `FIELD_TYPES`, i.e., `str` and `int`.
    """

    pattern: re.Pattern
    """The compiled pattern that a concrete name must fully match, built from the template and the
    field types."""

    source: str
    """The builder function's source, without its decorator. The code hash covers it, so editing
    anything else in the file (other builders, other cells, prose) leaves the artifact fresh."""

    chapter: str | None = None
    """The chapter that defines this builder, or `None` if it is defined in a module."""

    location: str = ""
    """`file:line` of the builder's `def`, for error messages."""

    def code_hash(self, dynamic=()) -> str:
        """Hash of the code this builder runs, for freshness.

        It covers the function's normalized source and the `convfinqa` modules that source imports
        (anywhere inside the function), but nothing else in the file the function lives in. `dynamic`
        adds modules that were found loading while it ran.
        """
        return inline_code_hash(self.source, dynamic)

    def build_seed(self, params: dict) -> int:
        """The seed for a build: the name's `seed` field if it has one, else the declared default."""
        return int(params.get("seed", self.seed))

    def config(self, params: dict, root: Path) -> dict:
        """What a build is configured with: the name's fields, the seed, and the content hash of each
        declared input file (`None` for a file that does not exist). Recorded in `node.json`; freshness
        compares it with the current one."""
        input_hashes = {
            rel: sha256_file(path) if (path := root / rel).is_file() else None for rel in self.files
        }
        return {"params": params, "seed": self.build_seed(params), "files": input_hashes}

    def dep_names(self, params: dict) -> list[str]:
        """The concrete names of the dependencies, from this artifact's name fields."""
        return [d.format(**params) for d in self.deps]

    @property
    def module(self) -> str:
        """The module the builder is defined in (a synthetic name for a chapter cell)."""
        return self.fn.__module__


def template_parts(template: str) -> list[tuple[str, str | None]]:
    """Split a name template into (literal, field) pairs, rejecting anything but simple fields.

    Each pair is the literal text before a field, and the field's name. Any text after the last field
    is also a pair, with `None` as the field name. For example, the template
    "file_{name}_{version}_suffix" is split into
    `[("file_", "name"), ("_", "version"), ("_suffix", None)]`.

    Raises `ArtifactError` for an empty template, one that is not well formed (an unclosed `{` or a
    lone `}`), a field that is not a plain identifier or has a format spec or conversion, literal
    text outside letters, digits and underscores, or two fields with nothing between them.
    """
    if not template:
        raise ArtifactError("bad name template '': a template must not be empty")
    bad = (
        f"bad name template {template!r}: names use letters, digits and underscores, "
        "with fields written as {name}"
    )
    parts = []
    try:
        pieces = list(string.Formatter().parse(template))
    except ValueError as e:  # an unclosed `{`, a lone `}`
        raise ArtifactError(f"{bad} ({e})") from None
    for lit, field, spec, conv in pieces:
        if spec or conv or not _LITERAL.fullmatch(lit) or (field is not None and not field.isidentifier()):
            raise ArtifactError(bad)
        parts.append((lit, field))
    for (_, a), (lit, b) in zip(parts, parts[1:]):
        if a is not None and b is not None and lit == "":
            raise ArtifactError(f"bad name template {template!r}: two fields need text between them")
    return parts


def check_signature(template: str, fn: Callable, fields: dict[str, type], has_files: bool) -> None:
    """Check that a builder has exactly the signature the rules allow.

    The positional parameters must be `out`, `deps`, and `files` only if input files are declared. The
    keyword-only parameters must be exactly the template's `fields`, each annotated `str` or `int`; a
    `seed` field must be an `int`.
    """
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
        if t not in FIELD_TYPES:
            raise ArtifactError(f"builder for {template!r}: field {f!r} needs a str or int annotation")
    if fields.get("seed", int) is not int:
        raise ArtifactError(
            f"builder for {template!r}: the `seed` field is the build seed, so it must be an int"
        )


def module_builder_source(template: str, fn: Callable) -> tuple[str, str]:
    """The source of a builder defined in a module, without its decorator, and the `<path>:<line>` of
    its `def`.

    Raises `ArtifactError` if the source cannot be read or the builder is not a plain `def`.
    """
    try:
        lines, first_line = inspect.getsourcelines(fn)
        path = inspect.getsourcefile(fn) or "<unknown>"
        text = textwrap.dedent("".join(lines))
        node = ast.parse(text).body[0]
    except (OSError, TypeError, SyntaxError, IndexError) as e:
        raise ArtifactError(f"builder for {template!r}: cannot read its source ({e})") from None
    if not isinstance(node, ast.FunctionDef):
        raise ArtifactError(f"builder for {template!r} must be a plain `def`, not a lambda or `async def`")
    return ast.get_source_segment(text, node), f"{path}:{first_line + node.lineno - 1}"


def check_pure(name: str, location: str, source: str) -> None:
    """Raise `ArtifactError` unless the builder uses only its parameters, builtins and imports inside it.

    Everything the hash does not cover would otherwise be invisible to it: a module-level helper or
    constant could change without the artifact going stale. Imports inside the function are fine,
    because the hash follows them, but they must be absolute: a relative import would need the module's
    package to be followed.
    """
    bad = outside_names(source)
    if bad:
        raise ArtifactError(
            f"{location}: builder {name!r} uses {sorted(bad)} from outside the function. A builder may "
            "use only its parameters, builtins and imports made inside it."
        )
    if any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(ast.parse(source))):
        raise ArtifactError(f"{location}: builder {name!r} uses a relative import; use absolute imports")


def check_options(template: str, *, deps, files, heavy, version, seed) -> None:
    """Check the options given to `@artifact`, which chapters write as literals and could get wrong.

    `deps` and `files` must be lists (or tuples) of strings, not a bare string, which would be read
    as one entry per character. Each input file must be a relative path that stays inside the project.
    `heavy` must be a real boolean (the string `"false"` is truthy), and `version` and `seed` integers.
    """
    for option, value in (("deps", deps), ("files", files)):
        if isinstance(value, str) or not all(isinstance(item, str) for item in value):
            raise ArtifactError(f"{template!r}: {option} must be a list of strings, got {value!r}")
    for rel in files:
        path = PurePosixPath(rel)
        if not rel or path.is_absolute() or ".." in path.parts:
            raise ArtifactError(
                f"{template!r}: input file {rel!r} must be a relative path inside the project (no '..')"
            )
    if not isinstance(heavy, bool):
        raise ArtifactError(f"{template!r}: heavy must be True or False, got {heavy!r}")
    for option, value in (("version", version), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactError(f"{template!r}: {option} must be an integer, got {value!r}")


def make_spec(
    template: str, fn: Callable, *, deps, files, heavy, version, seed,
    chapter: str | None = None, source: str | None = None, location: str = "",
) -> Spec:
    """Validate a builder declaration and build its `Spec`; raise `ArtifactError` if it breaks a rule.

    The rules: a simple name template (`template_parts`); sound options (`check_options`); a builder
    signature of exactly `out`, `deps`, optionally `files`, and the template's fields as typed
    keyword-only parameters (`check_signature`); a builder that lives in an importable module or a
    chapter cell; a self-contained body (`check_pure`); and dependency templates that use only the
    name's own fields.

    For a chapter builder `chapter`, `source` and `location` are given (from the scan); for a module
    builder they are read from the module.
    """
    parts = template_parts(template)
    check_options(template, deps=deps, files=files, heavy=heavy, version=version, seed=seed)
    fields = [f for _, f in parts if f is not None]
    try:
        hints = typing.get_type_hints(fn)
    except NameError as e:
        raise ArtifactError(
            f"builder for {template!r}: annotation uses an unknown name ({e}). "
            f"Annotations may use only {', '.join(ANNOTATION_NAMES)}, builtins and the module's own imports."
        ) from None
    spec_fields = {f: hints.get(f) for f in fields}
    check_signature(template, fn, spec_fields, bool(files))
    if chapter is None and module_path(fn.__module__) is None:
        raise ArtifactError(
            f"builder for {template!r} must live in an importable module or in a chapter code cell "
            "(a function defined elsewhere, such as an interactive session, cannot be rebuilt in a "
            "subprocess)"
        )
    if chapter is None:
        source, location = module_builder_source(template, fn)
    elif source is None:
        raise ValueError("a chapter builder needs its source")  # a bug in the caller, not in the declaration
    check_pure(fn.__name__, location, source)
    for d in deps:
        unknown = {f for _, f in template_parts(d) if f is not None} - set(fields)
        if unknown:
            raise ArtifactError(
                f"{template!r}: dependency {d!r} uses fields {sorted(unknown)} the name lacks"
            )
    regex = "".join(
        re.escape(lit) + (f"(?P<{f}>{_FIELD_REGEX[spec_fields[f]]})" if f else "") for lit, f in parts
    )
    return Spec(
        template, fn, list(deps), list(files), heavy, version, seed, spec_fields, re.compile(regex),
        source, chapter, location,
    )
