"""What a builder declaration is: the `Spec` record and the rules a declaration must follow.

Pure validation and data; nothing here touches the artifacts on disk. The registry calls `make_spec`
when a builder is registered, and uses `NAME` and `SAMPLE` when resolving and validating names.
"""

from __future__ import annotations

import inspect
import re
import string
import typing
from collections.abc import Callable
from dataclasses import dataclass

from .chapters import ANNOTATION_NAMES
from .errors import ArtifactError
from .hashing import code_hash, inline_code_hash, module_closure, module_path

@dataclass
class Spec:
    """A specification for an artifact builder.
    
    A builder allows a group of related artifacts to be grouped under a common template,
    which contains placeholders for variable parts of the artifact names.
    """
    template: str
    """The template string for the artifact names, with placeholders for variable parts.
    
    For example, a template like "foo_{variant}_{seed}" has two variable parts: "variant" and
    "seed".  The `seed` field has special meaning; if absent, a default value is used.
    """

    fn: Callable
    """The function associated with this artifact builder.
    
    Must accept the same keyword arguments as the placeholders in the template.
    """

    deps: list[str]
    """Template strings for the dependencies of this artifact. Allowed placeholders are
    those defined in `template`."""

    files: list[str]
    """Declared static input files, as literal paths relative to the project root (no placeholders).
    Their content hashes are part of the artifact's config."""

    heavy: bool
    """Whether the build runs in a subprocess and needs RUN_HEAVY=1 or the build command."""

    version: int
    """The version of this artifact builder. Incrementing this forces rebuilds of artifacts."""

    seed: int
    """A default seed value for this artifact builder. Used to generate reproducible artifacts
    when `seed` is not provided in the template."""

    fields: dict[str, type]
    """Type annotations for the fields in the template.
    
    The only allowed types are `FIELD_TYPES`, i.e., `str` and `int`.
    """

    pattern: re.Pattern
    """The compiled regular expression pattern for matching artifact names based on the template."""

    source: str | None = None
    """The source code of the function defining this artifact, if available (i.e., when
    defined in a chapter cell)."""

    chapter: str | None = None
    """The chapter that defines this artifact, if available."""

    location: str = ""
    """`file:line` of the definition in the chapter file, for error messages. Empty for builders
    defined in modules."""

    def code_hash(self, dynamic=()) -> str:
        """Hash of the code this builder runs, for freshness.

        A builder declared in a chapter is hashed from its own source and the `convfinqa` modules
        that source imports; a module builder from its module's whole import closure. `dynamic` adds
        modules that were found loading while it ran.
        """
        if self.source is not None:
            return inline_code_hash(self.source, dynamic)
        return code_hash(module_closure(self.module) | set(dynamic))

    def dep_names(self, params: dict) -> list[str]:
        """The concrete names of the dependencies, from this artifact's name fields."""
        return [d.format(**params) for d in self.deps]

    @property
    def module(self) -> str:
        """Return the module in which the artifact's function is defined."""
        return self.fn.__module__


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


def template_parts(template: str) -> list[tuple[str, str | None]]:
    """Split a name template into (literal, field) pairs, rejecting anything but simple fields.
    
    Templates are split into parts as pairs: each field name and the literal text before it. Any
    trailing literal part after the last field is also included as a pair with `None` as the field
    name. For example, the template "file_{name}_{version}_suffix" would be split into:
    `[("file_", "name"), ("_", "version"), ("_suffix", None)]`.
    """
    if not template:
        raise ArtifactError("bad name template '': a template must not be empty")
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


def check_signature(template: str, fn: Callable, fields: dict[str, type], has_files: bool) -> None:
    """Check that a builder has exactly the signature the rules allow, and that its fields are typed."""
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
        raise ArtifactError(f"builder for {template!r}: the `seed` field is the build seed, so it must be an int")


def make_spec(
    template: str, fn: Callable, *, deps, files, heavy, version, seed,
    chapter: str | None = None, source: str | None = None, location: str = "",
) -> Spec:
    """Validate a builder declaration and build its `Spec`; raise `ArtifactError` if it breaks a rule.

    The rules: a simple name template (`template_parts`); a builder signature of exactly `out`, `deps`,
    optionally `files`, and the template's fields as typed keyword-only parameters (`check_signature`);
    a builder that lives in an importable module or a chapter cell; and dependency templates that use
    only the name's own fields.
    """
    parts = template_parts(template)
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
            "(a function defined elsewhere, such as an interactive session, cannot be rebuilt in a subprocess)"
        )
    for d in deps:
        unknown = {f for _, f in template_parts(d) if f is not None} - set(fields)
        if unknown:
            raise ArtifactError(f"{template!r}: dependency {d!r} uses fields {sorted(unknown)} the name lacks")
    regex = "".join(
        re.escape(lit) + (f"(?P<{f}>{_FIELD_REGEX[spec_fields[f]]})" if f else "") for lit, f in parts
    )
    return Spec(
        template, fn, list(deps), list(files), heavy, version, seed, spec_fields, re.compile(regex),
        source, chapter, location,
    )
