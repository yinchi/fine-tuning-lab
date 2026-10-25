"""Named, cached results ("artifacts") with recursive freshness: compute heavy things once, reuse them.

An artifact is a directory `artifacts/<name>/` holding the outputs of a builder, and a `node.json` that
records what it was built from. A chapter asks for one with `get("sft_dsl_0")`: it is loaded if it is
fresh, and otherwise built or refused, as the policy says.

**Declaring a builder.** It sits in the chapter cell that uses it:

```python
@artifact("sft_{variant}_{seed}", deps=["tokenized_{variant}"], files=["configs/sft.json"], heavy=True)
def build_sft(out: Path, deps: Deps, files: Files, *, variant: str, seed: int) -> None:
    from convfinqa.llm import train
    ...

sft = get("sft_dsl_0")  #HEAVY
```

The name is a template: letters, digits and underscores, with `{fields}` that are a `str` or an `int`;
`sft_dsl_0` is one member of the family. A builder takes only `out` (a scratch directory it fills), the
handles of its dependencies, its declared input files, and the fields of its name. It must be
self-contained: parameters, builtins and imports made inside the function, nothing else, so hyperparameters
live in a declared file or a light config artifact. Builders shared by several chapters live in
`convfinqa.builders`. The `.qmd` files are read, not run, to find builders and `get("literal")` calls, so
`status` and `build --all` know every artifact before any chapter runs.

**Fresh or stale.** An artifact is fresh when `node.json` matches: the builder's `version`; its config
(name fields, seed, contents of its input files); its code hash, which covers the builder's own function
and the `convfinqa` modules that function imports (formatting, comments and docstrings do not count, and
nothing else in the file does); and each dependency, which must itself be fresh and have the same output
hash as when this artifact was built. A dependency rebuilt with identical output leaves its dependents
fresh. Bump `version` when something the hashes cannot see changes, such as a library upgrade.

**Building.** A light artifact builds in the process that asks, usually a notebook kernel. A heavy one
builds in a subprocess, and only with `RUN_HEAVY=1` or `python -m convfinqa.artifacts build`; otherwise
`get` raises `StaleArtifactError` naming the command. `ALLOW_STALE=1` lets a stale heavy artifact load,
with a visible warning, so a preview keeps working (see `policy`).

**From the command line:** `python -m convfinqa.artifacts status|build|refresh` (see `cli`), and
`./build.sh` to build everything, render the book, and preview it.

**The modules, by role**

- what exists: `spec` (the declaration rules), `chapters` (finding builders in the `.qmd` files),
  `catalog` (loading, name resolution, graph validation)
- what is on disk and what ran: `store` (layout, `node.json`, lock), `records` (which artifacts each
  chapter used), `provenance`, `seeding`, `subprocess_build`
- rules and types: `policy` (when `get` may build or load stale), `handles`, `errors`, `hashing`
- coordination: `registry` (freshness, building, `get`), `cli`
"""

from __future__ import annotations

from .errors import ArtifactError, BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .policy import Policy
from .registry import DEFAULT, Registry, artifact, get
from .seeding import seed_everything

__all__ = [
    # declaring and asking for artifacts
    "artifact", "get", "Deps", "Files",
    # what comes back, and the rules for getting it
    "Handle", "InputFile", "Status", "Policy",
    # the registry behind them
    "DEFAULT", "Registry",
    # errors
    "ArtifactError", "BuildError", "StaleArtifactError",
    # seeding
    "seed_everything",
]
