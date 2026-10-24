"""Named, cached results ("artifacts") with recursive freshness.

An artifact is a directory `artifacts/<name>/` produced by a registered builder:

```python
@artifact("sft-{variant}-{seed}", deps=["tokenized-{variant}"], files=["configs/sft.json"],
          heavy=True, version=1)
def build_sft(out: Path, deps: Deps, files: Files, *, variant: str, seed: int) -> None: ...
```

Builder purity: a builder receives only `out` (a scratch directory it fills), the handles of its
declared dependencies (`deps`), its declared static input files (`files`), and the fields of its
own name (its identity). Hyperparameters live in a declared file or a light config artifact.

An artifact is *fresh* when `node.json` exists and matches: the builder's `version`, its config
(name fields, input-file hashes, seed), a hash of the normalized source of the modules it uses, and
the output hashes of its dependencies, each of which must itself be fresh. Bump `version` when a
change this cannot see (a library upgrade, a changed external resource) should invalidate results.

Heavy artifacts build in a subprocess, and only with `RUN_HEAVY=1` or the `build` command;
otherwise a stale or missing one raises `StaleArtifactError`. `ALLOW_STALE=1` additionally
lets a stale-but-present heavy artifact load, with a visible warning.
"""

from __future__ import annotations

from .errors import ArtifactError, BuildError, StaleArtifactError
from .handles import Deps, Files, Handle, InputFile, Status
from .registry import DEFAULT, Registry, artifact, get
from .seeding import seed_everything

__all__ = [
    "ArtifactError", "BuildError", "StaleArtifactError", "Deps", "Files", "Handle", "InputFile", "Status",
    "DEFAULT", "Registry", "artifact", "get", "seed_everything",
]
