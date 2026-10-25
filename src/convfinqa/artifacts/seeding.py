"""Seeding for builds: make the global random generators repeatable, and put them back afterwards.

`seed_everything` is what makes a build's random draws independent of whatever the process did before.
`preserved_rng_state` is for a build that runs inside a chapter's own process (a light artifact in a
Jupyter kernel): it keeps the build's seeding from changing the random numbers the chapter sees later.
"""

from __future__ import annotations

import contextlib
import importlib.util
import random
import sys
from collections.abc import Generator


def seed_everything(seed: int) -> None:
    """Seed the global random generators of Python, NumPy and torch (CPU and CUDA).

    Afterwards, code that draws from those generators gets the same numbers whatever the process did
    before, which is what makes a build repeatable. NumPy and torch are imported if installed: a library
    that was not imported yet would otherwise seed itself from entropy when the builder first imports it.

    It covers the global generators only. A generator made with `np.random.default_rng()` or
    `torch.Generator()` has its own state and must be given the seed by the code that makes it. It does
    not cover `PYTHONHASHSEED` (fixed when the interpreter starts, so a build subprocess gets it through
    its environment), the order of floating-point work on a GPU, or DataLoader workers.
    """
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed % (2**32))  # the legacy global generator accepts 32-bit seeds
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # queued if CUDA is not initialized yet


@contextlib.contextmanager
def preserved_rng_state() -> Generator[None, None, None]:
    """Restore the global random generators afterwards, so a build never perturbs the chapter around it.

    **Usage:**
    ```
    with preserved_rng_state():
        # code that uses RNG
        ...
    ```

    The state of Python's `random`, NumPy's global generator, torch (CPU) and CUDA is saved on entry and
    put back on exit. A library that was not imported on entry, or CUDA that was not initialized, has no
    state to put back, and whatever seeding the block did to it must not outlast the block: such a
    generator is reseeded from entropy, the way it would have started.

    The state is process-global, so this is not safe to use from several threads at once.
    """
    py = random.getstate()
    np_state = sys.modules["numpy"].random.get_state() if "numpy" in sys.modules else None
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch else None
    cuda_state = torch.cuda.get_rng_state_all() if torch and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        numpy = sys.modules.get("numpy")
        if numpy is not None:
            if np_state is not None:
                numpy.random.set_state(np_state)
            else:
                numpy.random.seed()  # imported inside the block: back to entropy
        torch = sys.modules.get("torch")
        if torch is not None:
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            else:
                torch.seed()  # imported inside the block: back to entropy
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            else:
                torch.cuda.seed_all()  # CUDA was not initialized on entry; queued if it still is not
