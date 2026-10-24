"""Seeding and process-state helpers used around a build.

Contains context managers for capturing and restoring RNG state and the current working directory,
such that temporary changes to RNG state or the working directory do not affect the rest of the
build.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import random
import sys
from collections.abc import Generator
from pathlib import Path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) so a build does not depend on earlier RNG use."""
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed % (2**32))
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def preserved_rng_state() -> Generator[None, None, None]:
    """Restore the global RNG state afterwards, so an in-kernel build never perturbs a chapter.
    
    **Usage:**
    ```
    with preserved_rng_state():
        # code that uses RNG
        ...
    ```

    Tracks and restores the RNG state for Python, NumPy, and torch (CPU and CUDA).
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
        if np_state is not None:
            sys.modules["numpy"].random.set_state(np_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextlib.contextmanager
def chdir(path: Path) -> Generator[None, None, None]:
    """Temporarily change the current working directory to `path`.
    
    **Usage:**
    ```
    with chdir(Path("/some/directory")):
        # code that runs in the specified directory
        ...
    ```
    """
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
