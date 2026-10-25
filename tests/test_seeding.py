"""Seeding a build's random generators, and putting them back afterwards."""

import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from convfinqa.artifacts.seeding import preserved_rng_state, seed_everything


def draws():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_the_same_seed_gives_the_same_draws_whatever_happened_before():
    seed_everything(7)
    first = draws()
    for _ in range(5):  # use the generators in between
        draws()
    seed_everything(7)
    assert draws() == first


def test_different_seeds_give_different_draws():
    seed_everything(7)
    a = draws()
    seed_everything(8)
    assert draws() != a


@pytest.mark.parametrize("seed", [0, 1, 2**32, 2**32 + 5, 2**63 - 1, -3])
def test_any_integer_seed_is_accepted_and_repeatable(seed):
    seed_everything(seed)
    first = draws()
    seed_everything(seed)
    assert draws() == first


def test_the_numpy_seed_wraps_to_32_bits():
    seed_everything(5)
    a = float(np.random.rand())
    seed_everything(5 + 2**32)  # the legacy generator takes seeds modulo 2**32
    assert float(np.random.rand()) == a


def test_generators_with_their_own_state_are_not_seeded():
    seed_everything(3)
    first = np.random.default_rng().random()
    seed_everything(3)
    assert np.random.default_rng().random() != first  # they must be given the seed explicitly


def test_the_block_restores_the_state_that_was_there():
    random.seed(11), np.random.seed(11), torch.manual_seed(11)
    expected_next = draws()
    random.seed(11), np.random.seed(11), torch.manual_seed(11)
    with preserved_rng_state():
        seed_everything(99)
        draws()
    assert draws() == expected_next


def test_the_state_is_restored_when_the_block_raises():
    random.seed(11), np.random.seed(11), torch.manual_seed(11)
    expected_next = draws()
    random.seed(11), np.random.seed(11), torch.manual_seed(11)
    with pytest.raises(RuntimeError):
        with preserved_rng_state():
            seed_everything(99)
            raise RuntimeError("the build failed")
    assert draws() == expected_next


FRESH_INTERPRETER = """
import sys
from convfinqa.artifacts.seeding import preserved_rng_state, seed_everything
assert "numpy" not in sys.modules and "torch" not in sys.modules
with preserved_rng_state():
    seed_everything(5)  # imports numpy and torch inside the block
import numpy as np, torch
after = (np.random.get_state()[1][:4].tolist(), torch.initial_seed())
np.random.seed(5); torch.manual_seed(5)
seeded = (np.random.get_state()[1][:4].tolist(), torch.initial_seed())
print(after[0] == seeded[0], after[1] == seeded[1])
"""


def test_libraries_first_imported_inside_the_block_do_not_keep_the_build_seed():
    result = subprocess.run(
        [sys.executable, "-c", FRESH_INTERPRETER], capture_output=True, text=True, check=True, env=dict(os.environ),
    )
    assert result.stdout.split() == ["False", "False"]  # neither generator is left seeded with 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_draws_are_repeatable_and_restored():
    seed_everything(3)
    first = torch.rand(3, device="cuda").tolist()
    seed_everything(3)
    assert torch.rand(3, device="cuda").tolist() == first
    before = torch.cuda.get_rng_state_all()
    with preserved_rng_state():
        seed_everything(99)
        torch.rand(3, device="cuda")
    assert all((a == b).all() for a, b in zip(before, torch.cuda.get_rng_state_all()))
