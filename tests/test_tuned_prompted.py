import json
from pathlib import Path

import pytest

from convfinqa.tuned_prompted import baseline_prompt

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "baseline.json").read_text())
POOL = {i: {"id": i} for i in ("a", "b", "c", "d")}
SETS = {"sets": {"0": ["a", "b", "c"], "1": ["d", "a", "b"]}}


def test_a_variant_with_examples_takes_the_first_of_the_set():
    system, examples = baseline_prompt(CONFIG, SETS, POOL, "e3p", 1)
    assert system == CONFIG["system"]["plain"]
    assert [e["id"] for e in examples] == ["d", "a", "b"]
    assert [e["id"] for e in baseline_prompt(CONFIG, SETS, POOL, "e2d", 0)[1]] == ["a", "b"]


def test_the_variant_decides_the_system_message():
    assert baseline_prompt(CONFIG, SETS, POOL, "e3d", 0)[0] == CONFIG["system"]["description"]
    assert baseline_prompt(CONFIG, SETS, POOL, "e3f", 0)[0] == CONFIG["system"]["flat"]


def test_a_variant_without_examples_has_none_and_only_set_zero():
    assert baseline_prompt(CONFIG, SETS, POOL, "e0p", 0) == (CONFIG["system"]["plain"], [])
    with pytest.raises(ValueError):
        baseline_prompt(CONFIG, SETS, POOL, "e0p", 1)
