import json
from pathlib import Path

import pytest

from convfinqa import tuned_eval
from convfinqa.evaluate import NotFrozenError

ROOT = Path(__file__).resolve().parents[1]


def test_val_is_returned_and_dev_is_refused_before_the_freeze(tmp_path):
    val, dev = [{"id": "v"}], [{"id": "d"}]
    assert tuned_eval.split_conversations("val", val, dev) is val
    frozen = tmp_path / "FROZEN"
    with pytest.raises(NotFrozenError):
        tuned_eval.split_conversations("dev", val, dev, frozen)
    frozen.write_text("frozen")
    assert tuned_eval.split_conversations("dev", val, dev, frozen) is dev


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError):
        tuned_eval.split_conversations("test", [], [])


def test_the_generation_limit_is_the_baselines():
    baseline = json.loads((ROOT / "configs" / "baseline.json").read_text())
    assert tuned_eval.MAX_NEW_TOKENS == baseline["max_new_tokens"]
