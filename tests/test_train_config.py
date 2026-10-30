"""The training settings: the common ones, overridden per run."""

import json
from pathlib import Path

import pytest

from convfinqa.train_config import REQUIRED, run_config

CFG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "train.json").read_text())


def test_every_defined_run_has_every_setting():
    for run in CFG["runs"]:
        assert set(run_config(CFG, run)) == set(REQUIRED)


def test_a_run_overrides_the_common_settings_and_nothing_else():
    smoke, main = run_config(CFG, "smoke"), run_config(CFG, "main")
    assert {k for k in smoke if smoke[k] != main[k]} == {"model", "max_steps", "eval_every"}
    assert main["max_steps"] is None and smoke["max_steps"] == 50


def test_the_lora_entry_is_merged_key_by_key():
    cfg = {"common": {**CFG["common"]}, "runs": {"wide": {"model": "m", "lora": {"r": 64, "alpha": 128}}}}
    lora = run_config(cfg, "wide")["lora"]
    assert (lora["r"], lora["alpha"], lora["dropout"]) == (64, 128, CFG["common"]["lora"]["dropout"])


def test_an_unknown_run_or_setting_is_refused():
    with pytest.raises(KeyError):
        run_config(CFG, "nope")
    cfg = {"common": CFG["common"], "runs": {"odd": {"model": "m", "momentum": 0.9}}}
    with pytest.raises(ValueError, match="momentum"):
        run_config(cfg, "odd")
    with pytest.raises(ValueError, match="missing"):
        run_config({"common": {}, "runs": {"empty": {}}}, "empty")


def test_a_run_changes_nothing_in_the_others():
    before = run_config(CFG, "main")
    cfg = json.loads(json.dumps(CFG))
    cfg["runs"]["extra"] = {"model": "m", "lr": 1.0}
    assert run_config(cfg, "main") == before


def test_the_final_step_is_evaluated_even_when_it_is_not_a_multiple_of_eval_steps():
    from types import SimpleNamespace as NS

    from convfinqa.finetune import _eval_at_end_callback

    callback = _eval_at_end_callback()
    args = NS(eval_steps=50)
    for step, expected in [(49, False), (50, False), (51, False), (342, True), (300, False)]:
        control = NS(should_evaluate=False)
        callback.on_step_end(args, NS(global_step=step, max_steps=342), control)
        assert control.should_evaluate is expected
    control = NS(should_evaluate=False)
    callback.on_step_end(args, NS(global_step=50, max_steps=50), control)
    assert control.should_evaluate is False       # a multiple: the Trainer evaluates there itself
