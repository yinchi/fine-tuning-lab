"""Reading the baseline runs: nested calls, the invalid outputs, and the rule that chooses a prompt."""

import pytest

from convfinqa.baseline_report import (
    choose_variant, comparison_rows, has_nested_call, nested_among_invalid, turn_level_paired,
)
from convfinqa.stats import paired_difference


def turn(conv, k, kind):
    return {"id": conv, "turn": k, "kind": kind, "correct": kind == "correct"}


def scores_with(correct_by_conv: dict[str, list[bool]]) -> list[dict]:
    return [turn(c, k, "correct" if ok else "wrong_answer") for c, flags in correct_by_conv.items() for k, ok in enumerate(flags)]


@pytest.mark.parametrize("text,nested", [
    ("divide(subtract(5, 3), 3)", True),
    ("subtract(5, 3), divide(#0, 3)", False),
    ("add(add(1, 2), 3)", True),
    ("5.5", False),
    ("greater(#0, 0)", False),
    ("```\nadd(1, 2)\n```", False),
    ("add(1, 2))), subtract(3, 4)", False),   # stray closing brackets do not push the depth below zero
    ("", False),
])
def test_has_nested_call(text, nested):
    assert has_nested_call(text) is nested


def test_nested_among_invalid_counts_only_outputs_that_do_not_run():
    scores = [turn("a", 0, "syntax"), turn("a", 1, "correct"), turn("b", 0, "syntax"), turn("b", 1, "wrong_operations")]
    rows = [
        {"id": "a", "turn": 0, "output": "divide(subtract(5, 3), 3)"}, {"id": "a", "turn": 1, "output": "add(1, 2)"},
        {"id": "b", "turn": 0, "output": "divide 5 by 3"}, {"id": "b", "turn": 1, "output": "add(add(1, 2), 3)"},
    ]
    assert nested_among_invalid(scores, rows) == 1    # b/1 runs (wrongly), so it is not an invalid output


def test_the_choice_prefers_fewer_examples_then_the_plain_message_among_ties():
    convs = {f"c{i}": [True, True] for i in range(40)}
    flip = lambda n: {c: ([False, True] if i < n else flags) for i, (c, flags) in enumerate(convs.items())}
    scores = {"e3p": scores_with(convs), "e3d": scores_with(flip(1)), "e2p": scores_with(flip(1)), "e0p": scores_with(flip(30))}
    result = choose_variant(scores, n_boot=500)
    assert result["best"] == "e3p"
    assert result["tied"] == ["e2p", "e3p", "e3d"]     # e0p is clearly worse
    assert result["chosen"] == "e2p"
    assert set(result["leads"]) == {"e3d", "e2p", "e0p"}


def test_a_clear_winner_is_chosen_even_with_more_examples():
    good = scores_with({f"c{i}": [True, True] for i in range(40)})
    bad = scores_with({f"c{i}": [False, False] for i in range(40)})
    assert choose_variant({"e3p": good, "e2p": bad}, n_boot=500)["chosen"] == "e3p"


def test_comparison_rows():
    scores = {"e3p": [turn("a", 0, "syntax"), turn("a", 1, "correct")]}
    preds = {"e3p": [
        {"id": "a", "turn": 0, "output": "add(add(1, 2), 3)", "prompt_tokens": 100, "new_tokens": 5},
        {"id": "a", "turn": 1, "output": "5", "prompt_tokens": 300, "new_tokens": 1},
    ]}
    (row,) = comparison_rows(scores, preds)
    assert (row["variant"], row["accuracy"], row["invalid"], row["nested"], row["nested_share"]) == ("e3p", 0.5, 1, 1, 1.0)
    assert row["prompt_tokens"] == 200


def test_turn_level_paired_matches_the_hand_computation():
    a = scores_with({"c1": [True, True], "c2": [True, False]})
    b = scores_with({"c1": [True, False], "c2": [False, False]})
    r = turn_level_paired(a, b)
    # differences 0, 1, 1, 0: mean 0.5, sample sd sqrt(1/3), se = sd / 2
    assert r["difference"] == pytest.approx(0.5)
    assert r["half_width"] == pytest.approx(1.96 * (1 / 3) ** 0.5 / 2)
    assert r["low"] < 0.5 < r["high"]


def test_turn_level_paired_is_narrower_than_the_cluster_bootstrap_when_conversations_are_homogeneous():
    # each conversation is all right for A and all wrong for B, or the same for both: outcomes cluster
    convs = {f"c{i}": [i % 2 == 0] * 4 for i in range(30)}
    a = scores_with(convs)
    b = scores_with({c: [False] * 4 for c in convs})
    assert turn_level_paired(a, b)["half_width"] < (lambda d: (d["high"] - d["low"]) / 2)(paired_difference(a, b))


def test_turn_level_paired_refuses_different_turns():
    with pytest.raises(ValueError):
        turn_level_paired(scores_with({"c": [True]}), scores_with({"d": [True]}))
