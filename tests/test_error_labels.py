"""The label of a wrong turn."""

from convfinqa.error_labels import (
    INHERITED, INVALID, NOT_FROM_PAGE, NUMBERS, OPERATION, OTHER, REVERSED, SCALE, SIGN, label_turn, label_turns,
)

OK = {"scale": None, "lit_grounded": True}


def test_each_label():
    assert label_turn("syntax", OK, "add(1,", "add(1, 2)", 3.0) == INVALID
    assert label_turn("wrong_numbers", OK, "add(1, 3)", "add(1, 2)", 3.0, right_with_gold_history=True) == INHERITED
    assert label_turn("wrong_numbers", {**OK, "scale": "factor_100"}, "5", "5", 3.0) == SCALE
    assert label_turn("wrong_numbers", OK, "subtract(4027, 1280)", "subtract(-4027, -1280)", -2747.0) == SIGN
    assert label_turn("wrong_answer", OK, "subtract(1, 5)", "subtract(5, 1)", 4.0) == REVERSED
    assert label_turn("wrong_numbers", {**OK, "lit_grounded": False}, "add(1, 9)", "add(1, 2)", 3.0) == NOT_FROM_PAGE
    assert label_turn("wrong_numbers", OK, "add(1, 9)", "add(1, 2)", 3.0) == NUMBERS
    assert label_turn("wrong_answer", OK, "352", "348", 348.0) == NUMBERS          # another number where one was asked for
    assert label_turn("wrong_operations", OK, "add(1, 2)", "subtract(1, 2)", -1.0) == OPERATION
    assert label_turn("wrong_answer", OK, "add(1, 2), add(#0, 3)", "add(1, 3), add(#0, 2)", 7.0) == OTHER


def test_the_first_label_that_applies_wins():
    assert label_turn("syntax", OK, "x", "add(1, 2)", 3.0, True) == INVALID
    assert label_turn("wrong_numbers", {"scale": "other_power", "lit_grounded": False}, "5", "5", 1.0, True) == INHERITED


def test_label_turns_covers_the_wrong_turns_only():
    scores = [
        {"id": "a", "turn": 0, "correct": True, "kind": "correct"},
        {"id": "a", "turn": 1, "correct": False, "kind": "wrong_numbers"},
        {"id": "a", "turn": 2, "correct": False, "kind": "wrong_numbers"},
    ]
    checks = [{"id": "a", "turn": k, **OK} for k in range(3)]
    rows = [{"id": "a", "turn": k, "output": "add(1, 9)"} for k in range(3)]
    gold = [{"id": "a", "turn": k, "program": "add(1, 2)", "answer": 3.0} for k in range(3)]
    with_gold = [{"id": "a", "turn": k, "correct": k == 2} for k in range(3)]
    assert label_turns(scores, checks, rows, gold, with_gold) == {("a", 1): NUMBERS, ("a", 2): INHERITED}
    assert label_turns(scores, checks, rows, gold) == {("a", 1): NUMBERS, ("a", 2): NUMBERS}
