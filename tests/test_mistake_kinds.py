"""Which kinds of difference explain a generated program, and the smallest sets that do."""

from convfinqa.mistake_kinds import count_minimal_sets, explained_by, minimal_explanations

F = frozenset


def test_a_matching_program_needs_nothing():
    assert minimal_explanations("add(1, 2)", "add(2, 1)") == [F()]


def test_each_kind_alone():
    assert minimal_explanations("subtract(206, 100)", "subtract(206, const_100)") == [F({"constants"})]
    assert minimal_explanations("multiply(2.0, 3)", "multiply(2, 3)") == [F({"zeros"})]
    assert minimal_explanations("subtract(4027, 1280)", "subtract(-4027, -1280)") == [F({"signs"})]
    assert minimal_explanations("subtract(1651, 1127)", "subtract(1127, 1651)") == [F({"order"})]


def test_two_kinds_at_once_need_both():
    assert minimal_explanations("subtract(331.43, 100.00)", "subtract(331.43, const_100)") == [F({"constants", "zeros"})]
    assert explained_by("subtract(331.43, 100.00)", "subtract(331.43, const_100)", F({"constants", "zeros"}))
    assert not explained_by("subtract(331.43, 100.00)", "subtract(331.43, const_100)", F({"constants"}))
    assert minimal_explanations("subtract(5, 3)", "subtract(-3, -5)") == [F({"signs", "order"})]


def test_other_differences_and_invalid_outputs_are_unexplained():
    assert minimal_explanations("add(1, 2)", "add(1, 3)") == []
    assert minimal_explanations("add(1,", "add(1, 2)") == []
    assert minimal_explanations("subtract(5, 3)", "add(5, 3)") == []


def test_sets_are_minimal():
    sets = minimal_explanations("subtract(206, 100)", "subtract(206, const_100)")
    assert all(not a < b for a in sets for b in sets)


def test_counting():
    pairs = [
        ("add(1, 2)", "add(2, 1)"), ("subtract(206, 100)", "subtract(206, const_100)"),
        ("subtract(4027, 1280)", "subtract(-4027, -1280)"), ("add(1, 2)", "add(1, 3)"), ("x", "add(1, 2)"),
    ]
    out = count_minimal_sets(pairs)
    assert out == {"n": 5, "equal": 1, "by_set": {("constants",): 1, ("signs",): 1}, "multiple": 0, "unexplained": 2}
