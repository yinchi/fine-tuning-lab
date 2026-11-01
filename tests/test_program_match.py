"""Program accuracy: the official symbolic comparison."""

import pytest

from convfinqa.dsl import executor
from convfinqa.program_match import add_program_match, equal_program, equal_up_to_constants, program_expression, same_text


def test_the_order_of_a_sum_or_a_product_does_not_matter():
    assert equal_program("add(5, 7)", "add(7, 5)")
    assert equal_program("multiply(a1, 3)", "multiply(3, a1)")
    assert same_text("add(5, 7)", "add(5, 7)") and not same_text("add(7, 5)", "add(5, 7)")


def test_the_order_of_a_difference_or_a_quotient_does_matter():
    assert not equal_program("subtract(5, 7)", "subtract(7, 5)")
    assert not equal_program("divide(5, 7)", "divide(7, 5)")
    assert not equal_program("exp(2, 3)", "exp(3, 2)")
    assert not equal_program("greater(5, 7)", "greater(7, 5)")


def test_arguments_are_compared_as_text():
    assert not equal_program("divide(2.70, 3)", "divide(2.7, 3)")
    assert not equal_program("divide(17%, 100)", "divide(0.17, 100)")
    assert not equal_program("add(const_1, 3)", "add(1, 3)")
    assert not equal_program("5", "5.0")
    assert equal_program("5", " 5 ")


def test_references_are_expanded_so_the_split_into_steps_does_not_matter():
    assert equal_program("add(1, 2), add(#0, 3)", "add(3, 2), add(#0, 1)")      # a sum of three, in any order
    assert equal_program("add(1, 2), add(#0, 3)", "add(2, 1), add(3, #0)")
    assert not equal_program("add(1, 2), add(#0, 3)", "add(1, 2), multiply(#0, 3)")
    assert equal_program("subtract(8, 3), divide(#0, 3)", "subtract(8, 3), divide(#0, 3)")
    assert not equal_program("subtract(8, 3), divide(#0, 3)", "subtract(8, 3), divide(3, #0)")


def test_a_step_the_result_does_not_depend_on_does_not_count():
    assert equal_program("add(9, 9), add(1, 2)", "add(2, 1)")
    assert equal_program("add(1, 2), add(3, 4), add(#0, 5)", "add(1, 2), add(7, 8), add(#0, 5)")


def test_cancellation_is_not_equality():
    assert not equal_program("subtract(5, 5)", "subtract(7, 7)")
    assert not equal_program("divide(5, 5)", "divide(7, 7)")
    assert equal_program("subtract(5, 5)", "subtract(5, 5)")


def test_an_invalid_prediction_is_never_equal():
    for bad in ("", "add(1)", "foo(1, 2)", "add(1, #0)", "add(1, #x)", "```add(1, 2)```", "add(1, 2) because"):
        assert not equal_program(bad, "add(1, 2)")
        assert not same_text(bad, "add(1, 2)")


def test_a_bad_gold_program_is_an_error_not_a_mismatch():
    with pytest.raises(executor.ProgramError):
        equal_program("add(1, 2)", "add(1, #3)")


def test_expression_counts_the_arguments_it_uses():
    _, used = program_expression("add(1, 2), multiply(#0, 1)")
    assert used == {"1": 2, "2": 1}


def test_scores_get_two_flags():
    gold = [{"id": "a", "turn": 0, "program": "add(1, 2)"}, {"id": "a", "turn": 1, "program": "5"}]
    rows = [{"id": "a", "turn": 0, "output": "add(2, 1)"}, {"id": "a", "turn": 1, "output": "5.0"}]
    scores = [{"id": "a", "turn": 0, "correct": True}, {"id": "a", "turn": 1, "correct": True}]
    out = add_program_match(scores, rows, gold)
    assert [(s["program_correct"], s["text_match"]) for s in out] == [(True, False), (False, False)]
    assert [s["correct"] for s in out] == [True, True]


def test_a_constant_and_the_plain_number_are_equal_up_to_constants_only():
    assert not equal_program("subtract(206, 100)", "subtract(206, const_100)")
    assert equal_up_to_constants("subtract(206, 100)", "subtract(206, const_100)")
    assert equal_up_to_constants("add(const_m1, 3)", "add(3, -1)")
    assert not equal_up_to_constants("subtract(206, 100.00)", "subtract(206, const_100)")
    assert not equal_up_to_constants("subtract(206, 1000)", "subtract(206, const_100)")
