"""The share of a gap that a prompt closes, and the comparison of a system on two splits."""

import numpy as np
import pytest

from convfinqa.final_results import outcome_counts, share_of_gap_closed, split_difference

N_CONV, TURNS = 300, 4


def system(p, seed, n_conv=N_CONV, prefix="c"):
    """Scores of a system whose turns are correct with probability p (a number, or one per conversation)."""
    rng = np.random.default_rng(seed)
    p = np.broadcast_to(p, (n_conv,))
    return [
        {"id": f"{prefix}{i}", "turn": t, "correct": bool(rng.random() < p[i])}
        for i in range(n_conv) for t in range(TURNS)
    ]


def test_share_of_a_gap_is_the_point_estimate_and_its_interval_contains_it():
    low, mid, high = system(0.1, 1), system(0.5, 2), system(0.9, 3)
    r = share_of_gap_closed(low, mid, high)
    k = [sum(s["correct"] for s in x) for x in (low, mid, high)]
    assert r["share"] == pytest.approx((k[1] - k[0]) / (k[2] - k[0]))
    assert r["low"] < r["share"] < r["high"]
    assert 0.4 < r["share"] < 0.6 and r["gap"] == pytest.approx(0.8, abs=0.05)


def test_a_middle_system_at_the_floor_or_at_the_ceiling_closes_none_or_all():
    low, high = system(0.1, 1), system(0.9, 3)
    assert share_of_gap_closed(low, list(low), high)["share"] == 0
    assert share_of_gap_closed(low, list(high), high)["share"] == 1


def test_share_needs_the_same_turns_and_a_ceiling_above_the_floor():
    low, mid, high = system(0.1, 1), system(0.5, 2), system(0.9, 3)
    with pytest.raises(ValueError, match="same turns"):
        share_of_gap_closed(low, mid[:-1], high)
    with pytest.raises(ValueError, match="above the floor"):
        share_of_gap_closed(high, mid, low)


def test_the_same_system_on_two_splits_with_no_difference_straddles_zero():
    a = system(0.7, 1, prefix="v")
    b = system(0.7, 2, n_conv=400, prefix="d")
    r = split_difference(a, b)
    assert not r["excludes_zero"] and r["low"] < 0 < r["high"]


def test_a_planted_difference_between_splits_is_found():
    a = system(0.9, 1, prefix="v")
    b = system(0.6, 2, n_conv=400, prefix="d")
    r = split_difference(a, b)
    assert r["excludes_zero"] and r["difference"] == pytest.approx(0.3, abs=0.05)
    assert r["low"] > 0.2


def test_outcomes_group_every_kind_that_does_not_run_as_invalid():
    summary = {
        "n": 10,
        "by_kind": {"correct": 4, "wrong_numbers": 2, "wrong_operations": 1, "code_fence": 2, "syntax": 1},
    }
    assert outcome_counts(summary) == {
        "correct": 4, "invalid output": 3, "wrong operations": 1, "wrong numbers": 2, "wrong answer, right shape": 0,
    }


def test_outcomes_that_do_not_add_up_are_refused():
    with pytest.raises(ValueError, match="add up"):
        outcome_counts({"n": 11, "by_kind": {"correct": 4}})
