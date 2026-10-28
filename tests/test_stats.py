"""The bootstrap: identical systems, a planted difference, and the cost of ignoring conversations."""

import numpy as np
import pytest

from convfinqa.stats import bootstrap_accuracy, paired_difference, turn_level_interval

N_CONV, TURNS = 300, 4


def system(p_by_conv, seed):
    """Scores of a system whose turns in conversation i are correct with probability p_by_conv[i]."""
    rng = np.random.default_rng(seed)
    return [
        {"id": f"c{i}", "turn": t, "correct": bool(rng.random() < p_by_conv[i])}
        for i in range(N_CONV) for t in range(TURNS)
    ]


def correlated_p(seed=0):
    """Conversations differ a lot in difficulty, so their turns' outcomes are correlated."""
    return np.random.default_rng(seed).choice([0.2, 0.9], size=N_CONV)


def test_identical_systems_give_an_interval_centred_on_zero():
    a = system(correlated_p(), seed=1)
    d = paired_difference(a, list(a))
    assert d["difference"] == 0 and d["low"] == 0 == d["high"] and d["p"] == 1.0 and not d["excludes_zero"]


def test_two_systems_with_the_same_accuracy_are_not_declared_different():
    p = correlated_p()
    d = paired_difference(system(p, 1), system(p, 2))
    assert not d["excludes_zero"] and d["low"] < 0 < d["high"] and d["p"] > 0.05


def test_a_planted_five_point_difference_is_detected():
    a = system(np.full(N_CONV, 0.6), seed=3)
    rng = np.random.default_rng(4)
    b = []
    for s in a:   # B repairs one wrong turn in 8 of A's: +5 points
        wrong_fixed = (not s["correct"]) and rng.random() < 0.125
        b.append({**s, "correct": s["correct"] or wrong_fixed})
    d = paired_difference(b, a)
    assert d["excludes_zero"] and d["low"] > 0 and d["p"] < 0.01
    assert d["difference"] == pytest.approx(0.05, abs=0.02)


def test_the_conversation_bootstrap_is_wider_than_the_turn_bootstrap_when_outcomes_are_correlated():
    a = system(correlated_p(), seed=5)
    by_conversation, by_turn = bootstrap_accuracy(a), turn_level_interval(a)
    assert by_conversation["accuracy"] == pytest.approx(by_turn["accuracy"])
    assert (by_conversation["high"] - by_conversation["low"]) > 1.5 * (by_turn["high"] - by_turn["low"])


def test_the_interval_contains_the_accuracy_and_the_resampling_is_reproducible():
    a = system(np.full(N_CONV, 0.7), seed=6)
    r = bootstrap_accuracy(a, seed=9)
    assert r["low"] < r["accuracy"] < r["high"]
    assert r == bootstrap_accuracy(a, seed=9) and r != bootstrap_accuracy(a, seed=10)
    assert r["accuracy"] == pytest.approx(np.mean([s["correct"] for s in a]))


def test_systems_must_cover_the_same_turns():
    a = system(np.full(N_CONV, 0.5), seed=7)
    with pytest.raises(ValueError, match="same turns"):
        paired_difference(a, a[:-1])
