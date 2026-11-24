"""The bootstrap: identical systems, a planted difference, and the cost of ignoring conversations."""

import numpy as np
import pytest

from convfinqa.stats import _by_page, bootstrap_accuracy, paired_difference

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


def test_conversations_written_from_one_page_are_one_cluster():
    scores = [
        {"id": i, "turn": 0, "correct": True}
        for i in (
            "Single_AB/2001/page_1.pdf-1", "Single_AB/2001/page_1.pdf-2", "Double_AB/2001/page_1.pdf",
            "Single_CD/2002/page_7.pdf-3",
        )
    ]
    pages, n_turns, n_correct, _ = _by_page(scores)
    assert pages == ["AB/2001/page_1.pdf", "CD/2002/page_7.pdf"]
    assert list(n_turns) == [3, 1] and list(n_correct) == [3, 1]


def test_the_interval_is_wider_when_conversations_of_a_page_share_their_outcome():
    """Three conversations per page, all right or all wrong together: resampling pages sees only 100 clusters."""
    rng = np.random.default_rng(0)
    page_ok = rng.random(100) < 0.5
    together = [
        {"id": f"Single_P{p}/2000/page_{p}.pdf-{j}", "turn": 0, "correct": bool(page_ok[p])}
        for p in range(100) for j in range(3)
    ]
    separate = [  # the same 300 outcomes, each on a page of its own
        {"id": f"Single_Q{k}/2000/page_{k}.pdf-0", "turn": 0, "correct": s["correct"]}
        for k, s in enumerate(together)
    ]
    w = lambda r: r["high"] - r["low"]
    assert w(bootstrap_accuracy(together)) > 1.4 * w(bootstrap_accuracy(separate))
