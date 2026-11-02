"""Where a wrong program parts from the gold one."""

import math

import pytest
import torch

from convfinqa.divergence import choice_at, first_divergence


def test_the_first_difference_is_found():
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([5], [6]) == 0


def test_a_proper_prefix_differs_where_it_ends():
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2, 3], [1, 2]) == 2


def test_rank_and_ratio_of_two_candidates():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = choice_at(logits, gold_id=1, pred_id=0)
    assert out["rank_gold"] == 2 and out["rank_pred"] == 1 and out["top"] == 0
    assert out["gold_over_pred"] == pytest.approx(0.6)
    assert out["logp_pred"] == pytest.approx(math.log(0.5))
    assert choice_at(logits, 3, 2)["rank_gold"] == 4


def test_logits_need_not_be_normalised_and_ties_share_the_better_rank():
    out = choice_at(torch.tensor([2.0, 2.0, 0.0]), 1, 0)
    assert out["rank_gold"] == out["rank_pred"] == 1
    assert out["gold_over_pred"] == pytest.approx(1.0)
