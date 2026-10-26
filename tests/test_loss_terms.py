"""The by-hand loss equals the library's; leaving out the mask changes it."""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from convfinqa.probing import loss_terms


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=50, tie_word_embeddings=True,
                      initializer_range=0.2)
    return AutoModelForCausalLM.from_config(cfg).float().eval()


def example():
    torch.manual_seed(1)
    ids = torch.randint(0, 50, (1, 24))
    labels = torch.full_like(ids, -100)
    labels[0, 15:20] = ids[0, 15:20]  # a five-token answer in the middle
    return ids, labels


def test_by_hand_equals_the_library(model):
    out = loss_terms(model, *example())
    assert out["by_hand"] == pytest.approx(out["library"], abs=1e-5)
    assert len(out["surprisals"]) == 5
    assert sum(out["surprisals"]) / 5 == pytest.approx(out["by_hand"], abs=1e-6)


def test_no_mask_averages_over_every_position(model):
    ids, labels = example()
    out = loss_terms(model, ids, labels)
    assert abs(out["no_mask"] - out["by_hand"]) > 1e-3
    everything = loss_terms(model, ids, ids.clone())  # every position counts
    assert out["no_mask"] == pytest.approx(everything["by_hand"], abs=1e-6)
