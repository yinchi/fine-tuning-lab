"""A whole conversation trained in one pass equals training on each truncated prefix.

With causal attention, the loss on a turn's tokens depends only on the tokens before them. So the summed loss
over all assistant turns of one sequence, and its gradient, equal the sums over the sequence truncated after
each turn (with the loss on that turn only). We test this on a tiny Qwen3 in float32.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

# Assistant answers occupy these positions of a 30-token sequence; everything else is prompt.
ANSWERS = [range(6, 10), range(16, 20), range(26, 30)]


@pytest.fixture(scope="module")
def model_and_ids():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=50, tie_word_embeddings=True,
                      initializer_range=0.2)  # larger weights, so that the context visibly matters
    model = AutoModelForCausalLM.from_config(cfg).float().eval()
    return model, torch.randint(0, 50, (1, 30))


def summed_loss(model, ids, positions):
    """Cross-entropy summed over the tokens at `positions`, each predicted from the tokens before it."""
    logits = model(input_ids=ids).logits[0]
    pos = torch.tensor(list(positions))
    return torch.nn.functional.cross_entropy(logits[pos - 1], ids[0, pos], reduction="sum")


def prefix_losses(model, ids):
    """One loss per turn: the sequence cut right after the turn, loss on that turn only."""
    return [summed_loss(model, ids[:, : a[-1] + 1], a) for a in ANSWERS]


def test_whole_conversation_loss_equals_the_sum_over_prefixes(model_and_ids):
    model, ids = model_and_ids
    whole = summed_loss(model, ids, [p for a in ANSWERS for p in a])
    parts = prefix_losses(model, ids)
    assert torch.allclose(whole, sum(parts), rtol=1e-5, atol=1e-5)


def test_whole_conversation_gradient_equals_the_sum_over_prefixes(model_and_ids):
    model, ids = model_and_ids
    params = [p for p in model.parameters() if p.requires_grad]
    whole = torch.autograd.grad(summed_loss(model, ids, [p for a in ANSWERS for p in a]), params)
    parts = [torch.autograd.grad(loss, params, allow_unused=True) for loss in prefix_losses(model, ids)]
    for i, g in enumerate(whole):
        total = sum(p[i] for p in parts if p[i] is not None)
        assert torch.allclose(g, total, rtol=1e-4, atol=1e-5), f"gradient {i} differs"


def test_truncating_the_start_is_not_equivalent(model_and_ids):
    """The limit of the claim: only the end can be cut. Dropping early tokens changes every later loss."""
    model, ids = model_and_ids
    last = ANSWERS[-1]
    full = summed_loss(model, ids, last)
    shifted = summed_loss(model, ids[:, 5:], [p - 5 for p in last])
    assert not torch.allclose(full, shifted, rtol=1e-3, atol=1e-3)
