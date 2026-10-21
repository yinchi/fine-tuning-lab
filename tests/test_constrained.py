import torch

from convfinqa.constrained import AllowedTokens


def test_only_allowed_tokens_survive():
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    out = AllowedTokens([1, 3])(torch.zeros(2, 1, dtype=torch.long), scores)
    assert torch.isinf(out[:, [0, 2, 4]]).all() and (out[:, [0, 2, 4]] < 0).all()
    assert torch.equal(out[:, [1, 3]], scores[:, [1, 3]])


def test_probabilities_renormalise_over_the_allowed_tokens():
    p = torch.softmax(AllowedTokens([0, 2])(torch.zeros(1, 1, dtype=torch.long), torch.tensor([[1.0, 9.0, 1.0]])), -1)
    assert p[0, 1] == 0 and torch.allclose(p[0, [0, 2]], torch.tensor([0.5, 0.5]))
