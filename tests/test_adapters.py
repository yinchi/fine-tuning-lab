"""LoRA adapters start as the base model, and merging one into the weights gives the same function."""

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from convfinqa.adapters import merge_comparison, randomize_b

pytest.importorskip("peft")


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=50, tie_word_embeddings=True)
    return AutoModelForCausalLM.from_config(cfg).float().eval()


IDS = torch.randint(0, 50, (1, 12))


def test_adapter_starts_as_the_base_model_and_merging_is_faithful():
    out = merge_comparison(tiny_model(), IDS, r=4, alpha=8, b_std=0.05, seed=0)
    assert out["step0_max_abs_diff"] == 0.0
    assert out["adapter_effect_max_abs"] > 1e-3                      # the random adapter does something
    assert out["merge_max_abs_diff"] < 1e-4 * max(1.0, out["adapter_effect_max_abs"])
    assert out["merge_top1_agreement"] == 1.0
    assert out["adapted_modules"] == 2 * 7                           # two layers, seven projections each


def test_randomize_b_depends_only_on_the_seed():
    from peft import LoraConfig, get_peft_model

    def values(seed):
        m = get_peft_model(tiny_model(), LoraConfig(r=4, target_modules="all-linear", task_type="CAUSAL_LM"))
        randomize_b(m, 0.05, seed)
        return [p.clone() for n, p in m.named_parameters() if "lora_B" in n]

    first, again, other = values(0), values(0), values(1)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
