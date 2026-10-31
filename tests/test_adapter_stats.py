import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from convfinqa.adapter_stats import adapter_update_stats, update_singular_values


def test_singular_values_match_the_dense_update():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 30, generator=g), torch.randn(20, 4, generator=g)
    got = update_singular_values(a, b, 2.0)
    want = torch.linalg.svdvals(2.0 * (b @ a).double())[:4]
    assert torch.allclose(got, want, atol=1e-9)


def test_stats_of_a_saved_adapter(tmp_path):
    g = torch.Generator().manual_seed(1)
    tensors = {}
    for layer in (0, 1):
        for name, prefix in (("q_proj", "self_attn"), ("up_proj", "mlp")):
            base = f"base_model.model.model.layers.{layer}.{prefix}.{name}"
            tensors[f"{base}.lora_A.weight"] = torch.randn(2, 8, generator=g)
            tensors[f"{base}.lora_B.weight"] = torch.randn(6, 2, generator=g)
    safetensors.save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    stats = adapter_update_stats(tmp_path)
    assert stats["scale"] == 2.0 and stats["layers"] == 2 and stats["modules"] == ["q_proj", "up_proj"]
    assert len(stats["matrices"]) == 4
    m = stats["matrices"][0]
    dense = 2.0 * tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] \
        @ tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert m["norm"] == pytest.approx(float(torch.linalg.matrix_norm(dense)), rel=1e-6)
    assert m["singular_values"] == sorted(m["singular_values"], reverse=True)
