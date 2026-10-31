"""What a trained LoRA adapter changed: the size and the rank of its update to each weight matrix."""

from __future__ import annotations

import re
from pathlib import Path

KEY = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight")


def update_singular_values(a, b, scale: float):
    """Singular values of the update `scale * B @ A`, largest first.

    The update has the shape of the weight matrix but rank at most `r`, so its singular values come from a small
    `r × r` matrix: with `B = Qb Rb` and `Aᵀ = Qa Ra` (QR factorisations), `B A = Qb (Rb Raᵀ) Qaᵀ`, and the two
    `Q`s have orthonormal columns.

    Args:
        a: The LoRA `A` matrix, `r × in_features`.
        b: The LoRA `B` matrix, `out_features × r`.
        scale: `alpha / r`.

    Returns:
        A tensor of `r` singular values.
    """
    import torch

    rb = torch.linalg.qr(b.double()).R
    ra = torch.linalg.qr(a.double().T).R
    return scale * torch.linalg.svdvals(rb @ ra.T)


def adapter_update_stats(adapter_dir: Path) -> dict:
    """The update of every adapted matrix of a saved adapter.

    Args:
        adapter_dir: A directory with `adapter_config.json` and `adapter_model.safetensors`.

    Returns:
        `scale` (`alpha / r`), `r`, `modules` (the module names in order), `layers` (their count), and `matrices`:
        a list with, per adapted matrix, its `layer`, `module`, Frobenius `norm` and the `singular_values`
        of its update.
    """
    import json

    from safetensors import safe_open

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    factors: dict[tuple[int, str], dict] = {}
    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():
            m = KEY.fullmatch(key)
            if m is None:
                raise ValueError(f"unexpected adapter weight name: {key}")
            factors.setdefault((int(m[1]), m[2]), {})[m[3]] = f.get_tensor(key)
    matrices = []
    for (layer, module), ab in sorted(factors.items()):
        s = update_singular_values(ab["A"], ab["B"], scale)
        matrices.append({
            "layer": layer, "module": module,
            "norm": float((s**2).sum().sqrt()), "singular_values": [float(x) for x in s],
        })
    modules = sorted({m["module"] for m in matrices})
    return {"scale": scale, "r": cfg["r"], "modules": modules,
            "layers": 1 + max(m["layer"] for m in matrices), "matrices": matrices}


def base_weight_norms(model_name: str) -> dict[tuple[int, str], float]:
    """Frobenius norm of every adapted weight matrix of the base model, read from the local Hugging Face cache.

    Args:
        model_name: The base model, already downloaded.

    Returns:
        The norm by `(layer, module)`.
    """
    import json

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    root = Path(snapshot_download(model_name, local_files_only=True, allow_patterns=["*.json", "*.safetensors"]))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    pattern = re.compile(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight")
    norms = {}
    for name, shard in weight_map.items():
        m = pattern.fullmatch(name)
        if m is None:
            continue
        with safe_open(str(root / shard), "pt") as f:
            norms[(int(m[1]), m[2])] = float(torch.linalg.matrix_norm(f.get_tensor(name).double()))
    return norms
