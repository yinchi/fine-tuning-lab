"""Checks on LoRA adapters made with PEFT: that they start as the base model, and that merging is faithful."""

from __future__ import annotations

import gc

GB = 1024**3


def randomize_b(peft_model, std: float, seed: int) -> None:
    """Give every adapter's `B` matrix small random values, so the adapter changes the output.

    The values depend only on `seed`, `std` and the order and shapes of the matrices, so the same
    adapter can be built on the same model loaded in another precision.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data = (std * torch.randn(p.shape, generator=generator)).to(p.device, p.dtype)


def merge_comparison(model, ids, *, r: int, alpha: int, b_std: float, seed: int) -> dict:
    """Logits of a model with a random adapter kept separate and merged into the weights.

    Args:
        model: A causal language model (it is wrapped in place).
        ids: A `(1, T)` tensor of token ids on the model's device.
        r: LoRA rank.
        alpha: LoRA alpha.
        b_std: Standard deviation of the random `B` values.
        seed: Seed of those values.

    Returns:
        `step0_max_abs_diff` (adapter with `B = 0` against the original), `adapter_effect_max_abs`
        (how far the random adapter moves the logits), `merge_max_abs_diff` (merged against
        separate), `merge_top1_agreement`, `adapter_top1_agreement`, `trainable_params`,
        `adapted_modules` and `logit_scale` (the largest logit of the original model).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer

    def logits(m):
        with torch.no_grad():
            return m(input_ids=ids).logits[0].float()

    base = logits(model)
    peft = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM",
    )).eval()
    out = {
        "step0_max_abs_diff": (logits(peft) - base).abs().max().item(),
        "trainable_params": sum(p.numel() for p in peft.parameters() if p.requires_grad),
        "adapted_modules": sum(isinstance(m, LoraLayer) for m in peft.modules()),
        "logit_scale": base.abs().max().item(),
    }
    randomize_b(peft, b_std, seed)
    separate = logits(peft)
    merged = logits(peft.merge_and_unload())
    out.update({
        "adapter_effect_max_abs": (separate - base).abs().max().item(),
        "merge_max_abs_diff": (merged - separate).abs().max().item(),
        "merge_top1_agreement": (merged.argmax(-1) == separate.argmax(-1)).float().mean().item(),
        "adapter_top1_agreement": (separate.argmax(-1) == base.argmax(-1)).float().mean().item(),
    })
    return out


def lora_checks(
    model_name: str, messages: list[dict], *, r: int, alpha: int, b_std: float, seed: int, memory_cap_gb: float,
) -> dict:
    """Run `merge_comparison` on the real model in bf16 and in fp32, on one prompt.

    Args:
        model_name: Hugging Face model id.
        messages: A chat ending with a user message; the logits cover its generation prompt and every
            position before it.
        r: LoRA rank.
        alpha: LoRA alpha.
        b_std: Standard deviation of the random `B` values.
        seed: Seed of those values.
        memory_cap_gb: GPU memory the process may allocate.

    Returns:
        `base_params`, `n_positions`, and the `merge_comparison` results under `bf16` and `fp32`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(
        min(1.0, memory_cap_gb / (torch.cuda.get_device_properties(0).total_memory / GB))
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    ids = tok(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        return_tensors="pt", add_special_tokens=False,
    ).input_ids.to("cuda")
    out = {"n_positions": int(ids.shape[1])}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda").eval()
        out["base_params"] = sum(p.numel() for p in model.parameters())
        out[name] = merge_comparison(model, ids, r=r, alpha=alpha, b_std=b_std, seed=seed)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out
