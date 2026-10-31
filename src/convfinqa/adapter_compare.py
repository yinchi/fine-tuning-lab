"""The gold answers' log-probabilities under the base model and under the same model with an adapter."""

from __future__ import annotations

from pathlib import Path


def base_vs_adapter_logprobs(model_name: str, adapter: Path, convs: list[dict]) -> list[dict]:
    """Per-token log-probabilities of the gold answer tokens, before and after adapting.

    Both passes use the training format (`formatting.conversation_messages`, gold history) on the same bf16 base
    weights; the adapter is attached to the base model that was just scored.

    Args:
        model_name: The base model.
        adapter: Directory of a saved LoRA adapter.
        convs: The conversations to score.

    Returns:
        One dict per conversation: `id`, `tokens` (the trained token strings), `base` and `adapter`
        (log-probabilities, natural log).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .formatting import conversation_messages
    from .probing import gold_logprobs

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    chats = [conversation_messages(c) for c in convs]
    base = [gold_logprobs(model, tok, m) for m in chats]
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    tuned = [gold_logprobs(model, tok, m) for m in chats]
    rows = []
    for c, (ids, lb), (ids2, la) in zip(convs, base, tuned):
        assert ids == ids2
        rows.append({"id": c["id"], "tokens": [tok.decode([i]) for i in ids], "base": lb, "adapter": la})
    return rows
