"""Turn conversations into chat messages. (Training labels and eval prompts are added in 3.1.)"""

from __future__ import annotations

from .data import context_text, turns

SYSTEM_PROMPT = "Answer with a program in the ConvFinQA DSL."


def conversation_messages(conv: dict) -> list[dict]:
    """One multi-turn chat per conversation, with the gold program as every assistant turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in turns(conv):
        question = t.question if t.index else f"{context_text(conv)}\n\nQuestion: {t.question}"
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": t.program})
    return messages
