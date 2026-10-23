"""Turn conversations into chat messages.

(Training labels and eval prompts are added in 3.1.)
"""

from __future__ import annotations

from .data import context_text, turns

SYSTEM_PROMPT = "Answer with a program in the ConvFinQA DSL."


def conversation_messages(conv: dict) -> list[dict]:
    """One multi-turn chat per conversation, with the gold program as every assistant turn."""

    # Start with the system prompt as the first message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in turns(conv):
        # Attach the context to the first question turn (index 0)
        question = t.question if t.index else f"{context_text(conv)}\n\nQuestion: {t.question}"

        # Append the question and the assistant's response
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": t.program})
    return messages
