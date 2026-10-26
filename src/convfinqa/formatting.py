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


def prompt_messages(conv: dict, k: int, examples: list[dict] = (), history: list[str] | None = None) -> list[dict]:
    """The chat that ends with turn `k`'s question, ready for the generation prompt.

    Earlier turns of `conv` are answered by `history` (the model's own programs) or, if it is
    `None`, by the gold programs. Each example conversation is shown first, in full, as if it
    had been answered earlier in the same chat: this is the few-shot setting.

    Args:
        conv: The conversation being asked about.
        k: Index of the turn to answer.
        examples: Conversations shown first, with their gold programs; none for zero-shot.
        history: Programs for turns `0..k-1`; the gold ones when omitted.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages += conversation_messages(example)[1:]
    own = conversation_messages(conv)[1:]
    if history is not None:
        assert len(history) >= k
        for i in range(k):
            own[2 * i + 1] = {"role": "assistant", "content": history[i]}
    return messages + own[: 2 * k + 1]


def tokenize_conversation(tok, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Token ids of a chat and labels that keep only the assistant's answers.

    The labels are `-100` everywhere except on each assistant message's content and its
    `<|im_end|>`; the newline the template writes after `<|im_end|>` is not trained. Each prefix is
    rendered by the chat template and tokenized separately, and its tokens must be the start of the
    whole chat's tokens (they are, because the template puts a special token at every boundary).

    Args:
        tok: A Qwen chat tokenizer.
        messages: The chat, starting with a system message.

    Returns:
        `(input_ids, labels)` of equal length.
    """
    def ids(msgs, **kwargs):
        text = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
        return tok(text, add_special_tokens=False)["input_ids"]

    input_ids = ids(messages)
    labels = [-100] * len(input_ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start = len(ids(messages[:i], add_generation_prompt=True))
        end = len(ids(messages[: i + 1])) - 1  # drop the newline after <|im_end|>
        assert ids(messages[: i + 1])[:start] == input_ids[:start]
        labels[start:end] = input_ids[start:end]
    return input_ids, labels
