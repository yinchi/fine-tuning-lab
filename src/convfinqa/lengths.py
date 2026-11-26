"""Token lengths of whole conversations, and the `max_length` that follows from them."""

from __future__ import annotations

from .formatting import conversation_messages

ROUND_TO = 256
"""`max_length` is the longest conversation rounded up to a multiple of this."""


def n_tokens(tok, conv: dict) -> int:
    """The tokens of one conversation as training sees it: the chat template applied, then tokenized.

    Args:
        tok: A Hugging Face tokenizer with a chat template.
        conv: A ConvFinQA conversation.

    Returns:
        The length of the token sequence (the template already wrote the special tokens, so none are added).
    """
    text = tok.apply_chat_template(conversation_messages(conv), tokenize=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def max_length_for(longest: int, round_to: int = ROUND_TO) -> int:
    """The smallest multiple of `round_to` that holds `longest` tokens, so that nothing is truncated."""
    return -(-longest // round_to) * round_to


def conversation_lengths(tok, splits: dict[str, list[dict]]) -> list[dict]:
    """One row per conversation: its split, id and token length.

    Args:
        tok: A Hugging Face tokenizer with a chat template.
        splits: Conversations by split name.

    Returns:
        Rows `{"split", "id", "tokens"}`, in the order of `splits`.
    """
    return [
        {"split": name, "id": conv["id"], "tokens": n_tokens(tok, conv)}
        for name, convs in splits.items() for conv in convs
    ]
