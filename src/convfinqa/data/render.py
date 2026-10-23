"""Methods for turning conversations into readable text.

Includes methods for rendering markdown tables, the page context, and pretty-printed turns.
"""

from __future__ import annotations

from .raw import conversation_type, turns


def table_to_markdown(table: list[list[str]]) -> str:
    """Render a table (header row first) as a markdown table.
    
    Args:
        table: A list of lists of strings representing the table, with the first row as the header.

    Returns:
        A string containing the markdown representation of the table.
    """
    width = max(len(r) for r in table)
    rows = [list(r) + [""] * (width - len(r)) for r in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def context_text(conv: dict) -> str:
    """Render part of the page the questions refer to.

    Includes the text before the table, the table itself, and the text after the table.

    Args:
        conv: A dictionary representing the conversation. The `table` key is `list[list[str]]`,
            thus suitable for passing to `table_to_markdown`, while the `pre_text` and `post_text`
            keys are lists of strings representing the text before and after the table,
            respectively.

    Returns:
        A string containing the rendered context text.
    """
    return "\n\n".join(
        [" ".join(conv["pre_text"]), table_to_markdown(conv["table"]), " ".join(conv["post_text"])]
    )


def format_conversation(conv: dict) -> str:
    """Pretty-print a conversation's turns as `question -> program -> answer`.
    
    Each part of the conversation is rendered with a header ("<n> Q:", "program:", "answer:"),
    separated by newlines.  An overall header containing the conversation ID and type is included
    at the top.

    Args:
        conv: A dictionary representing the conversation.

    Returns:
        A string containing the pretty-printed conversation.
    """
    lines = [f"{conv['id']}  (Type {conversation_type(conv)})"]
    for t in turns(conv):
        lines.append(f"  [{t.index + 1}] Q: {t.question}")
        lines.append(f"      program: {t.program}")
        lines.append(
            f"      answer:  {t.answer:g}"
            + (f"   (part {t.part + 1})" if conv["id"].startswith("Double_") else "")
        )
    return "\n".join(lines)
