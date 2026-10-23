"""Turn conversations into readable text: markdown tables, the page context, pretty-printed turns."""

from __future__ import annotations

from .raw import conversation_type, turns


def table_to_markdown(table: list[list[str]]) -> str:
    """Render a table (header row first) as a markdown table."""
    width = max(len(r) for r in table)
    rows = [list(r) + [""] * (width - len(r)) for r in table]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def context_text(conv: dict) -> str:
    """The page the questions refer to: text before the table, the table, text after."""
    return "\n\n".join(
        [" ".join(conv["pre_text"]), table_to_markdown(conv["table"]), " ".join(conv["post_text"])]
    )


def format_conversation(conv: dict) -> str:
    """Pretty-print a conversation's turns as ``question -> program -> answer``."""
    lines = [f"{conv['id']}  (Type {conversation_type(conv)})"]
    for t in turns(conv):
        lines.append(f"  [{t.index + 1}] Q: {t.question}")
        lines.append(f"      program: {t.program}")
        lines.append(f"      answer:  {t.answer:g}" + (f"   (part {t.part + 1})" if conv["id"].startswith("Double_") else ""))
    return "\n".join(lines)
