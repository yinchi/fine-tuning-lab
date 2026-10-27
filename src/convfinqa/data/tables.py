"""The turns of conversations as tables, for scoring and breakdowns."""

from __future__ import annotations

from .raw import conversation_type, turns


def turn_rows(convs: list[dict]) -> list[dict]:
    """One row per turn of the given conversations.

    Args:
        convs: Conversations.

    Returns:
        Rows with the conversation `id`, the turn index (`turn`, from 0), the number of turns of the
        conversation (`n_turns`), the `question`, the gold `program` and `answer`, `part` (0 or 1; 1 only in
        the second half of a Type II conversation), the conversation's `type` ("I" or "II") and
        `number_turn` (the program is a bare number).
    """
    rows = []
    for conv in convs:
        all_turns = turns(conv)
        for t in all_turns:
            rows.append({
                "id": conv["id"], "turn": t.index, "n_turns": len(all_turns), "question": t.question,
                "program": t.program, "answer": t.answer, "part": t.part,
                "type": conversation_type(conv), "number_turn": "(" not in t.program,
            })
    return rows
