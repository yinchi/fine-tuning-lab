"""`print-example`: print one conversation of the dataset as JSON, the page `chat` takes."""

from __future__ import annotations

import argparse
import json
import sys

from ..data import turns
from .page import PAGE_KEYS

SPLITS = ("train", "val", "dev")


def parse_spec(spec: str) -> tuple[str, int]:
    """`dev:3` as `("dev", 3)`.

    Raises:
        ValueError: If it is not a split of `SPLITS`, a colon and a number from 0.
    """
    split, _, number = spec.partition(":")
    if split not in SPLITS or not number.isdigit():
        raise ValueError(f"{spec!r} is not SPLIT:N with SPLIT one of {', '.join(SPLITS)} and N a number from 0")
    return split, int(number)


def load_conversations(split: str) -> list[dict]:
    """The conversations of `split`: `train` is the training part of the book's split, `val` the held-out part."""
    from ..data import download_data, load_split, make_val_split
    from ..evaluate import require_freeze

    require_freeze(split)
    download_data()
    if split == "dev":
        return load_split("dev")
    fit, val = make_val_split(load_split("train"), fraction=0.1, seed=0)
    return fit if split == "train" else val


def example_object(conv: dict, show_gold: bool = False) -> dict:
    """The page of a conversation, with its `id`, and with the gold questions and answers if asked."""
    obj = {"id": conv["id"], **{key: conv[key] for key in PAGE_KEYS}}
    if show_gold:
        obj["gold"] = [{"question": t.question, "program": t.program, "answer": t.answer} for t in turns(conv)]
    return obj


def to_json(obj: dict) -> str:
    """JSON with one table row, and one gold turn, per line."""
    lines = []
    for key, value in obj.items():
        if isinstance(value, list) and value and all(isinstance(v, (list, dict)) for v in value):
            body = ",\n".join("    " + json.dumps(v, ensure_ascii=False) for v in value)
            lines.append(f'  {json.dumps(key)}: [\n{body}\n  ]')
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(lines) + "\n}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="print-example", description="Print one conversation of the dataset as JSON, to give to `chat --page`.",
    )
    parser.add_argument("example", nargs="?", metavar="SPLIT:N", help="the N-th conversation (from 0) of train, val or dev")
    parser.add_argument("--show-gold", action="store_true", help="add the gold questions, programs and answers")
    parser.add_argument("--list", metavar="SPLIT", choices=SPLITS, help="list the ids of a split instead")
    args = parser.parse_args(argv)
    if (args.example is None) == (args.list is None):
        parser.error("give SPLIT:N, or --list SPLIT")
    try:
        if args.list:
            convs = load_conversations(args.list)
            print(f"{args.list}: {len(convs)} conversations")
            for i, conv in enumerate(convs):
                print(f"{i}\t{conv['id']}")
            return
        split, n = parse_spec(args.example)
        convs = load_conversations(split)
        if n >= len(convs):
            raise ValueError(f"{split} has {len(convs)} conversations, so N goes from 0 to {len(convs) - 1}")
        print(to_json(example_object(convs[n], args.show_gold)))
    except (ValueError, RuntimeError) as e:
        print(f"print-example: {e}", file=sys.stderr)
        raise SystemExit(2) from None
