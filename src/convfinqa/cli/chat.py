"""`chat`: ask the fine-tuned model questions about one page, one question at a time.

The model sees its own earlier programs as the history, as in the evaluation's predicted-history setting (the
headline), so what is shown here is what was measured. Greedy decoding, bf16, at most `MAX_NEW_TOKENS` new tokens.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from ..tuned_eval import MAX_NEW_TOKENS
from .info import AdapterError, check_adapter
from .page import PageError, load_page
from .session import Reply, Session, format_answer

DEFAULT_ADAPTER = Path("outputs") / "adapter"
QUIT = ("/quit", "/exit")


def show_reply(reply: Reply, write: Callable[[str], None]) -> None:
    """Print a reply: the program, and its answer or the reason it does not run."""
    write(f"  program: {reply.program}")
    if reply.error:
        write(f"  invalid: {reply.error}")
    else:
        write(f"  answer:  {format_answer(reply.answer)}")
    if reply.finish == "length":
        write(f"  (the output was cut at {MAX_NEW_TOKENS} tokens)")


def converse(
    session: Session, generate: Callable[[list[dict]], dict], lines: Iterable[str],
    write: Callable[[str], None] = print, echo: bool = False,
) -> None:
    """Answer one question per line until the lines or `/quit` end.

    Args:
        session: The conversation.
        generate: The model call (see `Session.ask`).
        lines: The questions; `/reset` starts a new conversation and `/quit` ends. Empty lines are skipped.
        write: Where the output goes.
        echo: Print each question before its reply (for questions that were not typed at a prompt).
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line in QUIT:
            return
        if line == "/reset":
            session.reset()
            write("(new conversation)")
            continue
        if echo:
            write(f"> {line}")
        show_reply(session.ask(generate, line), write)


def prompt_lines(prompt: str = "> ") -> Iterable[str]:
    """Lines typed at a prompt, with line editing, until end of input."""
    try:
        import readline  # noqa: F401  (its import gives `input` line editing)
    except ImportError:
        pass
    while True:
        try:
            yield input(prompt)
        except EOFError:
            print()
            return


def load_model_with_adapter(adapter: Path, base_model: str, merge: bool, memory_cap_gb: float | None):
    """The base model with the adapter attached (and merged into it, if asked), and the tokenizer."""
    from peft import PeftModel
    from transformers.utils import logging

    from ..evaluate import load_model

    logging.disable_progress_bar()  # the weight-loading bar is a hundred lines in a pipe
    model, tok = load_model(base_model, memory_cap_gb if memory_cap_gb is not None else float("inf"))
    model = PeftModel.from_pretrained(model, str(adapter))
    if merge:
        model = model.merge_and_unload()
    return model.eval(), tok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="chat", description="Ask the fine-tuned model questions about a page. Questions are read from the "
        "terminal, from a pipe, or from -q.",
    )
    parser.add_argument("--page", required=True, metavar="FILE.json", help="a JSON object with pre_text, table and "
                        "post_text (a dataset entry works; `print-example` writes one)")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER, help=f"adapter directory (default {DEFAULT_ADAPTER})")
    parser.add_argument("-q", "--question", action="append", metavar="TEXT", help="ask this question (repeatable, "
                        "asked in order as one conversation); then exit")
    parser.add_argument("--merge", action="store_true", help="merge the adapter into the weights (the output can "
                        "differ slightly in bf16)")
    parser.add_argument("--memory-cap-gb", type=float, default=None, help="limit on the GPU memory to use")
    args = parser.parse_args(argv)

    try:
        session = Session(load_page(args.page))
        checked = check_adapter(args.adapter)
    except (PageError, AdapterError) as e:
        print(f"chat: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    for warning in checked.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    import torch

    if not torch.cuda.is_available():
        print("chat: no CUDA GPU is available; the model runs on one", file=sys.stderr)
        raise SystemExit(2)
    from ..evaluate import make_generator

    print(f"Loading {checked.base_model} with {args.adapter} ...", file=sys.stderr)
    model, tok = load_model_with_adapter(args.adapter, checked.base_model, args.merge, args.memory_cap_gb)
    generate = make_generator(model, tok, MAX_NEW_TOKENS)

    if args.question:
        converse(session, generate, args.question, echo=True)
    elif sys.stdin.isatty():
        table = session.page["table"]
        print(f"Page: a table of {len(table)} rows. Ask a question; /reset starts over, /quit (or Ctrl-D) ends.")
        converse(session, generate, prompt_lines())
    else:
        converse(session, generate, sys.stdin, echo=True)
