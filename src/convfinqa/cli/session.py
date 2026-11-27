"""One conversation about one page: the history, the prompt for the next question, and the answer to show.

The prompt is the one the model was trained on and evaluated with (`formatting`): the system message, the page and the
first question in the first user message, then each later question as a user message. The earlier answers are the
model's own, exactly as it wrote them, which is the predicted-history setting of the evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..data import context_text
from ..dsl.executor import try_execute
from ..formatting import SYSTEM_PROMPT
from .page import validate_page


@dataclass(frozen=True)
class Reply:
    """What the model said to one question."""

    question: str
    program: str
    """The model's output, exactly as generated."""
    answer: float | str | None
    """The program's value (`"yes"`/`"no"` for `greater`), or `None` if it does not run."""
    error: str | None
    """The kind of failure when the program does not run (`executor.ProgramError.kind`)."""
    finish: str | None = None
    """`stop`, or `length` if the output was cut at the generation limit."""


class Session:
    """A conversation about a page, which keeps the questions and the model's programs so far."""

    def __init__(self, page: dict):
        self.page = validate_page(page)
        self.history: list[tuple[str, str]] = []

    def reset(self) -> None:
        """Start a new conversation about the same page."""
        self.history = []

    def messages(self, question: str) -> list[dict]:
        """The chat that ends with `question`, ready for the generation prompt."""
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, (q, program) in enumerate(self.history):
            chat.append({"role": "user", "content": self._user(q, i)})
            chat.append({"role": "assistant", "content": program})
        chat.append({"role": "user", "content": self._user(question, len(self.history))})
        return chat

    def _user(self, question: str, index: int) -> str:
        return question if index else f"{context_text(self.page)}\n\nQuestion: {question}"

    def ask(self, generate: Callable[[list[dict]], dict], question: str) -> Reply:
        """Ask the model and remember its answer.

        Args:
            generate: From a chat to a dict with `output` (and `finish`), as `evaluate.make_generator` builds it.
            question: The question, as typed.

        Returns:
            The reply. An output that is not a valid program is a reply with `error` set, and stays in the history as
            written (as in the evaluation).
        """
        question = question.strip()
        result = generate(self.messages(question))
        program = result["output"]
        value, error = try_execute(program)
        self.history.append((question, program))
        return Reply(question, program, value, error, result.get("finish"))


def format_answer(value: float | str | None) -> str:
    """An answer for display: a whole number without a decimal point, other numbers as the executor rounded them."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
