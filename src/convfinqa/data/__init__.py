"""The ConvFinQA dataset: loading (``raw``) and rendering as text (``render``).

The names chapters use most are re-exported here, so ``data.load_split`` and ``data.turns`` work.
Imports ``dsl`` at most; never torch.
"""

from .raw import RAW_DIR, ROOT, Turn, conversation_type, download_data, load_split, turns
from .render import context_text, format_conversation, table_to_markdown

__all__ = [
    "RAW_DIR", "ROOT", "Turn", "conversation_type", "download_data", "load_split", "turns",
    "context_text", "format_conversation", "table_to_markdown",
]
