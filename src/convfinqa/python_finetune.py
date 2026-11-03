"""Fine-tuning for Python as the output format, and generation with the adapter.

The training is `finetune.train_lora` unchanged. The only difference from the DSL adapter is the data: the
conversations are rewritten with Python as the answers (`python_eval.python_conversation`) and the system message
is the Python prompt, so the adapter is trained on the prompt it is evaluated with. This lives apart from
`finetune` so that the DSL adapter, and everything built from it, keeps its code hash.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from .formatting import conversation_messages, tokenize_conversation
from .python_eval import MAX_NEW_TOKENS, SYSTEM_PYTHON, python_conversation


def python_messages(conv: dict) -> list[dict]:
    """One multi-turn chat per conversation: the Python system message, the questions, and the Python answers."""
    return [{"role": "system", "content": SYSTEM_PYTHON}] + conversation_messages(python_conversation(conv))[1:]


def python_dataset(tok, convs: list[dict], max_length: int):
    """`train.make_dataset` with Python answers and the Python system message.

    Args:
        tok: A Qwen chat tokenizer.
        convs: The conversations, with their DSL gold programs (translated here).
        max_length: Rows are cut at this many tokens.

    Returns:
        A `datasets.Dataset` of `input_ids` and `labels` (the labels keep the answers only).
    """
    from datasets import Dataset

    rows = [tokenize_conversation(tok, python_messages(c)) for c in convs]
    return Dataset.from_dict({
        "input_ids": [ids[:max_length] for ids, _ in rows],
        "labels": [labels[:max_length] for _, labels in rows],
    })


def train_lora_python(cfg: dict, train_convs: list[dict], val_convs: list[dict], out: Path, seed: int) -> dict:
    """`finetune.train_lora` on the Python version of the conversations.

    Training and validation both use `python_dataset`: `train_lora` builds its datasets with the `make_dataset`
    of its module, which is replaced for the length of the call.

    Args:
        cfg: The settings of the run (`train_config.run_config`).
        train_convs: Training conversations (DSL gold programs).
        val_convs: Conversations the validation loss is measured on (DSL gold programs).
        out: Directory to write into (`adapter/`).
        seed: Seed of the run.

    Returns:
        What `finetune.train_lora` returns.
    """
    from unittest import mock

    from . import finetune

    with mock.patch.object(finetune, "make_dataset", python_dataset):
        return finetune.train_lora(cfg, train_convs, val_convs, out, seed)


def generate_tuned_python(
    cfg: dict, adapter: Path, convs: list[dict], mode: str, sink: Callable[[dict], None] | None = None
) -> list[dict]:
    """One greedy Python answer for every turn of `convs` from the base model with the Python adapter attached.

    The prompt is the training prompt (the Python system message, no examples), one prompt at a time, with the
    generation limit of the untuned Python runs.

    Args:
        cfg: The settings of the run (`model`, `memory_cap_gb`).
        adapter: The directory `train_lora_python` wrote the adapter to.
        convs: The conversations to answer, with DSL gold programs (translated here).
        mode: `gold` or `pred` (the history of the earlier turns; see `evaluate.run_generation`).
        sink: Called with each row as soon as it exists.

    Returns:
        One row per turn, as `evaluate.run_generation` returns them.
    """
    import torch
    from peft import PeftModel

    from .evaluate import load_model, make_generator, run_generation

    model, tok = load_model(cfg["model"], cfg["memory_cap_gb"])
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    generate = make_generator(model, tok, MAX_NEW_TOKENS)
    rows = run_generation(
        [python_conversation(c) for c in convs], generate, system=SYSTEM_PYTHON, examples=[], mode=mode, sink=sink
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows
