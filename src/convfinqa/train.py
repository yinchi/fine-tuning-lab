"""Supervised fine-tuning with TRL's `SFTTrainer`, on conversations tokenized by `formatting`."""

from __future__ import annotations

import gc
import tempfile

from .data import turns
from .formatting import conversation_messages, prompt_messages, tokenize_conversation
from .probing import classify_output, generate_greedy, loss_terms

GB = 1024**3


def make_dataset(tok, convs: list[dict], max_length: int):
    """A pre-tokenized dataset of `input_ids` and `labels`, one row per conversation.

    The labels are `formatting.tokenize_conversation`'s: the token id on the assistant's answers and
    `-100` everywhere else. `SFTTrainer` uses a `labels` column as it is, and cuts rows at
    `max_length`.

    Args:
        tok: A Qwen chat tokenizer.
        convs: The conversations.
        max_length: Rows are cut at this many tokens.

    Returns:
        A `datasets.Dataset`.
    """
    from datasets import Dataset

    rows = [tokenize_conversation(tok, conversation_messages(c)) for c in convs]
    return Dataset.from_dict({
        "input_ids": [ids[:max_length] for ids, _ in rows],
        "labels": [labels[:max_length] for _, labels in rows],
    })


def _answers(model, tok, convs, n_turns):
    """Greedy zero-shot answers to the first `n_turns` turns of `convs`, with gold history."""
    items = [(c, k) for c in convs for k in range(len(c["annotation"]["dialogue_break"]))][:n_turns]
    outs = generate_greedy(model, tok, [prompt_messages(c, k) for c, k in items])
    rows = []
    for (c, k), out in zip(items, outs):
        t = turns(c)[k]
        rows.append({
            "id": c["id"], "turn": k, "output": out, "gold": t.program,
            "kind": classify_output(out, t.program, t.answer),
        })
    return rows


def run_sft_demo(
    model_name: str, train_convs: list[dict], eval_convs: list[dict], *, epochs: int, micro_batch: int,
    accumulation: int, lr: float, warmup_steps: float, max_length: int, eval_every: int,
    prompt_turns: int, seed: int, memory_cap_gb: float,
) -> dict:
    """A short full fine-tuning run through `SFTTrainer`, with the facts the chapter checks.

    The weights are fp32 and the arithmetic bf16 (`bf16=True`), the usual Trainer set-up. The
    process is capped at `memory_cap_gb` of GPU memory.

    Args:
        model_name: Hugging Face model id.
        train_convs: Training conversations.
        eval_convs: Conversations for the loss at step 0 and every `eval_every` steps; the first few
            are also answered before and after training.
        epochs: Passes over `train_convs`.
        micro_batch: Conversations per forward/backward pass.
        accumulation: Micro-batches per optimizer step.
        lr: Peak learning rate (cosine schedule).
        warmup_steps: Warmup; a float below 1 is a fraction of all steps.
        max_length: Sequence cut-off.
        eval_every: Steps between evaluations.
        prompt_turns: How many turns to answer before and after.
        seed: Seed of the run.
        memory_cap_gb: GPU memory the process may allocate.

    Returns:
        `log_history` (the Trainer's), `max_steps`, `step0_by_hand` (mean over conversations of the
        loss computed by hand, before training), `labels_match` (the Trainer's dataset has our
        labels), `trained_text` of the first conversation, `before` and `after` answers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.cuda.set_per_process_memory_fraction(
        min(1.0, memory_cap_gb / (torch.cuda.get_device_properties(0).total_memory / GB))
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map="cuda")
    train, evaluation = make_dataset(tok, train_convs, max_length), make_dataset(tok, eval_convs, max_length)

    model.eval()
    before = _answers(model, tok, eval_convs, prompt_turns)
    per_conversation = []
    for row in evaluation:
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            per_conversation.append(loss_terms(model, ids, labels)["by_hand"])

    with tempfile.TemporaryDirectory() as scratch:
        args = SFTConfig(
            output_dir=scratch, per_device_train_batch_size=micro_batch, gradient_accumulation_steps=accumulation,
            per_device_eval_batch_size=1, num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine",
            warmup_steps=warmup_steps, weight_decay=0.0, max_grad_norm=1.0, bf16=True, max_length=max_length,
            eval_strategy="steps", eval_steps=eval_every, eval_on_start=True, logging_steps=1,
            save_strategy="no", report_to="none", disable_tqdm=True, seed=seed, data_seed=seed,
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=evaluation, processing_class=tok)
        labels_match = all(
            list(a["labels"]) == list(b["labels"]) and list(a["input_ids"]) == list(b["input_ids"])
            for a, b in zip(trainer.train_dataset, train)
        ) and len(trainer.train_dataset) == len(train)
        first = trainer.train_dataset[0]
        trained_text = tok.decode([i for i, l in zip(first["input_ids"], first["labels"]) if l != -100])
        trainer.train()
        log_history, max_steps = trainer.state.log_history, trainer.state.max_steps
        model.eval()
        after = _answers(model, tok, eval_convs, prompt_turns)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "log_history": log_history, "max_steps": max_steps,
        "step0_by_hand": sum(per_conversation) / len(per_conversation),
        "labels_match": labels_match, "trained_text": trained_text, "before": before, "after": after,
    }
