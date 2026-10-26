"""Measure what a training step costs: peak GPU memory and time per optimizer step.

A short real training loop (a few optimizer steps), used to see whether a configuration fits and
to estimate how long a full run takes. It is not a training run: the loss is recorded only to show
the steps ran.
"""

from __future__ import annotations

import gc
import math
import random
import statistics
import time

from .formatting import conversation_messages, tokenize_conversation

GB = 1024**3


def collate(tok, convs: list[dict], max_length: int) -> dict:
    """Tokenize conversations and pad them into one batch (right padding, labels `-100` on padding).

    Args:
        tok: A Qwen chat tokenizer.
        convs: The conversations of one micro-batch.
        max_length: Longer conversations are cut at the end.

    Returns:
        `input_ids`, `attention_mask` and `labels`, as tensors of shape (batch, longest).
    """
    import torch

    rows = [tokenize_conversation(tok, conversation_messages(c)) for c in convs]
    rows = [(ids[:max_length], labels[:max_length]) for ids, labels in rows]
    longest = max(len(ids) for ids, _ in rows)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([ids + [pad] * (longest - len(ids)) for ids, _ in rows]),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (longest - len(ids)) for ids, _ in rows]),
        "labels": torch.tensor([labels + [-100] * (longest - len(labels)) for _, labels in rows]),
    }


def measure(
    model_name: str, convs: list[dict], *, lora: dict | None, checkpointing: bool, lr: float,
    micro_batch: int, accumulation: int, warmup_steps: int, steps: int, max_length: int, seed: int,
    memory_cap_gb: float,
) -> dict:
    """Run a few optimizer steps and record memory and time.

    Each optimizer step accumulates `accumulation` micro-batches of `micro_batch` randomly drawn
    conversations, as in the planned training run. After `warmup_steps` unmeasured steps (the first
    allocates the optimizer state and tunes kernels), `steps` are timed. Peak memory is then
    measured once more on the worst case: one micro-batch of the longest conversations.

    Args:
        model_name: Hugging Face model id; loaded in bf16 on the GPU.
        convs: Training conversations to draw from.
        lora: LoRA settings (`r`, `alpha`, `dropout`, on every linear layer) or `None` for full
            fine-tuning.
        checkpointing: Whether to use gradient checkpointing.
        lr: AdamW learning rate.
        micro_batch: Conversations per forward/backward pass.
        accumulation: Micro-batches per optimizer step.
        warmup_steps: Unmeasured steps.
        steps: Timed steps.
        max_length: Sequence cut-off.
        seed: Seed of the conversation draw.
        memory_cap_gb: GPU memory this process may allocate. Where the GPU shares memory with the
            desktop (the GB10), a run that uses everything freezes the machine; with the cap, going
            over raises `OutOfMemoryError`, which is recorded as "does not fit".

    Returns:
        A dict with parameter counts, `weights_gb`, `typical_peak_gb`, `worst_peak_gb`, the per-step
        times, `median_step_s`, tokens per step, the losses and `memory_cap_gb`. A phase that went
        over the cap has `None` for its numbers (`typical_peak_gb` and the timings, or
        `worst_peak_gb`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    total_gb = torch.cuda.get_device_properties(0).total_memory / GB
    torch.cuda.set_per_process_memory_fraction(min(1.0, memory_cap_gb / total_gb))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM",
        ))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    weights_gb = torch.cuda.memory_allocated() / GB

    order = random.Random(seed).sample(convs, (warmup_steps + steps) * accumulation * micro_batch)
    groups = [order[i : i + micro_batch] for i in range(0, len(order), micro_batch)]

    def run_step(micro_batches):
        losses, tokens = [], 0
        for group in micro_batches:
            batch = {k: v.to("cuda") for k, v in collate(tok, group, max_length).items()}
            loss = model(**batch).loss
            (loss / len(micro_batches)).backward()
            losses.append(loss.item())
            tokens += int(batch["attention_mask"].sum())
            del batch, loss
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return statistics.fmean(losses), tokens

    def reset():
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times, losses, tokens = [], [], []
    typical_peak = None
    try:
        for s in range(warmup_steps):
            run_step(groups[s * accumulation : (s + 1) * accumulation])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for s in range(warmup_steps, warmup_steps + steps):
            t0 = time.perf_counter()
            loss, n = run_step(groups[s * accumulation : (s + 1) * accumulation])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            losses.append(loss)
            tokens.append(n)
        typical_peak = torch.cuda.max_memory_allocated() / GB
    except torch.OutOfMemoryError:
        times, losses, tokens = [], [], []
    reset()

    # Worst case: the longest conversations together, in one micro-batch.
    def length(c):
        text = tok.apply_chat_template(conversation_messages(c), tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    longest = sorted(convs, key=length, reverse=True)[:micro_batch]
    try:
        run_step([longest])
        worst_peak = torch.cuda.max_memory_allocated() / GB
    except torch.OutOfMemoryError:
        worst_peak = None

    return {
        "model": model_name, "lora": lora, "checkpointing": checkpointing,
        "total_params": total, "trainable_params": trainable, "weights_gb": weights_gb,
        "typical_peak_gb": typical_peak, "worst_peak_gb": worst_peak,
        "step_seconds": times, "median_step_s": statistics.median(times) if times else None,
        "tokens_per_step": statistics.fmean(tokens) if tokens else None, "losses": losses,
        "memory_cap_gb": memory_cap_gb,
    }


def estimate_hours(median_step_s: float, conversations: int, epochs: int, effective_batch: int) -> float:
    """Wall-clock hours of a full run, from the median step time (training steps only, no evaluation)."""
    return median_step_s * math.ceil(conversations * epochs / effective_batch) / 3600
