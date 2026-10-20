# fine-tuning-lab

A hands-on lab for learning LLM fine-tuning with Hugging Face (transformers, PEFT, TRL), using the
[ConvFinQA](https://github.com/czyssrs/ConvFinQA) conversational financial QA dataset.

The lab is a **Quarto book** in [`docs/`](docs/): it explains how LLMs and fine-tuning work, runs the
experiments, and records the results. Reusable code lives in [`src/convfinqa/`](src/convfinqa/).

See [`PLAN.md`](PLAN.md) for the full plan, progress tracker and decision log.

## Quick start

```bash
uv sync                                   # create .venv with all dependencies
uv run pytest                             # run the tests
quarto preview docs/                      # live-preview the book
quarto render docs/<chapter>.qmd          # (re-)execute one chapter, e.g. a frozen training chapter
```

Hardware used: NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64, CUDA 13.2).
