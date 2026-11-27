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
quarto preview docs/ --no-execute-daemon   # live-preview the book
./build.sh                                # build stale/missing artifacts (heavy ones too), render, then preview
uv run python -m convfinqa.artifacts status   # what is fresh, stale or missing
quarto render docs/                       # render the whole book (stale heavy artifacts raise an error)
quarto render docs/<chapter>.qmd --no-execute-daemon   # (re-)execute one chapter
```

## Using the result without the book

Three commands, run from this directory (see the last chapter of the book for details):

```bash
uv run train-adapter                          # build the adapter from scratch (about 4 hours on the GB10); writes outputs/adapter/
uv run print-example dev:3 > page.json        # print one conversation of the dataset as a page (--show-gold adds the answers)
uv run chat --page page.json                  # ask the fine-tuned model questions about the page, one per line
uv run chat --page page.json -q "what was ...?" -q "and in 2007?"   # or give the questions on the command line
```

`chat` also works with the book's own adapter: `--adapter artifacts/lora_main_0/adapter`.

Hardware used: NVIDIA DGX Spark (GB10, 128 GB unified memory, aarch64, CUDA 13.2).
