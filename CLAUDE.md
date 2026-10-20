# fine-tuning-lab

A Quarto-book lab for learning LLM fine-tuning (Hugging Face, LoRA SFT) on ConvFinQA.

## Start of every session
1. Read `PLAN.md`: it's the source of truth for scope, decisions, and progress.
2. Find the first unchecked item in **Progress tracker** and continue from there, unless the user says otherwise.

## Keeping the plan current
- When an item is done **and verified**, tick it in **Progress tracker** with a short note (run id, headline number).
- When a decision is made or changed, add a dated entry to **Decision log**. Never rewrite old entries, and update the affected plan sections to match.
- Commit `PLAN.md` updates together with the change they describe.

## Conventions
- **Logic lives in `src/convfinqa/`** and is tested in `tests/`. Book chapters stay thin: narration, config, calls into the package, plots.
- **Stages pass data through files** in `artifacts/`, never through kernel state.
- **Heavy chapters** (baseline, training, evaluation, ablations) use `freeze: true`. `train()`/`evaluate()` skip work when `artifacts/<run_id>/DONE` exists unless `force=True`. Run long renders in `tmux`.
- **Splits:** tune on `val` (page-grouped 10% of train). Use `dev` only for final reported numbers.
- **Chapter files** are named `docs/PCC_first_words.qmd` (P = part, CC = chapter). Appendices are `9CC`. Titles carry the code, e.g. `"2.3 From base model to instruct model"`, and `number-sections` is off.
- **Teaching style:** demonstrate claims with code on the real model and data. Use "Under the hood" and "Further reading" callouts. Cite via `docs/references.bib` and link terms to the glossary.

## Commands
- `uv sync`: install dependencies. torch comes from the cu132 index.
- `uv run pytest`: run the tests.
- `quarto render docs/` renders the book. `quarto render docs/<file>.qmd` re-executes one chapter.
