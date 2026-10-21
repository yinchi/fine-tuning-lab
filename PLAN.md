# Plan: A Quarto-driven lab for LLM fine-tuning on ConvFinQA (Hugging Face, LoRA SFT)

## Context
The goal is to learn the Hugging Face fine-tuning workflow end to end, using ConvFinQA as the task: data inspection, formatting, LoRA training, evaluation, and authoring new examples in the same format. `fine-tuning-lab/` is empty.

**The lab is a Quarto book, written as teaching material.** It does two jobs:
1. It **explains** what LLMs and fine-tuning actually do, hands-on with the real models and ConvFinQA examples. This is Part 2, Foundations.
2. It **runs and documents** the experiments, so the rendered book is also the lab notebook and the record of results. This is Part 3.

Expensive computations, such as training and full evaluation, cache their results in `artifacts/`, so re-rendering a chapter loads them instead of recomputing. Quarto's freeze cache keeps unchanged chapters from re-running at all.

Decisions already made:
- **Output:** the model emits the ConvFinQA **DSL program**, and a Python executor computes the number. §3.5 runs the same pipeline with **Python output** to test whether models already trained on lots of code do better in a language they know.
- **Method:** **LoRA SFT with TRL**, in bf16.
- **Environment:** **uv + Python 3.14**. torch `2.14.1+cu132` and the rest of the stack have cp314 aarch64 wheels.
  - `nvidia-smi` reports "CUDA 13.2": the newest CUDA runtime driver 595 supports.
  - The system `nvcc` is also 13.2. Using `cu132` keeps the wheel's bundled runtime the same version as the system compiler, in case a CUDA extension is ever compiled locally.
  - `cu130` would also run, since drivers are compatible across 13.x minor versions.
- **Documentation:** **Quarto 1.10.19** (linux-arm64). pandoc and jupyter are already on the system.
- **Splits:**
  - **train-fit** is about 90% of `train.json`.
  - **val** is about 10% of `train.json`, **grouped by source report page** so no page is in both. It's used for checkpoint selection, prompt and few-shot choices, and any tuning.
  - **dev** is used **only for final reported numbers**.
- **Version control:** git, plus a **private GitHub repo** `yinchi/fine-tuning-lab`, created with `gh`.
  - Code, `docs/`, and `docs/_freeze/` are committed. `data/`, `artifacts/`, and `docs/_book/` are not.
- **Tracking:** TensorBoard, logging to `artifacts/runs/<id>/tb/`. The book reads the same logs and `trainer_state.json`.
- **Noise:** one training seed plus a paired cluster bootstrap on every comparison. Extra seeds (or few-shot example sets) are added only when a comparison is inconclusive (§3.4).

Hardware: DGX Spark (NVIDIA GB10, ~121 GB unified memory, aarch64, CUDA 13.2).

## Architecture: how Quarto drives the lab
1. **Each chapter is a pipeline stage.** It runs in its own Jupyter kernel, the uv venv registered as `fine-tuning-lab`.
2. **Stages pass data through files, never kernel state.** Each chapter writes artifacts to `artifacts/` (processed data, adapters, predictions, metrics), and later chapters read them. Freeze caches rendered output only, not Python objects.
3. **Logic lives in `src/convfinqa/`.** Chapters import it, explain it, and call it with visible config. pytest covers the package. Chapters hold narration, config, calls, and plots, never large logic.
4. **Freeze is set per chapter:**
   - `_quarto.yml` sets `execute: freeze: auto` for **every** chapter: a chapter re-runs, in full, only when its own `.qmd` changes. Editing a chapter therefore always shows your changes. Freeze watches only the `.qmd`, not `src/convfinqa/`, `_environment`, or the data; to force a re-run, delete `docs/_freeze/<chapter>/`.
   - We don't use `freeze: true`. Tested: it hides prose edits from a whole-book render, and rendering the chapter directly re-executes every cell. Accidental re-runs are prevented by the artifact system in item 5 instead.
   - `docs/_freeze/` is committed, so the book renders anywhere without a GPU.
5. **Artifacts are named, tracked, and built on demand** (`src/convfinqa/artifacts.py`). An artifact is a named result on disk, such as a processed split, a trained adapter, or an evaluation.
   - **Registration.** Builders are functions registered by name template, e.g. `@artifact("sft-{variant}-{seed}", deps=["tokenized-{variant}", "split-v1"], heavy=True)`. Dependencies are plain name strings. The whole graph is validated when the registry loads (every dep exists, no cycles), and a test checks it.
   - **Builders.** A builder receives `out` (a temp dir), one handle per dependency (path, output hash, `.json()` helpers), and keyword parameters. It uses **only** those: no chapter globals, no shared state. It loads its own model and needs no teardown, because heavy builders run in their own process.
   - **Node files.** Each artifact has `artifacts/<name>/node.json`: its config, the output hashes of its dependencies, content hashes of the outputs it declares significant (logs and TensorBoard files excluded), and a `version` field to bump when a code change should invalidate it. It is written last and atomically, after building in a temp dir.
   - **Freshness is recursive.** An artifact is fresh only if its config matches, **every dependency is itself fresh**, and the dependencies' current output hashes equal the recorded ones. Content hashes give early cutoff: a deterministic rebuild with identical output leaves dependents fresh.
   - **Light vs. heavy.** Light artifacts rebuild in the notebook kernel whenever stale. A stale or missing **heavy** artifact raises an error with the build command, unless `RUN_HEAVY=1` is set; then the decorator launches `python -m convfinqa.artifacts build <name> --params '<json>'` as a subprocess, logging to `artifacts/<name>/build.log`. The subprocess frees all memory on exit and isolates crashes. A lock file prevents two concurrent builds of one artifact.
   - **Across chapters.** Artifacts are global by name, so any chapter can depend on one built elsewhere. Each chapter records the output hashes of the artifacts it used in `artifacts/_chapters/<chapter>.json` (the chapter name comes from `QUARTO_DOCUMENT_FILE`). `python -m convfinqa.artifacts refresh` deletes `docs/_freeze/<chapter>/` for chapters whose artifacts changed or are stale, since Quarto's freeze only watches the `.qmd`. It runs automatically before every render and preview through `pre-render` in `_quarto.yml`, via a small stdlib-only shim script that starts the project venv's Python (a `pre-render` script otherwise runs under the system `python3`). Other subcommands: `build <name>` and `status` (a fresh/stale/missing tree with reasons).
   - **Chapter conventions.** Cells that call heavy artifacts carry a `#HEAVY` comment. **The recommended build is a full-book render:** `RUN_HEAVY=1 quarto render docs/ --log artifacts/render.log` inside `tmux` or `screen`. The freshness check rebuilds only the heavy artifacts that are missing or stale. `build <name>` builds one artifact without Quarto.
6. **Long runs:**
   - Run the full-book render, `quarto render docs/`, in `tmux` or `screen`, with `RUN_HEAVY=1` reaching it (see 1.1 for the exact commands; `tmux` needs `-e RUN_HEAVY=1`).
   - Disable tqdm in rendered output, log to `artifacts/<name>/build.log` and TensorBoard, and show only a summary and plots in the chapter.
   - Each artifact directory stores the resolved config, git commit, library versions, and metrics, so every chapter's numbers trace back to a run.
7. **Teaching conventions** used throughout:
   - every claim is demonstrated with code on the real model or data;
   - `::: {.callout-note title="Under the hood"}` boxes link results back to Part 2;
   - `::: {.callout-tip title="Further reading"}` boxes end chapters;
   - citations come from `references.bib` via `[@key]`;
   - new terms link to the glossary;
   - each chapter opens with a short "you will learn" list.
8. **Numbering:**
   - Files are named `PCC_first_words.qmd`: P = part, CC = chapter within the part, then the first words of the title. Appendices use `9CC`.
   - `_quarto.yml` sets `number-sections: true`, so Quarto numbers chapters itself: straight through the book, not per part, with appendices (listed under `appendices:`) lettered A, B, and so on. Titles carry no code.
   - The `P.C` codes (written §2.3 in this plan) are therefore **plan and filename identifiers only**. They don't match the rendered chapter numbers. Cross-references in the book use `@sec-` labels.

## What ConvFinQA looks like (content of §1.2)
Source: github.com/czyssrs/ConvFinQA (`data.zip`), from Chen et al., EMNLP 2022. Conversations are built from FinQA: earnings-report pages, each with a table and the text around it.

**Conversation level** (`train.json` 3,037, `dev.json` 421, `test.json` 434; test labels may be private, so **dev is our test set**):
```jsonc
{
  "id": "Single_JKHY/2009/page_28.pdf-3",
  "pre_text":  ["sentence", "..."],            // text before the table
  "table":     [["", "2009", "2008"], ["net cash from operating activities", "206588", "181001"], ...],
  "post_text": ["sentence", "..."],            // text after the table
  "annotation": {
    "dialogue_break": ["what was the net cash from operating activities in 2009?",
                       "what about in 2008?",
                       "what is the difference?",
                       "what percentage change does this represent?"],
    "turn_program":   ["206588", "181001", "subtract(206588, 181001)",
                       "subtract(206588, 181001), divide(#0, 181001)"],
    "exe_ans_list":   [206588, 181001, 25587, 0.14136],
    "qa_split":       [0, 0, 0, 0],            // Type II conversations merge 2 FinQA questions: 0s then 1s
    "original_program": "..."                  // Type II has original_program_0/_1 instead
  },
  "qa": {...}                                  // Type II has qa_0 / qa_1 instead
}
```
(Illustrative example from the paper. §1.2 renders real ones and confirms the exact keys.)

**Turn level** (`train_turn.json` 11,104, `dev_turn.json` 1,490): one row per turn. Each row has the fields above plus `cur_dial` (questions up to this turn), `cur_program`, `exe_ans`, `cur_type` (number selection or program), `turn_ind`, and `gold_ind` (supporting facts).

**DSL:**
- Operations: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`.
- Arguments: numbers, `#k` (the result of step k), or `const_100`-style constants.
- Steps are separated by `, `.
- A turn that just selects a number has a bare number as its program.

## Expected performance (reference points for §3.2 and §3.4)
Metric: execution accuracy on the ConvFinQA **dev** split. Papers differ in split, tolerance, and context setup, so treat these as ballpark figures.

| Setting | Exec. acc. | Source |
|---|---|---|
| General crowd workers | 46.9 | ConvFinQA paper, Tab. 3 |
| GPT-3 (text-davinci-002), few-shot, original DSL / normal DSL / CoT | 40.8 / 45.2 / 40.6 | ConvFinQA paper, Tab. 5 |
| FinQANet (RoBERTa-large), retriever + generator, fine-tuned | 68.9 | ConvFinQA paper, Tab. 3 |
| FinQANet with gold supporting facts | 77.3 | ConvFinQA paper, Tab. 3 |
| GPT-4, zero-shot | 76.5 | Xie et al. 2023; also cited in FinBen |
| PoT-GPT-4 / ENCORE / FINDER (2025 state of the art) | 74.8 / 76.0 / 82.0 | FINDER, arXiv 2510.13157, Tab. 2 |
| Qwen2.5-7B-Instruct → Fin-R1 (SFT + RL, reasoning traces) | 66 → 85 (SFT only: 81) | Fin-R1, arXiv 2503.16252 (own eval protocol) |
| Qwen3.5-4B + LoRA + calculator tools (community) | 82.4 (+12.8 over base) | featherless.ai listing (unverified) |
| Human experts | 89.4 | ConvFinQA paper, Tab. 3 |

What our setup changes:
- We give the model the full page instead of retrieved facts. That's closer to the "gold facts" setting, with more noise.
- How we show the earlier turns. No conversations or turns are ever filtered out; this only changes what the prompt shows for turns before *k*.
  - FinQANet in the paper sees only the earlier **questions**.
  - We show earlier questions **and the assistant's earlier answers**, formatted as a real multi-turn chat.
  - **Gold history:** the dataset's correct programs fill in the earlier answers. This is teacher forcing. It hands the model oracle information, so it's an **upper bound**, not comparable to the paper.
  - **Predicted history:** the model's own earlier outputs fill in the earlier answers, and the conversation runs turn by turn, so errors compound. This adds no oracle information, so it's the **headline number** we compare against the table.
  - **Training always uses gold history.**

Our targets:
- **§3.2, untuned Qwen3-4B-Instruct-2507, few-shot, DSL output:** about **50–65%**. Expect DSL format errors. The paper found GPT-3 was hurt by the unfamiliar DSL syntax. Zero-shot will be lower.
- **§3.4, after LoRA SFT on about 2,730 train-fit conversations (about 10k turns):** about **70–80%**. Clearing 70 beats FinQANet. Getting to the high 70s matches GPT-4 zero-shot. Going past 80 probably needs reasoning traces, RL, or a bigger model; those are extensions in §4.2.
- **Breakdowns** (shapes to look for, from paper Tab. 4 and Fig. 5):
  - number-selection turns score much higher than program turns (82.5 vs. 62.1);
  - accuracy falls with turn index (turn 1 about 76%, turn 6 about 34%);
  - the second part of Type II conversations is hardest (about 52%).
- **Gold history:** expect it to score a few points above predicted history. That gap is the cost of compounding errors.

## Project layout
```
fine-tuning-lab/
  PLAN.md                   # this plan: the living source of truth, with Progress tracker + Decision log
  CLAUDE.md                 # session bootstrap: read PLAN.md, update progress, project conventions
  pyproject.toml            # uv; python 3.14; torch from the cu132 index; xgrammar; dev deps: ipykernel, matplotlib, pytest
  .gitignore                # data/raw/, artifacts/, .venv/, docs/_book/
  data/raw/                 # unzipped ConvFinQA json (untracked)
  artifacts/                # per-artifact outputs: processed/, runs/<name>/{node.json,build.log,metrics.json,predictions.jsonl,adapter/,tb/}, _chapters/
  src/convfinqa/
    data.py                 # load raw json, table→markdown, pretty-print a conversation, page-grouped val split
    executor.py             # parse + execute DSL programs, FinQA tolerance
    python_format.py        # DSL → straight-line Python translator; AST-whitelisted safe evaluator
    constrained.py          # EBNF grammars (DSL, Python subset); xgrammar HF logits processor
    formatting.py           # conversation → chat messages; assistant-only label mask; per-turn eval prompts
    train.py                # train(cfg) -> run_dir   (TRL SFTTrainer + PEFT; heavy artifact builder)
    evaluate.py             # evaluate(model_or_run, split, history_mode, output_format, grammar) -> metrics (heavy artifact builder)
    authoring.py            # schema validation, grounding check, answer filling for new examples
    runs.py                 # run ids, config/env capture (git commit, library versions)
    artifacts.py            # @artifact registry, node.json, recursive freshness, subprocess builds, build/status/refresh CLI
    stats.py                # bootstrap CIs, paired cluster bootstrap over conversations, cross-seed agreement
  tests/                    # test_executor, test_python_format, test_formatting, test_constrained, test_stats, test_authoring, test_splits
  docs/                     # THE LAB: Quarto book
    _quarto.yml             # type: book; parts as below; number-sections: true; execute: freeze: auto;
                            #   jupyter: fine-tuning-lab; bibliography: references.bib; code-fold
    references.bib          # ConvFinQA, FinQA, Transformer, InstructGPT, LoRA, QLoRA, PoT, PAL, self-consistency,
                            #   self-repair, "Let Me Speak Freely?", Fin-R1, FINDER, Qwen3 report
    index.qmd               # goals, roadmap, how to read the book, how to render/re-run a chapter
    # Part 1 — Setup & data
    101_environment_setup.qmd     # 1.1 Environment setup
    102_what_convfinqa.qmd        # 1.2 What ConvFinQA looks like
    103_the_dsl.qmd               # 1.3 The DSL and its executor
    # Part 2 — Foundations (light compute, freeze: auto)
    201_from_text.qmd             # 2.1 From text to tokens to next-token probabilities
    202_anatomy_of.qmd            # 2.2 Anatomy of Qwen3-4B
    203_from_base.qmd             # 2.3 From base model to instruct model
    204_probing_the.qmd           # 2.4 Probing the untuned model on ConvFinQA
    205_how_fine_tuning.qmd       # 2.5 How fine-tuning works
    206_lora_in.qmd               # 2.6 LoRA in detail
    # Part 3 — Experiments
    301_formatting_conversations.qmd  # 3.1 Formatting conversations for training
    302_the_untuned.qmd           # 3.2 The untuned baseline            (HEAVY)
    303_training_with.qmd         # 3.3 Training with LoRA              (HEAVY)
    304_evaluating_the.qmd        # 3.4 Evaluating the fine-tuned model (HEAVY)
    305_dsl_vs.qmd                # 3.5 DSL vs Python output            (HEAVY)
    306_constrained_decoding.qmd  # 3.6 Constrained decoding            (HEAVY)
    # Part 4 — Beyond
    401_writing_your.qmd          # 4.1 Writing your own examples
    402_retrospective.qmd         # 4.2 Retrospective
    # Appendices
    901_glossary.qmd              # A.1 Glossary
    902_references.qmd            # A.2 References
```

## Chapters

### Part 1 — Setup & data

**1.1 Environment setup** (`101_environment_setup.qmd`)
- **Before the book exists:**
  - `uv init`; pin Python 3.14; map torch to `https://download.pytorch.org/whl/cu132` under `[tool.uv.sources]`, as an explicit index.
  - Add the dependencies: `torch transformers datasets accelerate peft trl xgrammar`, plus dev `ipykernel matplotlib pytest`.
    - xgrammar 0.2.8 has a cp314 aarch64 wheel. The fallback is `transformers-cfg` (pure Python). `outlines` doesn't support 3.14.
  - Install Quarto 1.10.19 from the linux-arm64 tarball into `~/.local`.
  - Register the kernel: `uv run python -m ipykernel install --user --name fine-tuning-lab`.
  - Copy this plan into the repo as `PLAN.md`, including the Progress tracker and Decision log below.
  - Write a short `CLAUDE.md` that says:
    - read `PLAN.md` first;
    - update the Progress tracker and Decision log as work happens, and commit them with the related change;
    - remember the conventions: logic in `src/convfinqa/`, chapters thin, heavy artifacts built only with `RUN_HEAVY=1` or the `build` command, dev only for final numbers, `PCC_first_words.qmd` naming.
  - `git init`; write `.gitignore`; make the first commit (including `PLAN.md` and `CLAUDE.md`). Then `gh repo create yinchi/fine-tuning-lab --private --source . --push`.
- **The chapter:** checks that CUDA is available, the device is GB10, and a bf16 matmul runs. It prints library versions.
  - It explains the driver-vs-runtime distinction: `nvidia-smi` shows "CUDA 13.2", `torch.version.cuda` shows "13.2", and `nvcc --version` shows "13.2".

**1.2 What ConvFinQA looks like** (`102_what_convfinqa.qmd`)
- The download cell clones the repo and unzips into `data/raw/`, skipping if already present.
- Renders one Type I and one Type II conversation: the text, the table as markdown, and each turn's question → program → answer.
- Plots turns per conversation, the share of number-selection vs. program turns, the DSL operation frequencies, and the distribution of token counts for **full conversations**. These counts set `max_length`.

**1.3 The DSL and its executor** (`103_the_dsl.qmd`)
- Explains the DSL grammar, then walks through `executor.py` on the example above.
- Executes every gold `turn_program` and shows the match rate against `exe_ans_list`, with any mismatches explained.
- The same check runs in pytest.

### Part 2 — Foundations
These chapters are light on compute (minutes each) and use `freeze: auto`. They work on `Qwen/Qwen3-4B-Instruct-2507`, plus `Qwen/Qwen3-4B-Base` in 2.3 and `Qwen/Qwen2.5-0.5B-Instruct` for the small training demos. Examples come from **val** or train-fit, never dev.

**2.1 From text to tokens to next-token probabilities** (`201_from_text.qmd`)
- Tokenize a real ConvFinQA question and table row. Show how numbers like `206588` and `0.14136` split into tokens, which explains why the arithmetic goes to an executor.
- Trace the path token IDs → embeddings → transformer → logits → softmax.
- Show the top-10 next tokens for a ConvFinQA prompt, and the effect of temperature.
- Greedy vs. sampling decoding, and why greedy is deterministic (this links forward to the seed discussion in §3.4).
- Logits processors, i.e. masking logits, which sets up §3.6.

**2.2 Anatomy of Qwen3-4B** (`202_anatomy_of.qmd`)
- Read `config.json`: number of layers, hidden size, attention heads vs. key/value heads (grouped-query attention), RoPE, vocabulary, context length.
- Print one decoder layer's module tree. Use forward hooks on one ConvFinQA input to show tensor shapes flowing through attention and the MLP (SwiGLU) with RMSNorm.
- Parameter count by component (embeddings, attention q/k/v/o, MLP gate/up/down), and memory accounting: bf16 weights, plus the KV cache per token for a typical conversation length.
- Which linear layers LoRA's `all-linear` will target. This sets up 2.6.

**2.3 From base model to instruct model** (`203_from_base.qmd`)
- The training pipeline: pretraining → SFT → preference tuning / RL. Explain what "Instruct-2507" means (a non-thinking variant).
- Hands-on: the same prompt to `Qwen3-4B-Base` (it continues the text) and to Instruct (it answers).
- Chat templates: render messages to the raw string, explain the special tokens and the generation prompt, and show how this tokenizes.
- Multi-turn SFT: teacher forcing, exposure bias, and how later stages (self-generated data, DPO, multi-turn RL) address it. Also why per-turn prompt/completion is the less common layout.

**2.4 Probing the untuned model on ConvFinQA** (`204_probing_the.qmd`)
- Run a handful of val conversations zero-shot and few-shot, read the outputs, and sort the failures by type.
- Per-token log-probabilities of the **gold** programs: a heatmap of which tokens surprise the model. This is exactly the loss fine-tuning will reduce.
- Compute the mean gold-program negative log-likelihood on val. That number is the predicted step-0 val loss for §3.3.
- Optional: attention from a copied number back to table cells. The chapter notes that attention maps are suggestive, not explanations.

**2.5 How fine-tuning works** (`205_how_fine_tuning.qmd`)
- Next-token cross-entropy, computed by hand on one example with `-100` label masking, then shown to match `model(labels=...).loss`.
- Backprop and AdamW: what the optimizer state holds; warmup + cosine LR plotted; gradient accumulation and effective batch size; bf16; the trade-off gradient checkpointing makes.
- Memory budget: full fine-tuning of 4B parameters (weights + gradients + Adam states) vs. LoRA, computed for our model.
- Hands-on with the 0.5B model:
  - a hand-written PyTorch training loop of about 30 lines on a few conversations, showing the loss falling and a before/after generation;
  - the same thing in `SFTTrainer`, mapping each loop line to a Trainer argument.

**2.6 LoRA in detail** (`206_lora_in.qmd`)
- The maths: `W' = W + (α/r)·B·A`, with `B` initialized to zero.
  - Verify numerically that the wrapped model's outputs at step 0 are identical to the base model's.
- A single `nn.Linear` wrapped by hand, then by PEFT, and compared.
- Trainable-parameter count for r=16 `all-linear` on Qwen3-4B (`print_trainable_parameters`). Intuition for rank and alpha. Merging an adapter. QLoRA, mentioned for completeness.

### Part 3 — Experiments

**3.1 Formatting conversations for training** (`301_formatting_conversations.qmd`)
This chapter uses standard multi-turn SFT: one training sequence per **conversation**, with teacher forcing. Background is in 2.3.
- **Messages:**
  - System: "Answer with a program in the ConvFinQA DSL."
  - User turn 1: the context (`pre_text`, the markdown table, `post_text`) followed by question 1.
  - Assistant turn 1: gold `turn_program[0]`.
  - Then user turn k: question k, and assistant turn k: gold `turn_program[k]`, for the rest of the conversation.
- **Loss** is computed on **assistant tokens only**, all turns at once. The page is encoded once per conversation, not once per turn, which uses about 3–4× less compute than a per-turn layout.
- **Masking:**
  - Check whether the Qwen3 chat template has `{% generation %}` tags, which TRL's `assistant_only_loss=True` needs.
  - If it doesn't, have `formatting.py` tokenize message by message and build `labels`, with -100 on system and user tokens. Write these as pre-tokenized `input_ids` and `labels`.
  - Either way, the chapter renders one conversation with the trained tokens highlighted.
- **Validation split** (`data.make_val_split`):
  - Group key: the `id` with the `Single_`/`Double_` prefix and the `-N` suffix removed, e.g. `JKHY/2009/page_28.pdf`.
  - Hold out 10% of groups, using a fixed seed.
  - Report how many pages are shared between train and dev, for reference only.
- `output_format: "dsl" | "python"` is a parameter (§3.5).
- **Outputs:**
  - `artifacts/processed/train_fit.jsonl`: about 2,730 conversations.
  - `artifacts/processed/val.jsonl`: about 300 conversations.
  - `dev` files in two shapes: per-conversation (421) and per-turn (1,490).

**3.2 The untuned baseline** (`302_the_untuned.qmd`, HEAVY)
- Runs `Qwen/Qwen3-4B-Instruct-2507` untuned, with 2–3 in-context examples, on dev.
  - The examples come from **train-fit**: 3 fixed sets, each drawn with its own seed. Report the mean ± std across sets and the per-set results.
  - Any prompt iteration happens on **val**. Dev is run once, with the final prompt.
- Uses greedy, batched generation with left padding. Each predicted program is parsed and executed.
- Reports **execution accuracy** (the headline metric), program accuracy, and the invalid rate. These follow the evaluation rules in §3.4.

**3.3 Training with LoRA** (`303_training_with.qmd`, HEAVY)
- **Base models:**
  - Smoke run: `Qwen/Qwen2.5-0.5B-Instruct`, 50 steps.
  - Main run: `Qwen/Qwen3-4B-Instruct-2507`.
  - Optional scale-up later: `Qwen/Qwen3.5-4B` or `Qwen/Qwen3.5-9B`.
    - These are newer (Feb 2026) and there is a public ConvFinQA point of comparison for them.
    - They aren't the default because they are vision-language models (`Qwen3_5ForConditionalGeneration`). Training them in TRL needs a little more setup than a text-only model.
    - Their hybrid-attention architecture may need fused kernels, which might not exist for aarch64 with Python 3.14. Without them, training could be slow.
    - Try them after the main pipeline works.
- **Config** is shown as a table and explained parameter by parameter, with links to 2.5 and 2.6:
  - LoRA: r=16, alpha=32, dropout 0.05, `all-linear`.
  - Optimization: bf16, gradient checkpointing; lr 2e-4, cosine schedule, warmup 0.03; 2 epochs; effective batch 16 (4×4); `max_length` from §1.2.
  - `assistant_only_loss=True`, or our pre-built labels.
  - About 340 optimizer steps (2,730 conversations × 2 epochs / 16). Eval every 50 steps on the **val** split; keep the best checkpoint by val loss.
  - After training, run a quick val execution-accuracy check. Dev is never looked at here.
- `train(cfg)` writes to `artifacts/runs/<run_id>/`.
- The chapter plots train and eval loss from `trainer_state.json`.
- Expected time for the main run: about 1–2 hours on GB10. It is built by the full-book render with `RUN_HEAVY=1` in `tmux` or `screen` (see 1.1).
- **Under the hood:**
  - The step-0 val loss should match the gold-program negative log-likelihood measured in 2.4.
  - Adapter analysis: the size of the weight change each adapter makes (‖(α/r)·B·A‖) as a heatmap of layer × module (q/k/v/o/gate/up/down), and the singular-value spectrum of those weight changes, which shows how much of rank 16 is actually used.
  - Re-run 2.4's per-token log-prob heatmap on the same val example with the adapter on, to show which tokens became easier.

**3.4 Evaluating the fine-tuned model** (`304_evaluating_the.qmd`, HEAVY)
- Compares the fine-tuned model with the baseline and with the reference table above, rendered as a chart with our two points overlaid on the published results.
- Accuracy is broken down by turn index and by Type I vs. Type II.
- Two history modes:
  - **predicted history (headline):** run each dev conversation turn by turn, feeding back the model's own earlier programs; score every turn.
  - **gold history (upper bound):** each turn is scored independently, with the gold programs as the earlier answers.
  - Report both, and plot the gap by turn index to show error compounding.
- The chapter explains these terms up front: gold history = teacher forcing. No conversations or turns are filtered out in either mode.
- **Invalid-output policy (headline):** greedy decoding, **one attempt, invalid counts as wrong.**
  - This matches the official FinQA/ConvFinQA `evaluate.py` (`invalid_flag` → counted wrong, denominator is all examples, results rounded to 5 decimal places) and PoT greedy decoding, so it's comparable with the reference table.
  - In predicted-history mode, an invalid output stays in the history exactly as generated.
  - Report the invalid rate by cause: parse error, unknown operator, bad `#k`, division by zero, rejected by the safe evaluator, empty output.
- **Program accuracy:** computed with sympy symbolic equivalence, as the official script does (`equal_program`), not by string match.
- **Extra-compute comparison** on the fine-tuned model, reported in separately labeled rows and never mixed into the headline:
  - **retry with feedback:** on an invalid output, append the error message and regenerate, up to 2 extra attempts;
  - **self-consistency:** sample N=5 at temperature 0.5, drop invalid programs, majority-vote on executed answers (PoT used n=30, T=0.5); all invalid → wrong.
  - Plot accuracy against the average number of generations per turn, so the two are compared at equal compute (following Olausson et al. on self-repair).
- **Uncertainty and seeds** (`stats.py`, also used in §3.5 and §3.6):
  - Every A-vs-B comparison gets a **paired cluster bootstrap**: resample the 421 dev conversations with replacement, 10,000 times, and score both systems on the same resample.
  - Report the 95% CI of the accuracy difference and an approximate p-value. Single-system accuracies also get bootstrap CIs.
  - Greedy evaluation is deterministic, so the generation seed doesn't matter here. Only the **training seed** does: LoRA initialization, data order, dropout.
  - A comparison whose CI straddles 0 gets 2 extra training seeds for each side. Report mean ± std across seeds, plus **cross-seed agreement**, the share of turns where all seeds give the same answer, and list the turns where they disagree in the error analysis.
  - All of these are **dev** scores.
  - **Untuned models** have no training seed. Their equivalent is the **few-shot example set**.
    - §3.2 always uses 3 sets, reported as mean ± std, as the ConvFinQA paper did for GPT-3 (Tab. 5).
    - §3.5 and §3.6 untuned rows use set 1. A comparison that straddles 0 adds sets 2–3, mirroring the training-seed rule.
  - **Base vs. fine-tuned:**
    - The paired bootstrap applies as usual. With an expected gap of about 15 points, it should settle the comparison without extra seeds.
    - The base model is evaluated few-shot and the fine-tuned model zero-shot, each in its best setting. The chapter states this explicitly.
  - The chapter explains the procedure with a worked example:
    1. Train A0 and B0 (seed 0) and score both on dev.
    2. Run the paired bootstrap on the difference.
    3. If its CI excludes 0, stop.
    4. If it straddles 0, train A1/A2 and B1/B2 and check whether the gap holds across seeds.
  - It includes a table of the two noise sources: test-sample noise, measured by the bootstrap, and training-run noise, measured by seeds.
- Error analysis: a sample of failures sorted into wrong number picked, wrong operation, broken `#k` reference, or a program that doesn't parse.
  - **Under the hood:** for some failures, compare base and fine-tuned log-probs of the gold program vs. the predicted one. Was the right answer close?

**3.5 DSL vs Python output** (`305_dsl_vs.qmd`, HEAVY)
- **Hypothesis:** models trained on lots of code write Python more reliably than an unfamiliar DSL.
  - Prior evidence: GPT-3 scored 45.2 with ordinary math notation vs. 40.8 with the original DSL (ConvFinQA paper, Tab. 5). PoT and FINDER both use Python.
  - Expectation: a clear gap untuned, which shrinks after fine-tuning.
- **Translation** (`python_format.py`):
  - Each DSL step becomes one line, `vk = <a> <op> <b>`.
  - Operator mapping: `add/subtract/multiply/divide/exp/greater` → `+ - * / ** >`. `#k` → `vk`, and `const_N` → the number N (`const_m1` → `-1`).
  - The last line is `ans = v<last>`, and a bare-number turn becomes `ans = 206588`.
  - Any `table_*` ops are mapped or flagged.
- **Execution:** a safe evaluator parses with `ast` and allows only `Assign`, `BinOp`, `UnaryOp`, `Compare`, `Name`, and numeric `Constant`. It never calls `exec`. The chapter explains why.
- **History:** turns stay self-contained, as in ConvFinQA, where programs restate their numbers. The model copies numbers from the chat history, so no new mechanism is needed.
- **Runs:**
  - The formatting, baseline, training, and evaluation functions take `output_format: "dsl" | "python"`. This chapter calls them with `"python"`, using the same model, hyperparameters, and few-shot examples, and only the output syntax changes.
- **Results:** a 2×2 table of {DSL, Python} × {untuned, fine-tuned}, showing:
  - execution accuracy, in both history modes;
  - the parse/format failure rate;
  - accuracy by turn index;
  - the retry and self-consistency variants from §3.4.
  - Prediction: if Python mainly cuts invalid outputs, retries will help the DSL more than Python, and the two formats will end up close.
  - Program accuracy is left out because it can't be compared across syntaxes.
- **Possible follow-up (§4.2):** a REPL-style variant where variables persist across turns.

**3.6 Constrained decoding** (`306_constrained_decoding.qmd`, HEAVY)
This is inference-only and reuses the §3.2 base model, the §3.3/§3.5 adapters, and the §3.4 evaluation (predicted history; greedy; one attempt; invalid counts as wrong).
- **Under the hood:** a grammar is a logits processor (see 2.1). Show the mask at one decoding step: which tokens are allowed vs. what the model would have picked.
- **Observation, not an assumption:** with the Qwen3-4B-Instruct-2507 tokenizer, every DSL operator in the gold programs is a single token, bare or with a leading space. This holds for all 9,775 operator occurrences in the 8,981 unique train and dev programs, and for all 12,806 occurrences inside the chat-formatted conversations; there are 11 distinct token strings for the 6 operators, and `exp` only ever appears with the leading space. It makes the operator-position mask small and easy to show. Grammars are written at the character level and must not depend on it: the model may emit any token sequence, and another tokenizer or base model may split operators differently. A chapter that uses the fact asserts it in a cell or test.
- **Syntaxes:**
  - **DSL:** least familiar, so the constraint should matter most.
  - **Python subset:** the same subset the safe evaluator accepts.
  - `bc` and `dc` are left out, and the chapter explains why:
    - `bc` is the same infix arithmetic as the Python subset, and its default `scale=0` turns division into integer division. Running it also means calling an external program.
    - `dc` is reverse Polish notation and rare in training data.
- **Grammar levels:**
  1. **none**;
  2. **syntax:** a static EBNF grammar for each format.
- A grounded variant, which restricts numbers to those in the context, is deferred to §4.2.
- **Runs:** {DSL, Python} × {none, syntax} × {untuned few-shot, fine-tuned}, for 8 evaluations.
- **Report:**
  - execution accuracy and invalid rate;
  - accuracy on number-selection vs. program turns;
  - generation speed, i.e. the overhead of the grammar.
- **Hypotheses:**
  - The syntax constraint drives the invalid rate to about 0 and helps the untuned DSL model most.
  - After fine-tuning, the gains are small.
- **Discussion:** format restrictions can hurt accuracy (Tam et al. 2024, "Let Me Speak Freely?"). Did we see it? This connects back to FinQANet's restricted decoder.

### Part 4 — Beyond

**4.1 Writing your own examples** (`401_writing_your.qmd`)
- **Source material:** take a 10-K page from SEC EDGAR, or reuse a FinQA page. Extract one table into `table` (rows, header first) and the surrounding sentences into `pre_text` and `post_text`.
- **Write a target calculation** as a FinQA program.
- **Decompose it into turns** the ConvFinQA way:
  1. number-selection turns first;
  2. then partial computations;
  3. then a final turn that restates the full program with `#k` references.

  Use follow-up phrasing so later turns depend on the earlier ones. Optionally, chain a second program on the same page to make a Type II conversation (`qa_split` 0s then 1s).
- **Validate** with `authoring.py`:
  - check the schema;
  - execute each `turn_program` and fill in `exe_ans_list`;
  - check grounding: every literal number must appear in the table or text.
- The chapter shows one hand-written example passing and two failing: a wrong answer and an ungrounded number.
- **Scale-up (optional):** an LLM drafts candidate dialogues, and only examples that pass validation are kept.
- New examples share the raw JSON shape, so §3.1's code consumes them unchanged. Keep them out of dev.

**4.2 Retrospective** (`402_retrospective.qmd`)
- What worked, what didn't, and what to try next:
  - model size (Qwen3.5-4B/9B), LoRA rank;
  - adding synthetic data from §4.1;
  - predicted-history training;
  - SFT on reasoning traces followed by the DSL;
  - RL with an execution-match reward. This is TRL GRPO, the route Fin-R1 took to 85.
  - a REPL-style Python variant where variables persist across turns;
  - constraining `#k` references so they can only point to earlier steps, which needs a step-count-aware grammar;
  - a **grounded grammar**, compiled per example, where numbers may only be literals from that page's text, table, and earlier turns, plus `const_*`:
    - First measure coverage: the share of gold programs it can still produce. Unit scaling and percentage forms will break some. That coverage is the accuracy ceiling.
    - Then rerun the §3.6 evaluations with it.

### Appendices
- **A.1 Glossary** (`901_glossary.qmd`): every term the chapters link to, e.g. logits, teacher forcing, exposure bias, LoRA rank, effective batch, execution accuracy, paired bootstrap.
- **A.2 References** (`902_references.qmd`): the bibliography rendered from `references.bib`.

## Progress tracker
Mark each item `[x]` when it's done and verified. Add a short note (run id, headline number) where useful.

- [x] 1.1 Environment: uv project, Quarto 1.10.19, kernel, git + private GitHub repo `yinchi/fine-tuning-lab`, `PLAN.md`/`CLAUDE.md` committed. Chapter renders: GB10, CUDA 13.2 everywhere, bf16 matmul OK, torch 2.14.1+cu132.
- [x] 1.2 Data downloaded and explored; `max_length` chosen. Train 3,037 convs / 11,104 turns, dev 421 / 1,490; test has no labels. Longest full conversation 3,460 tokens, so `max_length` = 3584 (saved to `artifacts/processed/lengths.json`; tied to the Qwen3 tokenizer).
- [x] 1.3 Executor; 100% gold match. `executor.py` reproduces all 11,104 train and 1,490 dev gold answers exactly (0 mismatches, 45 tests pass); errors reported by kind.
- [ ] Artifact system (`artifacts.py`, `test_artifacts.py`): needed before the first heavy cell (2.4)
- [x] 2.1 From text to tokens to next-token probabilities. Qwen3-4B-Instruct-2507: 151,669-token vocabulary, digits are single tokens; the trace on the AAPL turn-1 prompt (1,125 tokens) matches `model(...).logits` exactly; untuned top token is a code fence (p=0.614), gold first token `5` has p=0.024 (rank 5); greedy is repeatable, same-seed sampling is repeatable; a digits-only mask makes greedy output `5363`.
- [ ] 2.2 Anatomy of Qwen3-4B
- [ ] 2.3 From base model to instruct model
- [ ] 2.4 Probing the untuned model; val gold-program NLL recorded
- [ ] 2.5 How fine-tuning works; hand-written loss matches the model's loss
- [ ] 2.6 LoRA in detail; step-0 equality check passes
- [ ] 3.1 Formatting + splits; label-mask test passes
- [ ] 3.2 Untuned baseline, 3 example sets (DSL)
- [ ] 3.3 Smoke run (0.5B)
- [ ] 3.3 Main LoRA run (Qwen3-4B, DSL); adapter analysis
- [ ] 3.4 Evaluation: both history modes, invalid breakdown, retry/self-consistency, bootstrap CIs, error analysis
- [ ] 3.5 Python format: translator + safe evaluator, untuned eval, training run, comparison
- [ ] 3.6 Constrained decoding: grammars, 8 evaluations
- [ ] 4.1 Writing your own examples
- [ ] 4.2 Retrospective
- [ ] A.1 Glossary / A.2 References complete; full book renders from freeze

## Decision log
Dated entries, newest last. A changed decision gets a new entry; old ones are never rewritten.

- 2026-10-20: Base model Qwen3-4B-Instruct-2507 (smoke run: Qwen2.5-0.5B-Instruct). LoRA SFT with TRL, bf16. Output is the DSL program, executed in Python.
- 2026-10-20: uv + Python 3.14; torch cu132 (matches driver and nvcc 13.2).
- 2026-10-20: The Quarto book is the keystone of the lab. Heavy chapters use `freeze: true` plus a DONE guard. Files named `PCC_first_words.qmd`, with codes in the titles.
- 2026-10-20: Training is whole-conversation SFT with assistant-only loss and gold history. The headline eval uses predicted history; gold history is the upper bound.
- 2026-10-20: Headline rule: greedy, one attempt, invalid counts as wrong. Retry and self-consistency go in separately labeled rows.
- 2026-10-20: The val split is 10% of train, grouped by report page. Dev is for final numbers only.
- 2026-10-20: Noise: 1 seed plus a paired cluster bootstrap; extra seeds or example sets only when a CI straddles 0.
- 2026-10-20: Added the DSL-vs-Python ablation (§3.5) and syntax-constrained decoding with xgrammar (§3.6). The grounded grammar is deferred to §4.2. `bc`/`dc` rejected.
- 2026-10-20: Added Part 2 Foundations as teaching material, with "Under the hood" links from the experiment chapters.
- 2026-10-20: Private GitHub repo `ycchan-mumu/fine-tuning-lab`; TensorBoard for tracking.
- 2026-10-20: Project moved to the personal Linux account. The GitHub repo is now `yinchi/fine-tuning-lab` (private), replacing `ycchan-mumu/fine-tuning-lab`. Quarto is installed under `~/.local/opt/quarto`, symlinked into `~/.local/bin`.
- 2026-10-20: Native Quarto numbering replaces manual codes in titles (`number-sections: true`, appendices under `appendices:`). Chapter numbers run straight through the book and no longer match the `PCC` filenames or the plan's §P.C identifiers. This supersedes the numbering part of the 2026-10-20 decision on file naming.
- 2026-10-20: In VS Code, the Quarto extension exports `QUARTO_PYTHON` from the selected Python interpreter, overriding `docs/_environment*`. A bare interpreter made re-renders fail with `No module named 'yaml'`. The committed `.vscode/settings.json` selects `.venv`; no `_environment` file is used.
- 2026-10-21: Chapter environment variables live in a committed `docs/_environment` (`TQDM_DISABLE`, `HF_HUB_DISABLE_PROGRESS_BARS`, and a `PYTHONWARNINGS` filter for the tqdm IProgress warning). HTML code blocks default to `code-fold: show`; heavy cells can fold with `#| code-fold: true`. ConvFinQA facts for the DSL: 18 official constants; `%` literals evaluate as /100; the paper's generator can copy any page token that parses as a number or ends in `%`.
- 2026-10-21: No `freeze: true`. Tested in a scratch book: a frozen chapter ignores prose edits in a whole-book render, and rendering it directly re-executes every cell. All chapters use `freeze: auto`. Heavy cells (`#HEAVY`) load results from `artifacts/<run_id>/` (DONE + matching `config.json`) and raise an error if they are missing, unless `RUN_HEAVY=1` or `force=True`. This supersedes the 2026-10-20 decision that heavy chapters use `freeze: true`.
- 2026-10-21: Artifact system replaces the `DONE` + `force=True` guard (this supersedes the previous entry's mechanism, not its `freeze: auto` decision). Named artifacts with string deps and name templates in a registry; per-artifact `node.json` with config, dependency hashes and content hashes; **recursive** freshness; light artifacts build in-kernel, heavy ones in a subprocess only with `RUN_HEAVY=1` (or the `build` command); builders own their models and use only parameters and dependency handles; chapter records plus `refresh` re-render chapters whose artifacts changed. DVC was considered and rejected for now: the helper keeps training calls visible in chapters and needs no new tool.
- 2026-10-21: The recommended build is a full-book render with `RUN_HEAVY=1`, wrapped in `tmux` or `screen`; the artifact freshness check decides which heavy artifacts to rebuild. `refresh` runs automatically through Quarto's `pre-render` hook (tested in a scratch project: it runs before `render` and at preview start, inherits `RUN_HEAVY`, and runs from the project directory under the system `python3`, so the hook is a shim that starts the venv's Python). A frozen chapter is never executed, so without `refresh` a stale artifact in an unchanged chapter would go unnoticed.
- 2026-10-21: Evaluation and generation code always passes `do_sample=False` (and `temperature=None, top_p=None, top_k=None`) for greedy decoding, and sets every sampling setting explicitly otherwise. Qwen3-4B-Instruct-2507 ships generation defaults of temperature 0.7, top_p 0.8, top_k 20, so a bare `generate()` samples.
- 2026-10-21: Math in the book stays as LaTeX through Quarto's default MathJax 4 from a CDN. Vendoring MathJax is deferred until offline reading matters. Single symbols such as a minus infinity may be written as plain text.

## Verification
- `uv run pytest`: the executor reproduces the gold `exe_ans_list` values, and the formatting and authoring tests pass.
  - The formatting test decodes the tokens where `labels != -100` and asserts they are exactly the assistant programs, plus the end-of-turn tokens.
  - The Python translation round trip: every gold `turn_program` in train and dev, once translated and run by the safe evaluator, equals the DSL executor's result.
  - The safe evaluator rejects `__import__('os')`, attribute access, calls, and loops.
  - Grammars: every gold train and dev program, in both formats, is accepted by its syntax grammar.
  - A short constrained generation through xgrammar's HF logits processor works on GB10.
  - `stats.py` checks:
    - identical systems give a difference CI centered on 0;
    - a planted +5-point difference on synthetic data is detected;
    - the cluster bootstrap gives a wider CI than a turn-level bootstrap when outcomes are correlated within a conversation.
  - Split check: no page group appears in both train-fit and val.
- Foundations checks, written as asserts in the chapters so a failing check stops the render:
  - **2.5:** the hand-computed cross-entropy equals `model(labels=...).loss` to within 1e-4.
  - **2.6:** at step 0, the LoRA-wrapped model's logits equal the base model's exactly; the trainable-parameter count matches PEFT's.
  - **3.3:** the step-0 val loss is within a few percent of 2.4's gold-program negative log-likelihood (the formatting is identical).
- `RUN_HEAVY=1 quarto render docs/303_training_with.qmd` (smoke config first) finishes, writes `artifacts/runs/<name>/node.json`, and shows falling loss curves.
- After everything has run once, a full `quarto render docs/` finishes in minutes. Unchanged chapters come from `_freeze/`.
- Editing prose in a heavy chapter and rendering it re-executes the chapter, and the heavy cells load their artifacts and show the edit. With an artifact removed, the same render fails with the error naming the build command, and starts no training. `RUN_HEAVY=1` builds it in a subprocess.
- Artifact tests (`test_artifacts.py`): fresh; own config changed; a dependency rebuilt; a stale middle node with a fresh-looking leaf (recursive check); missing dependency; early cutoff on identical output; `RUN_HEAVY` set to `1`, `0` and unset; an interrupted build leaves no `node.json`; the graph has no missing deps or cycles; `refresh` deletes exactly the stale chapters' freeze folders.
- In the book, §1.2 shows real conversations, and §3.4 shows fine-tuned execution accuracy clearly above the §3.2 baseline.
- The rendered book shows chapter codes matching the filenames, and has working cross-references, glossary links, and a bibliography. Part 2 renders in a few minutes from a clean freeze.
