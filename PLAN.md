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
5. **Artifacts are named, tracked, and built on demand** (`src/convfinqa/artifacts/`). An artifact is a named result on disk, such as a processed split, a trained adapter, or an evaluation.
   - **Registration.** A builder is declared where it is used: in the chapter's code cell, right above the call that loads the artifact. E.g. `@artifact("sft_{variant}_{seed}", deps=["tokenized_{variant}"], files=["configs/sft.json"], heavy=True, version=1)` on a function, followed by `sft = get("sft_dsl_0")  #HEAVY`. The registry reads every `.qmd` file with `ast` (without running it; `artifacts/chapters.py`), so all builders and all literal `get("name")` calls are known before any chapter runs; that is what lets `build --all`, `status` and `refresh` work up front, across chapters. `build --all` builds the names found in `get("literal")` calls (a name built from an f-string is built on demand only). Builders shared by several chapters, or not tied to one, may instead live in modules listed in `DEFAULT_MODULES` (`convfinqa.builders`). The whole graph is validated when the registry loads (every dependency resolves, no cycles), and each literal `get` name is checked with its `file:line`.
   - **Names.** Letters, digits and underscores only; `_` separates the fields of a template. Fields are `str` (letters and digits) or `int` (canonical digits, no leading zeros), so a name matches exactly one way and cannot contain a path separator.
   - **Builders.** The signature is `(out, deps, files, *, <name fields>)` and nothing else (`files` only when declared). `out` is a scratch dir, `deps` maps each dependency name to a `Handle` (`.path`, `.hash`, `.file/.json/.jsonl`), `files` maps each declared static file to an `InputFile` (`.path`, `.hash`, readers). The name fields (`variant`, `seed`) are the artifact's identity. Hyperparameters and prompts therefore live in a declared file or a light config artifact, not in keyword arguments. Every builder, in a chapter or in a module, must be self-contained: it may use only its parameters, builtins and imports made inside it (absolute ones), which is what makes hashing the function alone sound; this is checked when it is registered. A builder in a chapter must also be a plain top-level `def` of a `{python}` cell (cells marked `#| eval: false` are ignored by the scan; `get` calls may sit anywhere, flow control included) and the check uses it may use only its parameters, builtins, and imports made inside it (checked with `symtable` at scan time), and it compiles with the `.qmd` as its filename and real line numbers, so tracebacks point into the chapter. Its logic is a few lines of glue around functions in `src/convfinqa/`, which stay testable. Enforced at registration (signature and purity), in the subprocess (clean cwd) and in tests. A builder loads its own model and needs no teardown, because heavy builders run in their own process. In the running chapter the decorator is a no-op, because the scan already registered that definition.
   - **Node files.** Each artifact has `artifacts/<name>/node.json`: config (name fields, input-file hashes, seed), the output hashes of its dependencies, content hashes of the outputs (logs and TensorBoard files excluded), a `code_hash`, a manual `version`, git commit and library versions. It is written last and atomically, after building in `artifacts/<name>.tmp/`.
   - **Freshness is recursive.** An artifact is fresh only if its `version`, config and `code_hash` match, **every dependency is itself fresh**, and the dependencies' current output hashes equal the recorded ones. Content hashes give early cutoff: a deterministic rebuild with identical output leaves dependents fresh. `code_hash` covers the normalized source of the builder's own function, without its decorator (`ast.unparse` with docstrings and annotations on function variables dropped, so whitespace, comments and line continuations don't matter; annotations on class and module variables are kept, because they define dataclass fields), and every `convfinqa.*` module that function imports (static import closure, plus modules loaded dynamically during the build). Nothing else in the file the builder lives in counts: other builders, other cells, prose and module-level code never invalidate it. The framework itself is excluded. Library-version and Python-minor mismatches are warnings, not staleness; `version` is the manual override for anything else.
   - **Light vs. heavy.** Light artifacts rebuild in the notebook kernel whenever stale. A stale or missing **heavy** artifact raises `StaleArtifactError` with the build command, unless `RUN_HEAVY=1` is set, or the `build` command is used; then it builds in a subprocess (`python -m convfinqa.artifacts _run <name>`), logging to `artifacts/<name>/build.log`. The subprocess frees all memory on exit and isolates crashes. A file lock per artifact prevents two concurrent builds. With `ALLOW_STALE=1`, a stale-but-present heavy artifact loads with a visible warning box (`Handle.stale`); a missing one still raises.
   - **Seeds.** The seed is part of the config (a `seed` name field, else the decorator's `seed`). The framework calls `seed_everything` (Python, NumPy, torch, CUDA) at the start of every build, and the subprocess gets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`; so a build's random draws don't depend on earlier RNG use. In-kernel (light) builds restore the global RNG state afterwards. GPU kernels are not bit-reproducible, so a retrained adapter normally has a new hash and invalidates its dependents; that is the correct behaviour.
   - **Across chapters.** Artifacts are global by name. Each chapter records the output hashes of the artifacts it used in `artifacts/_chapters/<chapter>.json` (from `QUARTO_DOCUMENT_FILE`; also whether any were used stale). `python -m convfinqa.artifacts refresh` deletes `docs/_freeze/<chapter>/` and the record for chapters whose artifacts changed, are stale, or were used stale, since Quarto's freeze only watches the `.qmd`. It runs automatically before every render and preview through `pre-render` in `_quarto.yml`, via the stdlib-only shim `scripts/pre_render.py`, which starts the project venv's Python. Other subcommands: `build <name>... | --all [--force]` and `status` (a fresh/stale/missing tree with reasons).
   - **Chapter conventions.** Cells that call heavy artifacts carry a `#HEAVY` comment. **The recommended build is `./build.sh`:** inside `screen`, `build --all` builds the missing and stale artifacts in DAG order (one subprocess per heavy node), then `quarto render docs/` runs without `RUN_HEAVY` (a leftover stale artifact fails loudly instead of starting hidden training), then `ALLOW_STALE=1 quarto preview docs/` serves the book so prose edits keep working. `RUN_HEAVY=1 quarto render docs/` remains a fallback.
6. **Long runs:**
   - Run `./build.sh` (it uses `screen`; for `tmux`, run `uv run python -m convfinqa.artifacts build --all` there, then render).
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
   - A YAML `title:` cannot carry a label (tested: the reference stays unresolved and the title shows the literal `{#sec-...}`), so a chapter's title is its first heading, `# Title {#sec-name}`. Prose refers to written chapters with `@sec-name` (rendered as "Chapter 7") and to chapters not yet written in words ("the LoRA chapter"), converted to `@sec-` when the chapter exists. `tests/test_book_refs.py` enforces labelled titles, resolvable references, unique labels and no `chapter 2.4` / `§3.3` in the prose.

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
    # Layers import downward only (tests/test_layers.py); only llm-facing modules may import torch.
    dsl/                    # pure Python, imports nothing from the package
      executor.py           #   parse + execute DSL programs, FinQA tolerance
      steps.py              #   program text helpers: split steps, number-selection turns, copyable numbers, constants
      python_format.py      #   (planned) DSL → straight-line Python translator; AST-whitelisted safe evaluator
    data/                   # imports dsl at most
      raw.py                #   download, load splits, Turn, turns(); render.py: table→markdown, context, pretty-print
      splits.py             #   (planned) page-grouped val split
      authoring.py          #   (planned) schema validation, grounding check, answer filling for new examples
    formatting.py           # conversation → chat messages; assistant-only label mask; per-turn eval prompts
    constrained.py          # EBNF grammars (DSL, Python subset); xgrammar HF logits processor
    train.py, evaluate.py   # (planned) TRL SFTTrainer + PEFT; evaluation; heavy artifact builders. They and formatting/constrained move
                            #   into an `llm/` package when train.py arrives.
    stats.py                # (planned) bootstrap CIs, paired cluster bootstrap over conversations, cross-seed agreement
    runs.py                 # (planned) run ids, config/env capture (git commit, library versions)
    artifacts/              # the artifact system; `python -m convfinqa.artifacts build|status|refresh`
      # what exists (no disk access)
      spec.py               #   Spec and the declaration rules: name templates, builder signature, field types (make_spec)
      chapters.py           #   scan .qmd files for @artifact cells and get() calls; purity check; compile with qmd line numbers
      catalog.py            #   Catalog: load builders (modules, chapters), resolve names, validate the graph; LoadState
      # what is on disk and what ran
      store.py              #   Store: directory layout, node.json, atomic commit, lock
      records.py            #   ChapterRecord(s): which artifacts each chapter used; refresh clears outdated freeze folders
      provenance.py         #   git commit/dirty, library and Python versions; drift_notes
      seeding.py            #   seed_everything, preserved_rng_state (global generators: seed, then put back)
      subprocess_build.py   #   child_command, child_environment, run: one heavy build in a child process
      # rules and types
      policy.py             #   Policy (RUN_HEAVY, ALLOW_STALE read from the environment), Action, messages
      handles.py            #   Handle, InputFile, Deps, Files, Status;  errors.py: the exceptions
      hashing.py            #   file/output hashes, normalized source, module closure, code hash, framework exclusion
      # coordination
      registry.py           #   Registry: composes catalog + store + records; status, ensure, execute, get; DEFAULT
      cli.py, __main__.py   #   command line
    book/                   # presentation helpers for the chapters
      viz.py                #   token displays and next-token plots
      diagrams/             #   SVG generators: `python -m convfinqa.book.diagrams`
  scripts/pre_render.py     # Quarto pre-render shim: starts the venv's Python for `artifacts refresh`
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
- Print one decoder layer's module tree. Use forward hooks on one ConvFinQA input to show tensor shapes flowing through attention and the feed-forward network (FFN, called `mlp` in the library; SwiGLU) with RMSNorm. Use "feed-forward network" in prose and diagrams, matching Vaswani et al., and mention `mlp` once as the code name.
- Parameter count by component (embeddings, attention q/k/v/o, feed-forward gate/up/down), and memory accounting: bf16 weights, plus the KV cache per token for a typical conversation length.
- Which linear layers LoRA's `all-linear` will target. This sets up 2.6.

**2.3 From base model to instruct model** (`203_from_base.qmd`)
- The training pipeline: pretraining → SFT → preference tuning / RL. Explain what "Instruct-2507" means (a non-thinking variant).
- Hands-on: four prompts (three unrelated to finance, plus ConvFinQA turn 1) to `Qwen3-4B-Base` and to Instruct, each as plain text and in the chat template. The base model answers some prompts directly and continues others; with the chat template it echoes the prompt and writes ordinary tokens where the markers belong, because its `<|im_start|>` and `<|im_end|>` embeddings are identical (cosine 1.0000) while the instruct model's differ (0.567). Every output contains 5363 but none is a program the executor accepts.
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
Concepts first, little maths: the only formulas are `W' = W + (α/r)·B·A` and the parameter count `r·(d_in + d_out)`. No hand-built LoRA layer.
- **The idea.** Opens with one or two sentences placing it among other uses of low-rank factorization (e.g. matrix factorization for recommendation, compressing weights): the same trick of replacing a big matrix by two thin ones, but here the product is the *update* to a frozen weight, not the data. Nothing more is drawn from the analogy. Then a diagram of the thin matrices `B` (d_out × r) and `A` (r × d_in) multiplying into a full-size update. The update to a pretrained weight, not the weight itself, is assumed to be low-rank (tested in §3.3 with the singular values of the trained update). In prose: why `B` starts at zero, what `α/r` does to the scale, rank as capacity.
- **What it costs.** Computed from the model config, not hand-built: `r·(d_in + d_out)` per matrix over the seven projections of the 36 layers, which should reproduce the 33,030,144 trainable parameters at r=16 and match PEFT's own count. The table of each matrix's rank ceiling (its smaller dimension: 2,560, and 1,024 for k and v) and break-even rank (731 to 2,027) where LoRA stops being cheaper than full fine-tuning.
- **Two checks on the real model through PEFT.** At step 0 the adapter model's logits equal the base model's exactly (`B` is zero); an adapter merged into the weights (`merge_and_unload`) gives the same outputs as the unmerged one.
- **Short prose:** intuition for rank and alpha (the sweep is in §4.2); QLoRA, mentioned for completeness.

### Part 3 — Experiments

**3.1 Formatting conversations for training** (`301_formatting_conversations.qmd`)
This chapter uses standard multi-turn SFT: one training sequence per **conversation**, with teacher forcing. Background is in 2.3.
- **Messages:**
  - System: "Answer with a program in the ConvFinQA DSL."
  - User turn 1: the context (`pre_text`, the markdown table, `post_text`) followed by question 1.
  - Assistant turn 1: gold `turn_program[0]`.
  - Then user turn k: question k, and assistant turn k: gold `turn_program[k]`, for the rest of the conversation.
- **Loss** is computed on **assistant tokens only**, all turns at once. With causal attention this equals the sum of the losses on each truncated prefix, and so do the gradients; `tests/test_causal_prefix.py` checks both on a tiny Qwen3. Only the end of a conversation can be cut; dropping early tokens changes every later loss. The page is encoded once per conversation, not once per turn, which uses about 3.5× fewer tokens than a per-turn layout (measured on the training split in 2.3: 3,525,034 against 12,333,479).
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
- Uses greedy, batched generation with left padding, with a **fixed batch size and a fixed example order** (sorted by length once), because batch composition changes bf16 logits slightly. Each predicted program is parsed and executed.
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
- Expected time for the main run: **about 4 hours** on GB10 (measured in 2.5: about 41 s per optimizer step of 16 conversations with gradient checkpointing, 342 steps; the first guess of 1–2 hours was wrong). It is built by `./build.sh` (see 1.1).
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
  - Greedy evaluation is deterministic for a fixed batch size and example order, so the generation seed doesn't matter here (see the 2026-10-22 decision-log entry). Only the **training seed** does: LoRA initialization, data order, dropout.
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
- **Merged vs unmerged adapter:** score the adapter merged into bf16 weights against the adapter kept separate (bf16 base, fp32 adapter), on val, and against a merge kept in fp32. Measured in §2.6 on a random adapter: the two are the same function in fp32, but not in bf16 (difference as large as the adapter's own effect), so a merged model is checked before use.
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
     - xgrammar was checked on the GB10 (2026-10-26; `constrained.DSL_GRAMMAR`, tests in `tests/test_constrained.py`): the DSL grammar accepts all 8,981 gold programs of train and dev, and Qwen3-4B constrained answers always followed it. In a 16-turn check, zero-shot went from 5 to 6 correct and from 7 to 16 syntactically valid answers; two-shot answers were identical with and without the grammar (15 of 16 correct); compiling takes 0.4 s and generation time was unchanged.
     - A grammar guarantees syntax, not termination: with an unbounded number rule an untuned zero-shot model ran on in digits (`2.7000...0`) up to the token limit. Gold numbers have at most 10 integer and 4 decimal digits and programs at most 120 characters, so a **bounded** variant (digit-count limits) is an option for this level; choose between the two on val.
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
  - model size (Qwen3.5-4B/9B);
  - a **LoRA rank sweep**, below and above the main run's r=16 (e.g. r = 8, 16, 32, 64, 128, 256, with alpha = 2r), on the same data, scored on val with the §3.4 evaluation. It shows how much capacity the task needs; full fine-tuning is the high-capacity end (the rank ceiling of Qwen3-4B's matrices is 2,560, and LoRA has no fewer parameters than full fine-tuning above ranks of about 730 to 2,030 depending on the matrix; see §2.6);
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

## Open topic: beyond the single page (literature notes, for later discussion)
Not in scope for the current chapters; saved from a 2026-10-26 discussion so it can be picked up later (a possible §4.x, or a "Further reading" callout). **Verification status:** items marked (search) come from web-search summaries and, for LOFin, a read of the paper's HTML (arXiv 2505.20368v3, the last ~13k characters unread); items marked (memory) are recalled, not checked. Check every title, year and number before it goes into `references.bib`.

**The problem.** ConvFinQA hands the model the one right page (text plus table). The harder, realistic task: the question names no document and the system must find the evidence in a large corpus, then reason over it. Three settings, in growing difficulty:
1. *Open-domain, few documents needed:* a corpus of hundreds to thousands of filings; each question needs one to a few of them; the model is not told which.
2. *Few-document multi-doc:* the documents are given or pre-selected (SEC-QA, MultiHiertt, TAT-DQA); not the target.
3. *Corpus-level aggregation:* the answer needs records from hundreds or thousands of documents, so top-k retrieval cannot work.

**Our interest:** setting 1, in particular telling apart many similar-looking tables across many filings. Setting 3 is noted for completeness.

**Benchmarks**
- *Open-domain, setting 1.*
  - **LOFin** (arXiv 2505.20368; search + paper): 145,897 reports from 516 companies; LOFin-1.4k / 1.6k test questions built from FinanceBench, FinQA and SEC-QA templates. Gold evidence is page-level (`GIS_2020_10K (p.51)`).
  - **FinanceBench** (Islam et al. 2023; search): ~150 open-book questions over ~360 filings, with a "shared vector store" setting (all filings in one index). GPT-4-Turbo: 50% correct with a per-document store, 19% shared (68% refused). A later pipeline (arXiv 2411.07142) got 47% shared with query rewriting, filename prefixes and GPT-4o, and 55% with finance-adapted (BAM) embeddings; the two pipelines differ, so the numbers are not comparable. One secondary source says 10,231 triplets, which conflicts with the original scale; check the release.
  - Seen in search results but not read: FinRCA-Bench (arXiv 2608.18534), LEDGER (arXiv 2606.13100), Fin-RATE (arXiv 2602.07294), FinSAgent (arXiv 2607.18102).
- *Few-document multi-doc, setting 2:* SEC-QA (Lai et al. 2024, arXiv 2406.14394; a question-generation framework over SEC filings, no fixed size), MultiHiertt, TAT-DQA, DocFinQA (long-context, full filing; memory).
- *Corpus-level aggregation, setting 3 (search):* MuDABench (arXiv 2604.22239; >80k pages, 332 questions, ~15 documents per question), CorpusQA (arXiv 2601.14952; hundreds to thousands of documents), AGGBench (arXiv 2602.01355; entity-level aggregation, an agentic disambiguate/filter/aggregate baseline), Structured RAG, S-RAG (arXiv 2511.08505; build a schema and a record per document at ingestion, then query the records). Text-to-SQL-over-documents systems (Evaporate, DocETL, Aryn) were not found by the search; unchecked.
- *Document-image retrieval (memory):* ColPali/ViDoRe, M3DocRAG, VisRAG, MP-DocVQA, DUDE, MMLongBench-Doc.
- *Open-domain table QA (memory):* OTT-QA, dense table retrieval (DTR).
- *Fine-tuning for retrieval-augmented reading (memory):* RAFT (train with the gold document plus distractors).

**LOFin results** (paper's Table 2, LOFin-1.4k; page recall / answer accuracy, %; ◊ = GPT-4o generator)

| Method | Avg recall | Avg acc. | Numeric-from-table acc. |
|---|---|---|---|
| GPT-4o zero-shot (no retrieval) | – | 13.9 | 3.8 |
| Perplexity | – | 10.6 | 2.5 |
| Self-RAG | 19.0 | 7.6 | 1.9 |
| RQ-RAG | 18.5 | 8.3 | 2.0 |
| IRCoT ◊ | 25.2 | 22.3 | 19.1 |
| HybridSearch ◊ | 24.4 | 25.6 | 19.1 |
| HHR ◊ | 33.3 | 28.7 | 26.5 |
| Dense ◊ | 34.8 | 29.2 | 23.7 |
| HiREC ◊ (paper's method) | 45.4 | 42.4 | 37.2 |
| Gold evidence ◊ | 100 | 65.9 | 65.0 |

- Generator size (HiREC retrieval): Qwen-2.5-7B 32.3, DeepSeek-14B 38.8, GPT-4o 42.4. Retrieval matters more than the generator (Dense + GPT-4o: 29.2).
- By source (accuracy, HiREC / Dense): FinanceBench 50.0 / 33.3, FinQA 41.6 / 27.6, SEC-QA (multi-document) 14.2 / 9.5. Every method is lowest on SEC-QA.
- Commercial search (40 questions per category): Perplexity 15.0, SearchGPT 27.5, HiREC 46.7 average; they find documents but often miss the exact numbers.
- No results for newer frontier models or agentic systems were found; a search for later LOFin papers is open.

**Approaches seen**
- *HiREC* (LOFin paper): hierarchical retrieval (document, then page), then evidence curation (filter, and generate follow-up queries when evidence is missing); the curation model is a fine-tuned 7B. Ablations (accuracy): full 42.4; without hierarchical retrieval 32.8; without curation 36.7; without fine-tuning 40.1; without the filter 42.1 (best recall, 50.2, but conflicting passages reach the generator).
- Metadata-aware retrieval (FinanceBench follow-up): query rewriting, filename prefixes, finance-tuned embeddings.
- Agentic iteration (IRCoT, Self-RAG, RQ-RAG): did worse than plain dense retrieval on LOFin.
- Corpus-level: extract structured records per document at ingestion, then answer by querying them (S-RAG and similar).
- RAFT-style training for the reader (memory).

**Why even gold evidence scores only ~66% on LOFin.** The paper has no error analysis of the gold-evidence runs. What it shows: gold evidence is whole pages (~1.3 per question) extracted as text with PyMuPDF (table serialization not described); numeric answers are checked with an unspecified rounding tolerance; textual answers are judged by GPT-4o; FinQA-derived gold pages came from a BM25 + NLI pipeline with manual resolution, and OCR errors merged some pages. Appendix F failures: wrong arithmetic from rounded inputs (General Mills FY2020 free cash flow: 3,239 vs 3,215), and a three-year question where only one filing was retrieved. Our guess, untested: multi-step programs fail on one slip in cell choice, units or sign. ConvFinQA's clean tables plus an executed DSL remove the parsing and arithmetic failures, so accuracy there would not transfer to raw pages without a table-extraction step.

**Ideas for this lab** (suggestions, not decisions)
- Add a retriever in front of the §3 reader and compare gold-page vs retrieved-page accuracy, which separates reader errors from retrieval errors.
- Use a pool of ConvFinQA source pages as a controlled corpus: they come from the same companies' reports, so near-duplicate tables are built in.
- Fine-tune the retrieval-side step with LoRA (the HiREC curation model is a fine-tuned 7B), or use RAFT-style distractor training for the reader.

**Open questions for the discussion**
- Which of settings 1 and 3 do we want, and is a small controlled corpus enough?
- How should tables be serialized for the reader on raw pages?
- Read the missing parts: LOFin Appendix D.4 (generation prompt) and the unread tail of the HTML; FinRCA-Bench and LEDGER setups; later LOFin results with newer models.

## Progress tracker
Mark each item `[x]` when it's done and verified. Add a short note (run id, headline number) where useful.

- [x] 1.1 Environment: uv project, Quarto 1.10.19, kernel, git + private GitHub repo `yinchi/fine-tuning-lab`, `PLAN.md`/`CLAUDE.md` committed. Chapter renders: GB10, CUDA 13.2 everywhere, bf16 matmul OK, torch 2.14.1+cu132.
- [x] 1.2 Data downloaded and explored; `max_length` chosen. Train 3,037 convs / 11,104 turns, dev 421 / 1,490; test has no labels. Longest full conversation 3,460 tokens, so `max_length` = 3584 (saved to `artifacts/processed/lengths.json`; tied to the Qwen3 tokenizer).
- [x] 1.3 Executor; 100% gold match. `executor.py` reproduces all 11,104 train and 1,490 dev gold answers exactly (0 mismatches, 45 tests pass); errors reported by kind.
- [x] Artifact system (`src/convfinqa/artifacts/`: 14 modules plus `__main__`, one test file per concern). Builders are declared in chapter cells and found by scanning the `.qmd` files; recursive freshness with early cutoff; heavy builds in subprocesses; `ALLOW_STALE` previews; chapter records and `refresh`. Every module was reviewed against its behaviour (see the decision log, 2026-10-23 to 2026-10-25). 454 tests pass. Verified in real Quarto renders with a scratch chapter (missing, build, stale, `ALLOW_STALE`, refresh, rebuild) and the pre-render hook. `./build.sh` rewritten, not yet run end to end because no artifacts or targets exist yet.
- [x] 2.1 From text to tokens to next-token probabilities. Qwen3-4B-Instruct-2507: 151,669-token vocabulary, digits are single tokens; the trace on the AAPL turn-1 prompt (1,125 tokens) matches `model(...).logits` exactly; untuned top token is a code fence (p=0.614), gold first token `5` has p=0.024 (rank 5); greedy is repeatable, same-seed sampling is repeatable; a digits-only mask makes greedy output `5363`.
- [x] 2.2 Anatomy of Qwen3-4B. Every block recomputed by hand from the model's weights and checked against hooked activations of layer 0 (relative differences 0.3-0.5%, bf16 rounding); all 36 layers by hand give the same top-5 tokens as the model; the causal-mask check leaves the first 925 of 1,125 positions unchanged (0.0 difference). 4,022,468,096 parameters (embedding 9.7%, feed-forward 66.9%, attention 23.5%); KV cache 147,456 bytes/token (0.53 GB at max_length 3,584); LoRA `all-linear` selects 7 projections per layer (252 modules), not the output head.
- [x] 2.3 From base model to instruct model. Summary of pretraining and post-training from the Qwen3 report, with a redrawn Figure 1; base vs instruct on four prompts (base answers some, continues others; chat markers untrained in base); chat template vs the original thinking template; one sequence per conversation uses 3.50× fewer tokens than per-turn (3,525,034 vs 12,333,479).
- [x] 2.4 Probing the untuned model. 20 val conversations (68 turns, gold history, greedy): zero-shot 24% exec. acc. (44 of 52 errors are format failures), 2-shot 68% (7 format, 15 valid-but-wrong); 13 of 51 invalid outputs contain the gold answer under a lenient reading (ad hoc scan, not in the chapter). Most surprisal is in DSL tokens (opening operator, `),`, `(#`, end of turn), not copied numbers. **Val gold-program NLL = 2.2062 nats/token** (306 convs, 16,417 tokens; artifact `val_gold_nll`), the predicted step-0 loss for §3.3. Attention-map probe skipped (optional).
- [x] 2.5 How fine-tuning works. By-hand loss on a 4B val conversation = `model(labels=...).loss` (1.889931 both) and = 2.4's `gold_logprobs`; AdamW and cosine-with-warmup written out (schedule checked against the library at every step); accumulation normalization quantified; memory budget (computed weights match measured); `step_cost_*` measurements (4B LoRA r=16 needs gradient checkpointing: 21 GB typical / 34 GB longest batch, ~41 s/step, ~4 h for 342 steps); `sft_demo` (0.5B full fine-tuning via `SFTTrainer`, 40 steps): Trainer's step-0 loss = hand-computed (1.4589), labels = ours, logged lr = our schedule, 3 -> 19 of 24 turns correct (20 in the first run; GPU training is not bit-reproducible, so the chapter computes its numbers from the artifact). No hand-written loop (see decision log).
- [x] 2.6 LoRA in detail. Parameter count at r=16 computed from the config = PEFT's (33,030,144; 252 modules; 0.82% of the model; about 132 MB in fp32); step-0 logits identical to the base model's (max difference 0.0, bf16 and fp32); merging a random adapter is exact in fp32 (0.002 on an effect of 14.7) but not in bf16 (15.3, with 4% of top tokens changed), so a merged bf16 model is checked before use (§3.4).
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
- 2026-10-22: Evaluations fix the batch size and the order of examples. Measured on Qwen3-4B-Instruct-2507 with two prompts (1,348 and 962 tokens) run alone and in a batch of two: the logits differed by up to about 0.9 (logits range from about -20 to 27), the top token was the same, and nothing was bit-identical; the unpadded prompt differed as much as the padded one, so the batch size, not the padding, is the main cause. Greedy decoding is therefore deterministic only for a fixed batch size and order. The default `padding_side` of this tokenizer is `right`; generation uses left padding.
- 2026-10-22: Artifact system built as planned in the 2026-10-21 entries, with these changes. (1) `./build.sh` builds the artifact DAG first (`build --all`, topological, heavy nodes in subprocesses), then renders **without** `RUN_HEAVY`, then previews with `ALLOW_STALE=1`; `RUN_HEAVY=1 quarto render` stays as a fallback. (2) A stale or missing heavy artifact raises by default; `ALLOW_STALE=1` loads a stale-but-present one with a visible warning (a missing one always raises), and a chapter that used a stale result is always re-executed by `refresh`. (3) Builders take only `out`, `deps` (artifacts), `files` (static files) and their own name fields: no `**extra` parameters, so config lives in declared files or light config artifacts. (4) Freshness adds a `code_hash`: the normalized AST source (`ast.unparse`, docstrings and local annotations dropped) of the builder's module and its `convfinqa.*` import closure, plus modules loaded during the build; `version` stays a manual override; library or Python-version changes warn only. (5) Seeds are part of the config; the framework seeds every build and light builds restore the global RNG state. (6) `src/convfinqa/targets.py` lists the artifact names `build --all` builds. (7) Build directories are `artifacts/<name>.tmp/` (deterministic, kept after a failure for the log) under a per-artifact lock in `artifacts/.locks/`.
- 2026-10-22: Package restructured into layers: `dsl/` (executor, program-text helpers), `data/` (raw loading, text rendering; `data/__init__` re-exports the common names so `data.load_split` still works), `book/` (viz, diagrams). Layers import downward only and the pure ones never import torch (`tests/test_layers.py`). `formatting.py` and `constrained.py` stay flat until `train.py` arrives, then move into `llm/`. `artifacts.py` stays one file. Moving code changes no artifact: none had been built yet, and the code hash covers module contents, not paths.
- 2026-10-23: `artifacts.py` split into the package `artifacts/` (registry, handles, errors, hashing, seeding, cli), done before any artifact existed, so no `node.json` was invalidated. The `__main__` delegation workaround is gone: `python -m convfinqa.artifacts` runs `artifacts/__main__.py` inside the imported package, so builders and the CLI share one registry. The code-hash exclusion is now a prefix check, `hashing.is_framework` (`convfinqa.artifacts` and `convfinqa.artifacts.*`, plus `convfinqa.targets`). Internals the tests use live in the submodules (`hashing`, `cli`); the package exports only the public API.
- 2026-10-23: Builders are declared in the chapter code cell that uses them, and found by scanning the `.qmd` files with `ast` rather than by importing modules. This keeps "load this artifact or run this code to create it" in one cell while `./build.sh` can still build the whole DAG before rendering. Consequences: builders defined in a chapter must be self-contained (checked at scan time); targets are derived from `get("literal")` calls, so `targets.py` is gone; the code hash of a chapter builder covers its own function plus the `convfinqa` modules it imports; `builders.py` remains for shared builders. Names are letters, digits and underscores only, with `str` (letters and digits) and `int` (canonical) fields, so name matching is unambiguous and cannot escape the artifacts directory; this supersedes the dashed example names in earlier entries (`sft-dsl-0` is now `sft_dsl_0`).
- 2026-10-23: Quarto keeps each chapter's Jupyter kernel alive for 5 minutes after a successful render and reuses it (found in a scratch chapter: `ALLOW_STALE` was `None` in the reused kernel, and the registry still held the old builder source). `execute: daemon: false` in `_quarto.yml` has no effect in Quarto 1.10.19 (a kernel stayed alive); the CLI flag `--no-execute-daemon` works for both `render` and `preview`, so `build.sh` and the documented commands pass it. As a second defence the registry compares the `.qmd` files' (name, mtime, size) on every lookup and re-scans when they change, so a long-lived kernel never serves a stale builder; environment variables cannot be fixed that way, hence the flag. Also verified end to end through real Quarto with a scratch chapter (since removed): a missing heavy artifact fails the render with the build command; `build --all` builds it in a subprocess; the render then succeeds and records its use; after a builder edit the artifact is stale, a plain render fails, `ALLOW_STALE=1` renders with a red warning box and records the stale use, and `refresh` marks the chapter for re-execution.
- 2026-10-23: `Registry` now composes parts, as `records` already did: `Catalog` (what exists: builders, loading, name resolution, graph validation; no disk access), `Store` (directory layout, `node.json`, atomic commit, lock), `Policy` (`RUN_HEAVY` and `ALLOW_STALE`, read from the environment when a `Policy` is made; `get` takes an optional policy, so tests need no `setenv`), and the `provenance` functions (git commit and dirty flag, library and Python versions, drift notes). `Spec` gained `build_seed` and `config`. The modules stay flat in `artifacts/`, not subpackages; `tests/test_layers.py` enforces which module may import which. A top-level `convfinqa.registry` package was rejected: after the split the catalog is the registry in the usual sense, and `artifacts` is the public name.
- 2026-10-25: Chapter scan rules fixed. Cells whose leading options say `#| eval: false` are ignored entirely (Quarto does not run them, so neither do we register their builders or count their `get` calls). An `@artifact` function that is not a plain top-level `def` of its cell (nested in `if`/`with`/loop/function/class, or `async def`) is an error with `file:line`; `get("name")` calls may be nested anywhere. `get` calls are reported in source order, so "first" in `target_locations` means earliest line. Not yet handled: fences nested inside a longer fence (an example cell shown inside a four-backtick block would be scanned as real).
- 2026-10-25: The chapter scan now follows CommonMark fences. A block opened by backticks or tildes ends at a fence of the same character and at least the same length, and everything inside it is text; only a top-level backtick fence with info string `{python}` (attributes allowed, up to three spaces of indentation) is a cell. A `{python}` cell shown inside a four-backtick or `~~~` block, as an example for the reader, is therefore not registered. This closes the "not yet handled" case in the previous entry. Checked against the book's 84 real cells in 8 chapters: identical to the old detection.
- 2026-10-25: Invalidation rules tightened, after probing what goes stale (`tests/test_invalidation.py`). (1) Every builder, module or chapter, is hashed by its own function plus the `convfinqa` modules it imports, and must be self-contained (parameters, builtins, imports inside the body; absolute imports only). Before, a module builder was hashed by its whole module's import closure, so editing one builder or any module-level code in `builders.py` staled all the others. `Spec.source` is now always set and `Spec.module` only identifies the builder. (2) The annotation normalizer now keeps class- and module-level annotations (they define dataclass fields), and strips only those inside functions. (3) `get` first rebuilds, in this process, the stale or missing light artifacts its target is built from, so a behaviour-preserving edit to a light input no longer makes a heavy artifact look stale (early cutoff on the `get` path, not only in `ensure`). It never starts a heavy build. (4) Under `ALLOW_STALE`, a light artifact whose rebuild is blocked by a stale heavy input loads its old result with the warning (it must exist), so previews keep working after a code change; `RUN_HEAVY` still wins when both are set. (5) `stale_message` became `refusal_message`.
- 2026-10-25: Provenance recorded in `node.json` now tracks the whole pipeline's libraries (torch, transformers, tokenizers, datasets, accelerate, peft, trl, xgrammar, sympy, numpy; reporting-only libraries such as tensorboard, pandas and matplotlib are left out). A difference from the build environment only produces a warning, never staleness; the lab is not expected to upgrade libraries except when rebuilt from scratch, when the warnings show exactly what moved. Git calls use `--no-optional-locks` and a 10-second timeout, and an unanswerable question is recorded as `None`, not as "clean".
- 2026-10-25: `preserved_rng_state` now restores what it promised. A library first imported inside the block (the build's own `seed_everything` imports NumPy and torch) used to keep the build seed afterwards, so in a kernel that had not used them yet a light build left the chapter's global generators seeded; they are now reseeded from entropy on exit, CUDA included. The private `chdir` helper is gone in favour of `contextlib.chdir` (Python 3.11+). The docs now say what seeding does not cover: generators made with `default_rng()` or `torch.Generator()`, `PYTHONHASHSEED` (set through the build subprocess's environment), GPU atomics, and DataLoader workers.
- 2026-10-25: The store is hardened. A `node.json` that cannot be trusted (unreadable, not text, not JSON, not a node) counts as "not built" and is rebuilt, instead of escaping as a raw exception or being used. Names must be plain file names, checked before anything touches the disk, because `scratch` deletes directories. `commit` clears a leftover `name.old` first (it used to fail at the swap and lose a finished build) and rolls back to the previous version if moving the new one into place fails. A crash in the instant between the two renames still leaves the artifact missing, and the next build recovers; this is documented and tested, as is that a killed build leaves no stale lock.
- 2026-10-25: Build subprocesses fail more usefully and start for every seed. `PYTHONHASHSEED` is the seed modulo 2**32 (Python refuses larger values, so an artifact named with a seed of 2**32 or more could not start its build at all). A missing `PYTHONPATH` no longer leaves a trailing empty entry that put the current directory on the child's import path. The child's stderr is captured until it has opened its log, so an early failure (an import error, a bad chapter) shows its cause in the `BuildError` instead of pointing at a log that was never created, and a child killed by a signal is reported as such (`was killed by SIGKILL (often the out-of-memory killer)`), not as exit code -9.
- 2026-10-25: The command line is stricter. `build` with names and `--all` together is an error (it used to ignore the names and report success). `status --check` exits 1 unless everything shown is fresh, for scripts. A registry with no builder modules tells its build subprocess so (`--import=`), where before the child silently fell back to the default modules and could register different builders than its parent. The hidden `_run` command no longer shows in the usage line.
- 2026-10-25: Chapter 2.4 needs the val split before §3.1 exists, so `data/splits.py` (`page_group`, `make_val_split`: 10% of pages, seed 0 -> 2,731 train-fit / 306 val conversations, 9,979 / 1,125 turns) and the label mask (`formatting.tokenize_conversation`, labels only on assistant content and `<|im_end|>`) were built there; §3.1 presents them. The split is the light shared artifact `train_val_split` (`builders.py`: `train_fit.jsonl`, `val.jsonl`). The model runs of 2.4 are heavy artifacts, `probe_val_sample` and `val_gold_nll`, even though nothing else reuses the first: under `freeze: auto` any prose edit re-executes a chapter, so slow model cells belong in artifacts. Dev stays the final test set; the official test split is unlabeled and unused.
- 2026-10-25: Gradient checkpointing stays on for the main run, and its duration is revised. Measured in chapter 2.5 (`step_cost_*` artifacts: micro-batch 4 × accumulation 4, `max_length` 3,584, 5 timed steps, process capped at 80 GB): Qwen3-4B with LoRA r=16 does **not** fit without checkpointing (over 80 GB on typical steps), and with it peaks at 21 GB (34 GB on the four longest conversations) at about 41 s per optimizer step, i.e. about 4 hours for 342 steps (the earlier 1–2 hour guess is superseded). On Qwen2.5-0.5B with full fine-tuning checkpointing saves 38–43% of peak memory for 18% more time. Precision: weights bf16 everywhere; the 0.5B full-fine-tuning runs keep gradients and AdamW state in bf16 (no fp32 master weights); PEFT keeps LoRA adapters in fp32. Memory measurements run under a per-process cap (`memory_cap_gb`), because the GB10's GPU shares its memory with the desktop and an uncapped 4B run without checkpointing froze the machine. Not measured: 4B without checkpointing at a smaller micro-batch (a §3.3 tuning question).
- 2026-10-26: LoRA stays the method, with r=16 and alpha=32 for the main run as planned; no full fine-tuning of the 4B model. Why: the task is narrow and the data small (about 10k turns), and in bf16 full fine-tuning would fit (estimated 56 GB for the longest batch, not measured) but with fp32 weights it would not (about 86 GB, over the 80 GB cap, on a GPU that shares memory with the desktop). The rank question is answered by a sweep in §4.2 that goes both below and above 16 (e.g. 8 to 256), not by changing the main run.
- 2026-10-26: Chapter 2.5 shows `SFTTrainer` with four checks (step count, labels, learning-rate schedule, step-0 loss against our own by-hand loss) instead of a hand-written training loop: the pieces are already explained in "From the loss to a weight change", and implementing the loop adds tensor calculus that obscures the concepts. This replaces the "about 30 lines of PyTorch" item of §2.5. The same four checks carry over to §3.3. Findings that carry over too: `SFTTrainer` uses a pre-tokenized `labels` column as it is (so §3.3 can pass `formatting.tokenize_conversation`'s labels and does not need `assistant_only_loss` or template markers); with this transformers version warmup is `warmup_steps=0.03` (a float below 1 is a fraction of all steps, rounded up), and the Trainer logs at step s the rate used for that step's update; with accumulation the Trainer sums the token losses and divides by the whole accumulated batch's token count (`num_items_in_batch`). The demo trains with fp32 weights and bf16 autocast, because pure-bf16 weights can lose updates much smaller than their rounding step at small learning rates.
- 2026-10-26: Chapter records are written through a temp file of their own (`tempfile.mkstemp`) instead of a fixed `<chapter>.json.part`. A `quarto preview` that re-renders on save and a command-line render of the same chapter ran at once and one renamed the other's `.part` file away (`FileNotFoundError`). A test runs four writer threads on one record.
- 2026-10-26: Chapter titles became `# Title {#sec-name}` headings (labels: `sec-environment`, `sec-convfinqa`, `sec-dsl`, `sec-tokens`, `sec-anatomy`, `sec-base-instruct`, `sec-probing`, `sec-fine-tuning`), and the prose of chapters 2.4 and 2.5 and the relative references of the earlier chapters ("the previous chapter") now use `@sec-` references, because the book's rendered chapter numbers (204 is Chapter 7) differ from the plan's. Also fixed: the 2.3 chapter said the formatting chapter follows it (the probing chapter does), and 2.5 mentioned a hand-written loop that was dropped. Changing every title re-executes every chapter once (freeze watches the `.qmd` source).
- 2026-10-26: Chapter 2.6 explains LoRA conceptually, with no hand-built LoRA layer (the plan's "single `nn.Linear` wrapped by hand, then by PEFT" item is dropped). Same reasoning as for the hand-written training loop in 2.5: re-implementing the formula teaches tensor shapes more than concepts. It keeps the idea (a diagram and prose), a computed parameter count that must match PEFT's 33,030,144, the rank-ceiling and break-even table, and two checks on the real model through PEFT: step-0 logits equal the base model's, and a merged adapter equals the unmerged one.
- 2026-10-26: Open-domain, multi-document financial QA (retrieval over a large corpus, then reasoning) is out of scope for the current chapters. A literature survey (LOFin, FinanceBench, SEC-QA, corpus-level aggregation benchmarks, results and approaches) is saved in "Open topic: beyond the single page" for a later discussion; nothing in the chapters or the progress tracker changes.
- 2026-10-26: xgrammar verified on the GB10 (Python 3.14, aarch64, CUDA 13.2) with the DSL grammar and the HF logits processor (`constrained.compile_grammar`, `grammar_accepts`, `generate_constrained`). The grammar accepts every gold program; constrained Qwen3-4B answers always follow it; no measurable generation-time overhead at this scale; two-shot answers are unchanged by the constraint, zero-shot ones become valid programs. The token vocabulary size passed to xgrammar must be the model's output size (151,936), not the tokenizer's (151,669), and the stop token is `<|im_end|>`. Whether to bound number lengths in the grammar is left to §3.6, decided on val.

## Verification
- `uv run pytest`: the executor reproduces the gold `exe_ans_list` values, and the formatting and authoring tests pass.
  - The formatting test decodes the tokens where `labels != -100` and asserts they are exactly the assistant programs, plus the end-of-turn tokens.
  - The Python translation round trip: every gold `turn_program` in train and dev, once translated and run by the safe evaluator, equals the DSL executor's result.
  - The safe evaluator rejects `__import__('os')`, attribute access, calls, and loops.
  - Grammars: every gold train and dev program, in both formats, is accepted by its syntax grammar.
  - A short constrained generation through xgrammar's HF logits processor works on GB10 (done, 2026-10-26: `tests/test_constrained.py`, plus a 16-turn check on Qwen3-4B).
  - `stats.py` checks:
    - identical systems give a difference CI centered on 0;
    - a planted +5-point difference on synthetic data is detected;
    - the cluster bootstrap gives a wider CI than a turn-level bootstrap when outcomes are correlated within a conversation.
  - Split check: no page group appears in both train-fit and val.
  - Causal-prefix equivalence (`test_causal_prefix.py`): on a tiny Qwen3, the summed loss and gradient of a whole conversation equal the sums over its truncated prefixes, and cutting the start does change the loss.
- Foundations checks, written as asserts in the chapters so a failing check stops the render:
  - **2.5:** the hand-computed cross-entropy equals `model(labels=...).loss` to within 1e-4.
  - **2.6:** at step 0, the LoRA-wrapped model's logits equal the base model's exactly; the trainable-parameter count matches PEFT's.
  - **3.3:** the step-0 val loss is within a few percent of 2.4's gold-program negative log-likelihood (the formatting is identical).
- `RUN_HEAVY=1 quarto render docs/303_training_with.qmd` (smoke config first) finishes, writes `artifacts/runs/<name>/node.json`, and shows falling loss curves.
- After everything has run once, a full `quarto render docs/` finishes in minutes. Unchanged chapters come from `_freeze/`.
- Editing prose in a heavy chapter and rendering it re-executes the chapter, and the heavy cells load their artifacts and show the edit. With an artifact removed, the same render fails with the error naming the build command, and starts no training. `RUN_HEAVY=1` builds it in a subprocess.
- Artifact tests (`test_artifacts.py`, also covering `ALLOW_STALE`, the builder signature rule, code-hash normalization and seeding): fresh; own config changed; a dependency rebuilt; a stale middle node with a fresh-looking leaf (recursive check); missing dependency; early cutoff on identical output; `RUN_HEAVY` set to `1`, `0` and unset; an interrupted build leaves no `node.json`; the graph has no missing deps or cycles; `refresh` deletes exactly the stale chapters' freeze folders.
- In the book, §1.2 shows real conversations, and §3.4 shows fine-tuned execution accuracy clearly above the §3.2 baseline.
- The rendered book shows chapter codes matching the filenames, and has working cross-references, glossary links, and a bibliography. Part 2 renders in a few minutes from a clean freeze.
