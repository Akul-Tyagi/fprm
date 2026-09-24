# Code for "When Does Per-Token Halting Help? Adaptive Token-Wise Damping for Fixed-Point Reasoning Models in Looped Transformers"

Supplementary code for an ICLR 2027 submission. Author identity is withheld
for double-blind review; a public, attributed release will follow the
decision. This package is a fork of the [FPRM](https://arxiv.org/abs/2606.18206)
codebase (Movahedi et al., 2026) — see [Base codebase](#base-codebase) below —
extended with **Adaptive Token-Wise Damping (ATWD)**, the method introduced in
the paper.

Every command below is exactly the one used to produce a specific number,
table, or figure in the paper; where a section number is given, it refers to
the paper's numbering.

## What ATWD is, in one paragraph

FPRM halts an entire sequence once the *maximum* per-token residual drops
below a threshold, so one slow-converging token keeps every other token
iterating at full cost. ATWD replaces the solver's single scalar damping
step-size with an independent per-token one, so a token that has stopped
improving gets damped on its own; an unconditional periodic refresh (every
`R` iterations) resets every token's damping state regardless of convergence
history, so no token can go more than `R − 1` iterations without a
full-strength update (Proposition 3.1, "Bounded Staleness"). The entire
mechanism is a ~20-line change to one file (`models/fixed_point_reasoning/model_utils.py`)
plus two new config fields — see [What ATWD actually changes](#what-atwd-actually-changes)
below for the exact diff.

## Installation

Requirements: Linux (or WSL2) with an NVIDIA GPU (CUDA 12.x driver) and Python
3.11–3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
cd FPRM_ICLR2027
uv sync
```

`uv sync` creates a project-local `.venv/` and installs everything, including
PyTorch from the CUDA 12.9 wheel index. Run commands either via `uv run <cmd>`
or after `source .venv/bin/activate`, always from the repository root.

Training logs to Weights & Biases by default; run `uv run wandb login` first,
or set `WANDB_MODE=offline` to disable it.

## What ATWD actually changes

Everything is gated behind two new `FPRMConfig` fields, both `False`/`0` by
default so the unmodified baseline path is bit-for-bit unchanged:

```python
token_wise_damping: bool = False   # per-token vs. scalar damping state
refresh_interval: int = 0          # 0 = no periodic refresh
```

The full mechanism lives in `FixedPointOptimizer.reset()` and
`FixedPointOptimizer.step()` in `models/fixed_point_reasoning/model_utils.py`,
and the sequence-level halting reduction in
`models/fixed_point_reasoning/fprm.py` (`FixedPointReasoningModel_ACTV1.forward`,
the `halting_mechanism == 'fixed_point'` branch) was extended to `max`-reduce
over the token axis when `token_wise_damping=True`, so halting behavior is
unchanged when it's `False`. `cont()` in `model_utils.py` needed no change —
`torch.quantile` with no `dim` argument already flattens whatever shape the
residual tensor has.

## Correctness tests

Run before trusting any ATWD result — these are what the Reproducibility
Statement points to (Appendix A.2):

```bash
uv run python tests/test_atwd.py
```

Four checks, all pure-CPU, seconds to run: (1) with `token_wise_damping=False`,
tensor shapes match the original scalar solver exactly; (2) with it `True` but
every token given identical dynamics and identical initial state, the
per-token solver reduces to the scalar solver's trajectory to floating-point
tolerance; (3) step-size resets to its initial value at exactly every `R`-th
iteration and not before; (4) given one token held static (already converged)
and one perpetually reset to a fresh random target (never converges), the
static token retains its full step-size while the non-converging one gets
damped once patience runs out — confirming damping targets stagnation above
threshold, not convergence (Equation 5 / the `adapt` condition in
`model_utils.py`).

## Reproducing the three experimental arms

All three arms per task share every hyperparameter except the two ATWD flags
(and, for Sudoku's hard-freeze ablation, `stepsize_decay_train`) — this is
what makes the baseline/hard-freeze/ATWD comparison controlled. Full
architecture and solver hyperparameters for every task are in the paper's
Table 5 (Appendix A.5 has the complete list); this section only calls out
the three arm-defining overrides.

| Arm | Config overrides |
|---|---|
| **Baseline** | none (the two ATWD fields keep their `False`/`0` defaults) |
| **Hard-freeze** (Sudoku only, the ablation in Section 5.2) | `arch.token_wise_damping=true arch.refresh_interval=0 arch.stepsize_decay_train=0.0` |
| **ATWD** | `arch.token_wise_damping=true arch.refresh_interval=20` |

`refresh_interval=20` is not hand-picked for the reported results — it was
fixed once, from a short single-seed calibration sweep on $A_5$ (Appendix
A.3), and reused unchanged for $S_5$ and Sudoku.

### 1. State tracking ($A_5$, $S_5$ — Section 5.1)

Build the datasets (500,000 training sequences, length 32, evaluated on
prefixes up to 128 — the disclosed, compute-constrained reduction from an
initially planned 2M-sample schedule; see Section 4.1):

```bash
uv run python dataset/build_state_tracking_dataset.py \
    --group A5 --k-train 32 --k-eval-max 128 --train-samples 500000 \
    --output-dir data/state_tracking-A5-train32-eval128-500k

uv run python dataset/build_state_tracking_dataset.py \
    --group S5 --k-train 32 --k-eval-max 128 --train-samples 500000 \
    --output-dir data/state_tracking-S5-train32-eval128-500k
```

Train (across both GPUs; drop `torchrun --standalone --nproc-per-node=2` for a
single-GPU run). $A_5$ uses `epochs=15`; $S_5$ uses `epochs=30` (extended from
an initial 15 once $A_5$'s budget proved insufficient for $S_5$'s later
grokking-style transition — Appendix A.4):

```bash
# A5 baseline, seed 0 (repeat with seed=1, seed=2 for the full 3-seed table)
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name cfg_pretrain_state_tracking_fprm_a5 \
    data_paths=[data/state_tracking-A5-train32-eval128-500k] \
    global_batch_size=832 epochs=15 eval_interval=5 \
    +checkpoint_every_n_steps=1000 seed=0 \
    +project_name=fprm-atwd +run_name=a5-baseline-seed0 \
    +checkpoint_path=checkpoints/a5-baseline-seed0

# A5 ATWD, seed 0
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name cfg_pretrain_state_tracking_fprm_a5 \
    data_paths=[data/state_tracking-A5-train32-eval128-500k] \
    global_batch_size=832 epochs=15 eval_interval=5 \
    +checkpoint_every_n_steps=1000 \
    arch.token_wise_damping=true arch.refresh_interval=20 \
    seed=0 +project_name=fprm-atwd +run_name=a5-atwd-r20-seed0 \
    +checkpoint_path=checkpoints/a5-atwd-r20-seed0

# S5: identical pattern, --config-name cfg_pretrain_state_tracking_fprm_s5,
# epochs=30, data_paths pointing at the S5 dataset, run names s5-baseline-seedN
# / s5-atwd-r20-seedN. S5 was reported over 2 seeds rather than 3 (Section 4.3).
```

`eval_interval` must always evenly divide `epochs` (`pretrain.py` asserts
this) — it also controls how many resumable checkpoints a long run produces,
so don't set it to a large fraction of `epochs`. The per-$k$
length-generalization curve (Figure 2) is computed automatically whenever
`k_eval_min`/`k_eval_max` are set, which both state-tracking configs already
do — no separate eval-only pass is needed.

### 2. Sudoku (Section 5.2–5.4)

Build the dataset (1,000 base puzzles × 1,000 symmetry-preserving
augmentations each, then truncate the evaluation split to a fixed 800
held-out puzzles — Section 4.1):

```bash
uv run python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug1k \
    --subsample-size 1000 --num-aug 1000

uv run python scripts/truncate_sudoku_test.py data/sudoku-extreme-1k-aug1k 800
```

Train all three arms (`epochs=5500 eval_interval=550`, identical across
arms except the overrides in the table above):

```bash
# Hard-freeze, seed 0
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name cfg_pretrain_sudoku \
    data_paths=[data/sudoku-extreme-1k-aug1k] \
    global_batch_size=736 epochs=5500 eval_interval=550 \
    +checkpoint_every_n_steps=1000 \
    arch.token_wise_damping=true arch.refresh_interval=0 arch.stepsize_decay_train=0.0 \
    seed=0 +project_name=fprm-atwd +run_name=sudoku-hardfreeze-seed0 \
    +checkpoint_path=checkpoints/sudoku-hardfreeze-seed0

# ATWD, seed 0
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name cfg_pretrain_sudoku \
    data_paths=[data/sudoku-extreme-1k-aug1k] \
    global_batch_size=736 epochs=5500 eval_interval=550 \
    +checkpoint_every_n_steps=1000 \
    arch.token_wise_damping=true arch.refresh_interval=20 \
    seed=0 +project_name=fprm-atwd +run_name=sudoku-atwd-r20-seed0 \
    +checkpoint_path=checkpoints/sudoku-atwd-r20-seed0

# Baseline, seed 0 (no ATWD flags at all)
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name cfg_pretrain_sudoku \
    data_paths=[data/sudoku-extreme-1k-aug1k] \
    global_batch_size=736 epochs=5500 eval_interval=550 \
    +checkpoint_every_n_steps=1000 \
    seed=0 +project_name=fprm-atwd +run_name=sudoku-baseline-seed0 \
    +checkpoint_path=checkpoints/sudoku-baseline-seed0
```

Repeat with `seed=1`, `seed=2` for the full 3-seed × 3-arm table (Table 2).
All Sudoku evaluation numbers and every downstream analysis use the
EMA-averaged weights specifically (the plain `step_<N>` checkpoint file, not
`step_<N>_train_state.pt`) — see [Analysis and figure scripts](#analysis-and-figure-scripts).

Every `+`-prefixed override (`checkpoint_every_n_steps`, `project_name`,
`run_name`, `checkpoint_path`, `resume_from`) needs the `+`, since these
fields have pydantic defaults but aren't written in the task YAML files;
Hydra's struct check rejects a plain override for a key that isn't already in
the composed config. Fields already in the YAML (`epochs`, `eval_interval`,
`global_batch_size`, `seed`, `data_paths`) don't take a `+`.

### Resuming a run

Every run above checkpoints every `eval_interval` epochs (a full resumable
bundle: model, optimizer state, dataset position, W&B run id) plus a
weights-only backup every 1,000 steps. To continue an interrupted or
epoch-extended run with the same identity (same W&B run, same LR schedule
position):

```bash
uv run torchrun --standalone --nproc-per-node=2 pretrain.py \
    --config-name <same config> data_paths=[<same dataset>] \
    global_batch_size=<same> epochs=<new total> eval_interval=<same> \
    <same arm-defining overrides> \
    seed=<same> +project_name=fprm-atwd +run_name=<same run_name> \
    +checkpoint_path=checkpoints/<same run_name> \
    +resume_from=checkpoints/<same run_name>
```

## Analysis and figure scripts

- **`scripts/analyze_token_convergence.py`** — loads a trained checkpoint,
  strips the `_orig_mod.model.` prefix `torch.compile(ACTLossHead(...))`
  leaves on every saved key, replays the fixed-point loop on real test
  examples, and records the per-token residual/step-size trajectory at every
  iteration. This produced the residual-curve, 97-token convergence, and
  per-cell heatmap figures in the paper (Figures 4, 6, and 7, all relocated
  to Appendix A.5; the sample-size caveats for these replays are in
  Appendix A.6).

  ```bash
  uv run python scripts/analyze_token_convergence.py \
      --checkpoint checkpoints/a5-atwd-r20-seed0 \
      --data-dir data/state_tracking-A5-train32-eval128-500k \
      --config-json debug_arch_config_a5.json \
      --num-examples 8 --out results/a5_atwd_seed0
  ```

  `debug_arch_config_a5.json` / `debug_arch_config_s5.json` (included in this
  repository) are one-time dumps of the resolved architecture config for each
  task, produced by two debug lines already left in place in
  `create_model.py` right after `model_cfg` is built; regenerate them for a
  different task/config by running any training command once.

- **`scripts/check_sudoku_violations.py`** — loads each arm's final,
  EMA-averaged checkpoint (`step_<N>`, not the `_train_state.pt` bundle),
  decodes the predicted grid, and computes the mean row/column/box
  duplicate-violation count reported in Table 3 and Figure 5 ("graceful
  failure"), both overall and conditional on the puzzle not being solved
  exactly. Run `--help` for exact arguments.

- **`scripts/truncate_sudoku_test.py`** — one-time dataset fix: the base
  repo's `--subsample-size` flag only shrinks the *train* split, leaving the
  test split at the full, un-subsampled Sudoku-Extreme size; this script
  truncates the test split's `.npy` arrays and metadata in place to a fixed
  puzzle count. Source is short enough to read directly.

- **`scripts/check_s5_predictions.py`**, **`scripts/final_sanity_checks.py`**,
  **`scripts/generate_paper_figures.py`**, **`scripts/plot_fig7.py`**,
  **`scripts/plot_stepsize_analysis.py`** — supporting analysis/plotting
  utilities behind specific paper figures and mechanistic checks (the
  $S_5$ prediction-diversity/mode-collapse check, the pre-training
  correctness/sanity sweep, and the final figure-assembly and reliability-inversion
  plots respectively). Each takes its inputs from `results/`; run with
  `--help` or read the source directly for exact arguments.

`results/` in this repository contains the `.npz` trajectory arrays and
`.png` figures these scripts already produced, so the paper's figures can be
inspected or reformatted without rerunning any training.

## Repository structure

This is a size-reduced copy prepared for double-blind review — see
`prepare_supplementary.sh` in the accompanying materials for exactly what was
included and why. Not included, and not needed to inspect the method or
re-run any experiment above:

- `checkpoints/` — full model weights for every reported run (~25+ GB);
  regenerate with the training commands above.
- `data/` — generated dataset arrays; regenerate with the dataset commands
  above (a few minutes to a few hours depending on the task).
- `outputs/`, `wandb/` — local experiment-tracking logs, not part of the
  method or its evaluation.

```
.
├── pretrain.py                    # training entrypoint (Hydra)
├── pretrain_config.py             # PretrainConfig schema
├── create_model.py                # model + optimizer construction
├── puzzle_dataset.py              # PuzzleDataset (train/test iteration)
├── length_gen_eval.py             # single-pass length-generalization eval
├── config/                        # Hydra configs (arch/*.yaml, cfg_pretrain_*.yaml)
├── dataset/                       # dataset builders (state-tracking, Sudoku, Maze, ARC)
├── models/
│   ├── fixed_point_reasoning/     # FPRM + ATWD: fprm.py, fprm_config.py, model_utils.py
│   ├── recursive_reasoning/       # HRM/TRM baselines (unused by this paper's experiments)
│   ├── transformer.py, layers.py, losses.py, ...
├── evaluators/                    # ARC evaluator (unused by this paper's experiments)
├── plots/                         # curve-writing utilities used by length_gen_eval.py
├── scripts/                       # analysis + figure-generation scripts (see above)
├── tests/test_atwd.py             # ATWD correctness tests
├── utils/                         # flop_profiler.py, resume.py, functions.py
├── results/                       # analysis outputs (.npz) and figures (.png) already generated
├── debug_arch_config_{a5,s5}.json # resolved architecture configs (for analyze_token_convergence.py)
└── pyproject.toml, uv.lock        # dependency pins
```

## Compute environment

All reported runs used Kaggle's free-tier 2×T4 GPU instances via
`torchrun --standalone --nproc-per-node=2`; local development and smoke tests
used a single Ampere-class consumer GPU. `bfloat16` autocast and
`torch.compile` are used throughout; see Appendix A.11 for full software
versions.

## Base codebase

This repository extends the released code for:

```bibtex
@misc{movahedi2026fixedpointreasonersstableadaptive,
      title={Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers},
      author={Sajad Movahedi and Vera Milovanović and Shlomo Libo Feigin and Alexander Theus and Thomas Hofmann and Valentina Boeva and T. Konstantin Rusch and Antonio Orvieto},
      year={2026},
      eprint={2606.18206},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.18206},
}
```

The puzzle-task infrastructure (`puzzle_dataset.py`, the ARC/Sudoku/Maze
builders, TRM/HRM baselines) originates from
[TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels); the
state-tracking task originates from
[FP-RNN](https://github.com/dr-faustus/fp-rnn). Neither inclusion affects the
double-blind status of this submission — both are third-party works being
built upon and cited, not this submission's own prior work.

## License

See `LICENSE`. (If preparing this for review and the file names you as
copyright holder, remove or redact that line for the review copy — it is not
required for evaluating the method or reproducing the results, and can be
restored at camera-ready.)
