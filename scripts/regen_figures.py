"""Regenerate the figures reviewers flagged (readable at final column width, >=8 pt fonts). From the repo root:

  python regen_figures.py violations  figures/fig3_sudoku_violations.png
  python regen_figures.py lengthgen   figures/fig1_A5_length_generalization.png  A5 \
        results/xeval/a5_s1_baseWeights_scalarSolver.json  results/xeval/a5_s1_atwdWeights_atwdSolver.json
  python regen_figures.py evalsteps   figures/fig2_eval_steps_overlay.png  results/wandb/sudoku_eval_steps_history.csv 0

`lengthgen` accepts either cross_eval.py outputs ({"tag", "metrics": {"length_gen": ...}}) or the per-k JSON that pretrain.py writes
for `+metrics_out=...` ({run_name: {"ks", "sequence_accuracy", ...}}).  Legend labels are derived from the file/run names.
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
NICE = {"baseWeights_scalarSolver": "baseline (native)", "atwdWeights_atwdSolver": "ATWD (native)",
        "atwdWeights_scalarSolver": "ATWD weights, scalar solver", "baseWeights_atwdSolver": "baseline weights, ATWD solver"}


def violations(out):
    arms = ["Hard Freeze", "Baseline", "ATWD ($R$=20)"]
    overall, failed = [24.85, 28.98, 29.33], [30.97, 39.65, 39.04]      # Table 3 (offline EMA audit) -> figure and table agree exactly
    x = range(len(arms)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    b1 = ax.bar([i - w / 2 for i in x], overall, w, label="Overall (all test puzzles)")
    b2 = ax.bar([i + w / 2 for i in x], failed, w, label="Conditional on failure")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(arms); ax.set_ylabel("Mean violations per puzzle"); ax.set_ylim(0, 47)
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)


def _curve(path):
    j = json.load(open(path))
    if "metrics" in j:                                   # cross_eval.py output
        tag = j["tag"]; label = next((v for k, v in NICE.items() if tag.endswith(k)), tag)
        return label, j["metrics"]["length_gen"]
    name, c = next(iter(j.items()))                      # pretrain.py +metrics_out output
    return name, c


def lengthgen(out, title, *paths):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for p in paths:
        label, c = _curve(p)
        ax.plot(c["ks"], c["sequence_accuracy"], lw=1.6, label=label)
    ax.set_title(title, fontsize=9); ax.set_xlabel("prefix length $k$"); ax.set_ylabel("sequence accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)


def evalsteps(out, csv, seed):
    import pandas as pd
    d = pd.read_csv(csv); fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for arm, name in (("hardfreeze", "Hard freeze"), ("baseline", "Baseline"), ("atwd-r20", "ATWD ($R$=20)")):
        s = d[d["run"] == f"sudoku-{arm}-seed{seed}"].dropna(subset=["eval/steps"]).sort_values("_step")
        ax.plot(s["_step"], s["eval/steps"], marker="o", ms=3, lw=1.5, label=name)
    ax.axhline(1000, color="gray", ls=":", lw=1); ax.set_xlabel("training step"); ax.set_ylabel("eval/steps (solver iterations)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out, dpi=300); print("saved", out)


if __name__ == "__main__":
    c = sys.argv[1]
    {"violations": lambda: violations(sys.argv[2]), "lengthgen": lambda: lengthgen(sys.argv[2], sys.argv[3], *sys.argv[4:]),
     "evalsteps": lambda: evalsteps(sys.argv[2], sys.argv[3], sys.argv[4])}[c]()
