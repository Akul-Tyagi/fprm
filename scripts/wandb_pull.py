"""Pull every number the reviewers asked for from W&B (no GPU needed; run anywhere with internet + WANDB_API_KEY).

  python scripts/wandb_pull.py core     --entity <wandb-username> --out results/wandb
  python scripts/wandb_pull.py controls --entity <wandb-username> --out results/wandb     # LaTeX rows for the new control runs
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, wandb

PROJECT = "fprm-atwd"
PAPER = {  # final eval/sequence_accuracy printed in the paper (tolerance 0.0015) -> flags any run mixup
    "a5-baseline-seed0": .930, "a5-baseline-seed1": .308, "a5-baseline-seed2": .058,
    "a5-atwd-r20-seed0": .976, "a5-atwd-r20-seed1": .891, "a5-atwd-r20-seed2": .978,
    "s5-baseline-seed0": .010, "s5-baseline-seed1": .229, "s5-atwd-r20-seed0": 0.0, "s5-atwd-r20-seed1": 0.0,
    "sudoku-hardfreeze-seed0": .1900, "sudoku-hardfreeze-seed1": .2200, "sudoku-hardfreeze-seed2": .1838,
    "sudoku-baseline-seed0": .2675, "sudoku-baseline-seed1": .2725, "sudoku-baseline-seed2": .2663,
    "sudoku-atwd-r20-seed0": .2713, "sudoku-atwd-r20-seed1": .2538, "sudoku-atwd-r20-seed2": .2200,
}


def get_runs(entity):
    api = wandb.Api(timeout=120)
    best = {}
    for r in api.runs(f"{entity}/{PROJECT}"):
        step = r.summary.get("_step", 0) or 0
        if r.name not in best or step > (best[r.name].summary.get("_step", 0) or 0):
            best[r.name] = r
    return best


def hist(run, keys):
    rows = list(run.scan_history(keys=keys + ["_step"]))
    return pd.DataFrame(rows, columns=keys + ["_step"]) if not rows else pd.DataFrame(rows).reindex(columns=keys + ["_step"])


def core(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = get_runs(a.entity)
    missing = [n for n in PAPER if n not in runs]
    print("runs not found in W&B:", missing or "none")
    # 0) QA: final eval numbers vs the paper
    qa = []
    for n, exp in PAPER.items():
        if n in runs:
            got = runs[n].summary.get("eval/sequence_accuracy")
            qa.append(dict(run=n, wandb=got, paper=exp, ok=(got is not None and abs(got - exp) <= 0.0015)))
    qa = pd.DataFrame(qa); qa.to_csv(out / "qa_vs_paper.csv", index=False)
    print("\n== QA vs paper (any False = investigate before using other numbers) ==\n", qa.to_string(index=False))

    # 1) A5/S5: eval/steps and accuracy at each eval (epochs 5/10/15[/30]) -> 'still improving at epoch 15?' + 'does ATWD change halting?'
    rows = []
    for n, r in runs.items():
        if n.startswith(("a5-", "s5-")):
            h = hist(r, ["eval/steps", "eval/accuracy", "eval/sequence_accuracy", "epoch"])
            h.insert(0, "run", n); rows.append(h)
    if rows:
        d = pd.concat(rows); d.to_csv(out / "state_tracking_eval_history.csv", index=False)
        print("\n== state tracking eval history (eval/steps = solver iterations at halting) ==\n",
              d.sort_values(["run", "_step"]).to_string(index=False))

    # 2) A5/S5 training: mean train/steps (proxy for early halting in training) and final train accuracy (mean of last 100 logs)
    rows = []
    for n, r in runs.items():
        if n.startswith(("a5-", "s5-")):
            h = hist(r, ["train/steps", "train/accuracy", "train/sequence_accuracy"])
            rows.append(dict(run=n, train_steps_mean=h["train/steps"].mean(), train_steps_min=h["train/steps"].min(),
                             train_acc_last100=h["train/accuracy"].tail(100).mean(), train_seqacc_last100=h["train/sequence_accuracy"].tail(100).mean()))
    pd.DataFrame(rows).to_csv(out / "state_tracking_train_summary.csv", index=False)
    print("\n== state tracking training summary ==\n", pd.DataFrame(rows).to_string(index=False))

    # 3) Sudoku: eval/steps history + first eval point where it reaches the cap, per seed; train/steps volatility numbers
    rows, vol = [], []
    for n, r in runs.items():
        if n.startswith("sudoku-"):
            h = hist(r, ["eval/steps", "epoch"]); h.insert(0, "run", n); rows.append(h)
            hit = h[h["eval/steps"] >= 990]
            t = hist(r, ["train/steps"])["train/steps"].dropna().values
            t = t[int(0.2 * len(t)):]                      # drop first 20% of training
            vol.append(dict(run=n, first_eval_step_at_cap=(int(hit["_step"].min()) if len(hit) else None),
                            train_steps_mean=t.mean(), train_steps_std=t.std(), train_steps_cv=t.std() / t.mean(),
                            mean_abs_step_to_step_change=np.abs(np.diff(t)).mean()))
    if rows:
        pd.concat(rows).to_csv(out / "sudoku_eval_steps_history.csv", index=False)
        v = pd.DataFrame(vol).sort_values("run"); v.to_csv(out / "sudoku_saturation_and_volatility.csv", index=False)
        print("\n== Sudoku: when eval/steps first reaches the 1000 cap, and train/steps volatility (last 80% of training) ==\n", v.to_string(index=False))
    print(f"\nCSV files written to {out}/")


def controls(a):
    runs = get_runs(a.entity)
    names = [n for n in runs if n.startswith(("a5-scalarrefresh", "a5-tokenwise-norefresh", "a5-nodecay", "sudoku-tokenwise-norefresh"))]
    print("LaTeX rows (copy into the new controls table; accuracies are final eval/* from W&B):")
    for n in sorted(names):
        s = runs[n].summary
        print(f"{n} & {s.get('eval/accuracy', float('nan')):.3f} & {s.get('eval/sequence_accuracy', float('nan')):.3f} & {s.get('eval/steps', float('nan')):.0f} \\\\   % state={runs[n].state}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["core", "controls"])
    ap.add_argument("--entity", required=True); ap.add_argument("--out", default="results/wandb")
    a = ap.parse_args()
    core(a) if a.cmd == "core" else controls(a)
