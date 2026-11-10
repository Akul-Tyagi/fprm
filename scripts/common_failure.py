"""Sudoku audit: per-puzzle solved flags and duplicate-constraint violations for every (arm, seed), then
  (1) re-derive Table 3 numbers, (2) VALIDATE against the paper's audit solved counts, (3) compute violations on puzzles that
  ALL THREE arms fail (reviewer 4Uap Q7 / paper Limitations).

Single GPU, batches of 32 (same as the original audit).  Run:
  python scripts/common_failure.py run  --ckpt-root checkpoints --data data/sudoku-extreme-1k-aug1k --out results/sudoku_audit
  python scripts/common_failure.py summarize --out results/sudoku_audit
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = {   # arm -> (run-name prefix, Hydra overrides that reproduce the training-time solver)
    "hardfreeze": ("sudoku-hardfreeze", ["arch.token_wise_damping=true", "arch.refresh_interval=0", "arch.stepsize_decay_train=0.0"]),
    "baseline":   ("sudoku-baseline",   []),
    "atwd":       ("sudoku-atwd-r20",   ["arch.token_wise_damping=true", "arch.refresh_interval=20"]),
}
PAPER_SOLVED = {"hardfreeze": [153, 178, 145], "baseline": [215, 218, 213], "atwd": [215, 205, 176]}   # appendix, audit solved counts (seeds 0/1/2)
IGNORE = -100


def violations_one(grid):
    v = 0
    for i in range(9):
        v += 9 - len(set(grid[i, :].tolist())) + 9 - len(set(grid[:, i].tolist()))
    for br in range(3):
        for bc in range(3):
            v += 9 - len(set(grid[br*3:br*3+3, bc*3:bc*3+3].flatten().tolist()))
    return v


def violations(P):                       # P: (N,81) predicted tokens (a constant shift does not change duplicate counts)
    return np.array([violations_one(p.reshape(9, 9)) for p in P])


def predict(config, model):
    import torch
    import eval_utils as U
    from pretrain import autocast_ctx
    loader, _ = U.test_loader(config, 0, 1)
    model.eval(); model.set_num_iters()
    preds_all, labels_all = [], []
    with torch.inference_mode():
        for set_name, batch, _g in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            while True:
                with autocast_ctx(config):
                    carry, _, _loss, _m, preds, all_finish = model(carry=carry, batch=batch, return_keys={"preds"})
                if all_finish:
                    break
            preds_all.append(preds["preds"].cpu().numpy()); labels_all.append(batch["labels"].cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(labels_all)


def run(args):
    import eval_utils as U
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for arm, (prefix, ov) in ARMS.items():
        if arm not in args.arms:
            continue
        for seed in args.seeds:
            f = out / f"{arm}_seed{seed}.npz"
            if f.exists() and not args.force:
                print("skip (exists)", f); continue
            ckpt_dir = Path(args.ckpt_root) / f"{prefix}-seed{seed}"
            ck = U.find_plain_ckpt(ckpt_dir)
            print(f"=== {arm} seed{seed}: {ck}", flush=True)
            config, model = U.build_model("cfg_pretrain_sudoku",
                                          [f"data_paths=[{args.data}]", "global_batch_size=32"] + ov, 0, 1)
            U.load_weights(model, ck)
            P, L = predict(config, model)
            valid = (L != IGNORE).any(1)
            P, L = P[valid], L[valid]
            solved = (P == L).all(1)
            viol = violations(P)
            np.savez(f, solved=solved, viol=viol, ckpt=str(ck))
            print(f"    solved {int(solved.sum())}/{len(solved)}  mean viol {viol.mean():.2f}  if-failed {viol[~solved].mean():.2f}", flush=True)
            del model
            import torch; torch.cuda.empty_cache(); torch._dynamo.reset()   # avoid hitting dynamo's recompile limit across 9 models


def summarize(args):
    out = Path(args.out)
    seeds = args.seeds
    D = {(a, s): np.load(out / f"{a}_seed{s}.npz") for a in ARMS for s in seeds if (out / f"{a}_seed{s}.npz").exists()}
    print("\n== per-run (this audit) vs paper audit solved counts ==")
    ok = True
    for a in ARMS:
        for i, s in enumerate(seeds):
            if (a, s) in D:
                n = int(D[a, s]["solved"].sum()); exp = PAPER_SOLVED[a][s] if s < 3 else None
                flag = "OK" if n == exp else "MISMATCH"
                ok &= (flag == "OK")
                print(f"{a:10s} seed{s}: solved={n:4d}  paper={exp}  {flag}   overall={D[a,s]['viol'].mean():.2f}  if_failed={D[a,s]['viol'][~D[a,s]['solved']].mean():.2f}")
    print("VALIDATION:", "all solved counts reproduce the paper's audit" if ok else
          "MISMATCH -> do NOT quote new numbers until explained (wrong checkpoint step, different data build, or batch size != 32)")
    print("\n== means over seeds (compare Table 3) ==")
    for a in ARMS:
        ov = [D[a, s]["viol"].mean() for s in seeds if (a, s) in D]
        cf = [D[a, s]["viol"][~D[a, s]["solved"]].mean() for s in seeds if (a, s) in D]
        print(f"{a:10s} overall {np.mean(ov):.2f}   if failed {np.mean(cf):.2f}")
    print("\n== COMMON-FAILURE subset: puzzles failed by ALL three arms (same seed) ==")
    rows = {a: [] for a in ARMS}; ns = []
    for s in seeds:
        if all((a, s) in D for a in ARMS):
            fail_all = np.logical_and.reduce([~D[a, s]["solved"] for a in ARMS])
            ns.append(int(fail_all.sum()))
            for a in ARMS:
                rows[a].append(D[a, s]["viol"][fail_all].mean())
            print(f"seed{s}: n_common_fail={int(fail_all.sum()):4d} | " + "  ".join(f"{a}={rows[a][-1]:.2f}" for a in ARMS))
    if ns:
        print("mean over seeds       | " + "  ".join(f"{a}={np.mean(rows[a]):.2f}" for a in ARMS), f"| mean n={np.mean(ns):.0f}")
        hf, ba, at = (np.mean(rows[a]) for a in ("hardfreeze", "baseline", "atwd"))
        print(f"hard-freeze vs baseline: {100*(ba-hf)/ba:.1f}% fewer;  vs ATWD: {100*(at-hf)/at:.1f}% fewer  (common-failure subset)")
    json.dump({"note": "see printed output"}, open(out / "summary_done.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        if name == "run":
            p.add_argument("--ckpt-root", default="checkpoints"); p.add_argument("--data", required=True); p.add_argument("--force", action="store_true")
            p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a)
