"""
Standalone analysis: loads a trained ATWD checkpoint, replays the fixed-point
loop on a handful of real test examples, and records the per-token residual
and stepsize trajectory at every iteration. No training-loop code is touched.

Run:
  uv run python scripts/analyze_token_convergence.py \
      --checkpoint checkpoints/a5-atwd-r20-seed0 \
      --data-dir data/state_tracking-A5 \
      --config-json debug_arch_config.json \
      --num-examples 8 \
      --out results/a5_atwd_trajectories
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1


def load_model(checkpoint_dir, config_dict, device):
    model = FixedPointReasoningModel_ACTV1(config_dict).to(device).to(torch.bfloat16)
    cands = glob.glob(os.path.join(checkpoint_dir, "step_*_train_state.pt"))
    assert cands, f"No checkpoint bundle found in {checkpoint_dir}"
    latest = max(cands, key=lambda p: int(os.path.basename(p).split("_")[1]))
    bundle = torch.load(latest, map_location=device)
    raw_sd = bundle["model"]
    # create_model.py wraps every model as torch.compile(ACTLossHead(model)),
    # so saved keys look like "_orig_mod.model.inner...." — a bare model here
    # expects unprefixed keys. Without stripping this, strict=False would
    # silently match nothing and you'd analyze a random-init model with no
    # error message at all.
    prefix = "_orig_mod.model."
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw_sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) < 5, f"Too many missing keys after prefix strip — checkpoint format may differ: {missing[:10]}"
    if unexpected:
        print(f"Note: {len(unexpected)} unexpected keys ignored (likely q_head or EMA-only keys): {unexpected[:5]}")
    model.eval()
    return model


def load_batch(data_dir, num_examples, device):
    inputs = np.load(os.path.join(data_dir, "test", "all__inputs.npy"))[:num_examples]
    labels = np.load(os.path.join(data_dir, "test", "all__labels.npy"))[:num_examples]
    puzzle_ids = np.load(os.path.join(data_dir, "test", "all__puzzle_identifiers.npy"))[:num_examples]
    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "puzzle_identifiers": torch.from_numpy(puzzle_ids).to(device),
    }


@torch.no_grad()
def run_and_record(model, batch, max_iter):
    inner = model.inner
    reset_flag = torch.ones(batch["inputs"].shape[0], dtype=torch.bool, device=batch["inputs"].device)
    carry = inner.reset_carry(reset_flag, batch["inputs"], inner.empty_carry(batch["inputs"].shape[0]))

    input_embeddings = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    cos_sin = None
    if hasattr(inner, "rotary_emb"):
        cos, sin = inner.rotary_emb()
        s = input_embeddings.shape[1]
        cos_sin = (cos[:s], sin[:s])
    seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=inner.puzzle_emb_len)

    residue_trace, stepsize_trace = [], []
    z_state = carry.z_L_state
    for i in range(max_iter):
        z_state = inner._z_step(z_state, input_embeddings, carry.dropout_mask, seq_info)
        residue_trace.append(z_state["residues"].detach().cpu().float().numpy().copy())
        stepsize_trace.append(z_state["stepsize"].detach().cpu().float().numpy().copy())
        if float(z_state["stepsize"].max()) < 1e-3:
            break

    return np.stack(residue_trace), np.stack(stepsize_trace)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config-json", required=True)
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.config_json) as f:
        config_dict = json.load(f)

    model = load_model(args.checkpoint, config_dict, device)
    batch = load_batch(args.data_dir, args.num_examples, device)
    residues, stepsizes = run_and_record(model, batch, args.max_iter)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", residues=residues, stepsizes=stepsizes)
    print(f"Saved: residues {residues.shape}, stepsizes {stepsizes.shape}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    r = residues[:, 0, :] if residues.ndim == 3 else residues[:, 0]
    if r.ndim == 1:
        plt.plot(r, label="sequence-level residual")
    else:
        for t in range(r.shape[1]):
            plt.plot(r[:, t], alpha=0.4, linewidth=1)
    plt.xlabel("fixed-point iteration")
    plt.ylabel("residual (log scale)")
    plt.yscale("log")
    plt.title("Per-token residual convergence (example 0)")
    plt.tight_layout()
    plt.savefig(args.out + "_residual_curves.png", dpi=150)
    print(f"Saved plot: {args.out}_residual_curves.png")


if __name__ == "__main__":
    main()