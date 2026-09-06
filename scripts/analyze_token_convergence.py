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
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1

def load_batch(data_dir, num_examples, device):
    inputs = np.load(os.path.join(data_dir, "test", "all__inputs.npy"))[:num_examples]
    labels = np.load(os.path.join(data_dir, "test", "all__labels.npy"))[:num_examples]
    puzzle_ids = np.load(os.path.join(data_dir, "test", "all__puzzle_identifiers.npy"))[:num_examples]
    return {
        "inputs": torch.from_numpy(inputs).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "puzzle_identifiers": torch.from_numpy(puzzle_ids).to(device),
    }

def load_model(checkpoint_dir, device, batch):
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    assert os.path.exists(config_path), f"Config not found at {config_path}"
    config = OmegaConf.load(config_path)
    arch_dict = OmegaConf.to_container(config.arch, resolve=True)
    
    all_cands = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    cands = [p for p in all_cands if not p.endswith("_train_state.pt") and not p.endswith(".yaml")]
    assert cands, f"No EMA checkpoint found in {checkpoint_dir}"
    latest = max(cands, key=lambda p: int(os.path.basename(p).replace('.pt', '').split("_")[1]))
    
    raw_sd = torch.load(latest, map_location=device)
    
    arch_dict["batch_size"] = batch["inputs"].shape[0]
    arch_dict["seq_len"] = batch["inputs"].shape[1]
    
    for k, v in raw_sd.items():
        if v.ndim == 2:
            if "puzzle" in k.lower() and "emb" in k.lower():
                arch_dict["num_puzzle_identifiers"] = v.shape[0]
            elif ("token" in k.lower() or "word" in k.lower() or "embed" in k.lower()) and "pos" not in k.lower():
                if v.shape[0] < 50:
                    arch_dict["vocab_size"] = v.shape[0]

    if "vocab_size" not in arch_dict:
        arch_dict["vocab_size"] = getattr(config, "vocab_size", 20)
    if "num_puzzle_identifiers" not in arch_dict:
        arch_dict["num_puzzle_identifiers"] = getattr(config, "num_puzzle_identifiers", 2)
    
    model = FixedPointReasoningModel_ACTV1(arch_dict).to(device).to(torch.bfloat16)
    
    prefix = "_orig_mod.model."
    sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw_sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

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

def sudoku_convergence_heatmap(residues, threshold, out_path):
    iters, T = residues.shape
    assert T == 81, f"expected 81 Sudoku cells, got {T}"
    
    window = min(100, iters)
    final_window = residues[-window:, :]
    convergence_score = np.mean(final_window < threshold, axis=0)
    grid = convergence_score.reshape(9, 9)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label=f"Fraction of final {window} iters < {threshold}")
    plt.title("Sudoku: Sustained Convergence (81 Grid Cells)")
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        plt.axhline(i - 0.5, color="white", linewidth=lw)
        plt.axvline(i - 0.5, color="white", linewidth=lw)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def sudoku_97_token_barchart(residues, threshold, out_path):
    iters, T = residues.shape
    assert T == 97, f"expected 97 total tokens, got {T}"
    
    window = min(100, iters)
    final_window = residues[-window:, :]
    convergence_score = np.mean(final_window < threshold, axis=0)

    plt.figure(figsize=(10, 4))
    colors = ['tomato']*16 + ['steelblue']*81
    plt.bar(range(T), convergence_score, color=colors)
    plt.axvline(15.5, color='black', linestyle='--', linewidth=2, label='Workspace / Grid boundary')
    plt.xlabel("Token Index")
    plt.ylabel(f"Sustained Convergence (Last {window} iters)")
    plt.title("Sudoku: Workspace (0-15) vs Grid (16-96) Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading data from {args.data_dir}...")
    batch = load_batch(args.data_dir, args.num_examples, device)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device, batch)
    
    print("Running fixed-point solver...")
    residues, stepsizes = run_and_record(model, batch, args.max_iter)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", residues=residues, stepsizes=stepsizes)

    num_to_plot = min(4, residues.shape[1])
    
    for ex_idx in range(num_to_plot):
        r = residues[:, ex_idx, :]
        
        # 1. Residual Curves
        plt.figure(figsize=(8, 5))
        for t in range(r.shape[1]):
            plt.plot(r[:, t], alpha=0.4, linewidth=1)
        plt.xlabel("fixed-point iteration")
        plt.ylabel("residual (log scale)")
        plt.yscale("log")
        plt.title(f"Per-token residual convergence (Example {ex_idx})")
        plt.tight_layout()
        curve_out = f"{args.out}_ex{ex_idx}_residual_curves.png"
        plt.savefig(curve_out, dpi=150)
        plt.close()
        print(f"Saved plot: {curve_out}")

        # 2. Sudoku Visualizations
        if "sudoku" in args.checkpoint:
            if r.shape[-1] == 97:
                bar_out = f"{args.out}_ex{ex_idx}_97token_convergence.png"
                sudoku_97_token_barchart(r, threshold=0.1, out_path=bar_out)
            
            grid_residues = r[:, -81:] if r.shape[-1] > 81 else r
            if grid_residues.shape[-1] == 81:
                heat_out = f"{args.out}_ex{ex_idx}_sudoku_heatmap.png"
                sudoku_convergence_heatmap(grid_residues, threshold=0.1, out_path=heat_out)

if __name__ == "__main__":
    main()