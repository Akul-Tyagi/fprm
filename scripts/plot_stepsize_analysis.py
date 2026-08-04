import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

"""
Run this locally to generate the 2-panel mechanistic proof for your paper!
Usage:
python scripts/plot_stepsize_analysis.py --npz results/s5_atwd_trajectories.npz --out results/s5_atwd_mechanistic_proof.png
"""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to the saved .npz file from the previous step")
    p.add_argument("--out", required=True, help="Path to save the output PNG")
    p.add_argument("--example-idx", type=int, default=0, help="Which sequence to plot (default: 0)")
    args = p.parse_args()

    # Load the data
    data = np.load(args.npz)
    # np.squeeze removes any empty dimensions like (200, 8, 1, 1) -> (200, 8)
    residues = np.squeeze(data["residues"])
    stepsizes = np.squeeze(data["stepsizes"])

    # Handle residues shape: either (iters, batch) or (iters, batch, seq_len)
    if residues.ndim == 2:
        res_ex = residues[:, args.example_idx]
    else:
        res_ex = residues[:, args.example_idx, :]
        
    # Handle stepsizes shape
    if stepsizes.ndim == 2:
        step_ex = stepsizes[:, args.example_idx]
    else:
        step_ex = stepsizes[:, args.example_idx, :]
    
    # Create a 2-panel plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Panel: Residuals
    if res_ex.ndim == 1:
        ax1.plot(res_ex, alpha=0.8, linewidth=2, label="Sequence Residual")
    else:
        seq_len = res_ex.shape[1]
        for t in range(seq_len):
            ax1.plot(res_ex[:, t], alpha=0.3, linewidth=1)
    
    ax1.set_yscale("log")
    ax1.set_ylabel("Residual (log scale)")
    ax1.set_title(f"Mechanistic Proof (Example {args.example_idx}) - Residuals")
    ax1.grid(True, alpha=0.3)

    # Bottom Panel: Stepsize
    if step_ex.ndim == 1:
        ax2.plot(step_ex, alpha=0.8, linewidth=2, label="Sequence Stepsize")
    else:
        seq_len = step_ex.shape[1]
        for t in range(seq_len):
            ax2.plot(step_ex[:, t], alpha=0.3, linewidth=1)
    
    ax2.set_yscale("log")
    ax2.set_ylabel("Step Size (log scale)")
    ax2.set_xlabel("Fixed-point iteration")
    ax2.set_title("Optimizer Token-wise Stepsize")
    ax2.grid(True, alpha=0.3)
    
    # Draw a line at the 1e-3 threshold to show where ATWD triggers a hard stop
    ax2.axhline(y=1e-3, color='r', linestyle='--', alpha=0.7, label='Halting Threshold (1e-3)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved mechanistic proof plot to: {args.out}")

if __name__ == "__main__":
    main()