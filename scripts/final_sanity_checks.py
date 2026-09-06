import numpy as np
import os

# 1. Difficulty Correlation Check
diff_path = "data/sudoku-extreme-1k-aug1k/test/all__group_difficulties.npy"
if os.path.exists(diff_path):
    diffs = np.load(diff_path)[:4]
    print("--- Difficulty Ratings ---")
    for i, d in enumerate(diffs):
        status = " (MESSY)" if i in [1, 3] else " (CLEAN)"
        print(f"Example {i}: Difficulty = {d:.4f}{status}")
else:
    print(f"Difficulty file not found at {diff_path}")

# 2. Periodic Refresh Spike Check
def check_spikes(npz_file, example_idx):
    if not os.path.exists(npz_file):
        print(f"File not found: {npz_file}")
        return
        
    data = np.load(npz_file)
    residues = data['residues'] # (iters, batch, tokens)
    
    # Get the mean residual across all tokens for the specified example
    mean_res = residues[:, example_idx, :].mean(axis=1)
    
    print(f"\n--- Spike Check: {npz_file} (Example {example_idx}) ---")
    print("Iter | Mean Residual")
    print("--------------------")
    # Print a small 25-iteration window to visually spot the spikes
    for i in range(800, 825):
        marker = "<-- REFRESH TRIGGER (Multiple of 20)" if i % 20 == 0 else ""
        print(f"{i:4d} | {mean_res[i]:.4f}  {marker}")

check_spikes("results/analysis_atwd_seed0.npz", 1)
check_spikes("results/analysis_atwd_seed2.npz", 3)