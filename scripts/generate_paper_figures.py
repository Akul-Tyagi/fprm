import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# --- FIGURE 3: Constraint Violations (Overall vs Conditional on Failure) ---
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
arms = ['Hard Freeze', 'Baseline', 'ATWD (r=20)']
# UPDATED: Corrected EMA-aligned overall means
overall = [24.84, 28.96, 29.31] 
cond_fail = [30.94, 39.61, 39.01]

x = np.arange(len(arms))
width = 0.35

rects1 = ax.bar(x - width/2, overall, width, label='Overall Test Set', color='#4C72B0')
rects2 = ax.bar(x + width/2, cond_fail, width, label='Conditional on Failure', color='#C44E52')

ax.set_ylabel('Mean Duplicate Violations / Puzzle')
ax.set_title('Sudoku: Graceful Failure via Greedy Local Optimization')
ax.set_xticks(x)
ax.set_xticklabels(arms)
ax.legend(frameon=True)
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

plt.savefig("results/figures/fig3_sudoku_violations.png")
plt.close()
print("Saved: results/figures/fig3_sudoku_violations.png")

# --- FIGURE 6: Clean vs. Messy Convergence Heatmap Contrast ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=300)
npz_s0 = np.load("results/analysis_atwd_seed0.npz")
res_s0 = npz_s0['residues']

# Clean example: ex2; Messy example: ex1
clean_grid = np.mean(res_s0[-100:, 2, -81:] < 0.1, axis=0).reshape(9, 9)
messy_grid = np.mean(res_s0[-100:, 1, -81:] < 0.1, axis=0).reshape(9, 9)

for ax, grid, title in zip(axes, [clean_grid, messy_grid], ['Clean Convergence (Ex 2)', 'Messy Thrashing (Ex 1)']):
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i - 0.5, color="white", linewidth=lw)
        ax.axvline(i - 0.5, color="white", linewidth=lw)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label("Fraction of Final 100 Iterations < 0.1")
plt.savefig("results/figures/fig6_clean_vs_messy_heatmap.png")
plt.close()
print("Saved: results/figures/fig6_clean_vs_messy_heatmap.png")

# --- FIGURE 7: A5 vs Sudoku Reliability Inversion ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Panel A: A5 (Causal)
a5_base = [0.930, 0.308, 0.058]
a5_atwd = [0.976, 0.891, 0.945]
x = np.arange(3)
width = 0.35

axes[0].bar(x - width/2, a5_base, width, label='Baseline', color='#4C72B0')
axes[0].bar(x + width/2, a5_atwd, width, label='ATWD', color='#55A868')
axes[0].set_title('A5 Permutation (Causal Attention)')
axes[0].set_ylabel('Sequence Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Seed 0', 'Seed 1', 'Seed 2'])
axes[0].set_ylim(0, 1.05)
axes[0].legend()

# Panel B: Sudoku (Bidirectional)
sudo_base = [0.2675, 0.2725, 0.2663]
sudo_atwd = [0.2713, 0.2538, 0.2200]

axes[1].bar(x - width/2, sudo_base, width, label='Baseline', color='#4C72B0')
axes[1].bar(x + width/2, sudo_atwd, width, label='ATWD', color='#55A868')
axes[1].set_title('Sudoku 9x9 (Bidirectional Attention)')
axes[1].set_ylabel('Sequence Accuracy')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Seed 0', 'Seed 1', 'Seed 2'])
axes[1].set_ylim(0, 0.35)
axes[1].legend()

plt.savefig("results/figures/fig7_reliability_inversion.png")
plt.close()
print("Saved: results/figures/fig7_reliability_inversion.png")