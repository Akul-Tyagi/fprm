import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("results/figures", exist_ok=True)

# --- FIGURE 7: A5 vs Sudoku Reliability Inversion ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Custom elegant ICML-style color palette
color_base = '#4A235A'
color_atwd = '#F39C12'

# Panel A: A5 (Causal)
a5_base = [0.930, 0.308, 0.058]
a5_atwd = [0.976, 0.891, 0.978]  # Value updated to 0.978 to match Astra Audit
x = np.arange(3)
width = 0.35

axes[0].bar(x - width/2, a5_base, width, label='Baseline', color=color_base, edgecolor='none')
axes[0].bar(x + width/2, a5_atwd, width, label='ATWD', color=color_atwd, edgecolor='none')
axes[0].set_title('A5 Permutation (Causal Attention)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_ylabel('Sequence Accuracy', fontsize=12, labelpad=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Seed 0', 'Seed 1', 'Seed 2'], fontsize=11)
axes[0].set_ylim(0, 1.05)

# Panel B: Sudoku (Bidirectional)
sudo_base = [0.2675, 0.2725, 0.2663]
sudo_atwd = [0.2713, 0.2538, 0.2200]

axes[1].bar(x - width/2, sudo_base, width, label='Baseline', color=color_base, edgecolor='none')
axes[1].bar(x + width/2, sudo_atwd, width, label='ATWD', color=color_atwd, edgecolor='none')
axes[1].set_title('Sudoku 9x9 (Bidirectional Attention)', fontsize=13, fontweight='bold', pad=12)
axes[1].set_ylabel('Sequence Accuracy', fontsize=12, labelpad=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Seed 0', 'Seed 1', 'Seed 2'], fontsize=11)
axes[1].set_ylim(0, 0.35)

# Apply global aesthetics to both panels
for ax in axes:
    # Subtle horizontal grid only
    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='x', visible=False)
    
    # Despine top and right borders completely
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bbbbbb')
    ax.spines['bottom'].set_color('#bbbbbb')
    
    # Elegant legend with no box
    ax.legend(frameon=False, fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig("results/figures/fig7_reliability_inversion.png", bbox_inches='tight', transparent=False)
plt.close()
print("Saved upgraded figure to: results/figures/fig7_reliability_inversion.png")