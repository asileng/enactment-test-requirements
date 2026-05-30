"""
Nature-style pairwise matrix visualization
5 parameters × 15 verb pairs for Chinese and English
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import matplotlib.pyplot as plt
import numpy as np
from nature_style import NatureStyle
import pandas as pd
from itertools import combinations

ns = NatureStyle()

# Chinese verbs
cn_verbs = ['reng', 'diu', 'pao', 'tou', 'shuai 摔', 'shuai 甩']
# English verbs
en_verbs = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']

# Create all unique pairs
cn_pairs = list(combinations(range(6), 2))
en_pairs = list(combinations(range(6), 2))

# ============================================================
# Data: significance levels for each parameter and pair
# Format: (param, verb1_idx, verb2_idx, significance)
# sig: 0=ns, 1=*, 2=**, 3=***
# ============================================================

# Chinese data (from Table 3, 4, and 5)
# FORCE significance (Table 3)
cn_force = {
    (0,1): 2, (0,2): 0, (0,3): 0, (0,4): 3, (0,5): 1,
    (1,2): 3, (1,3): 3, (1,4): 3, (1,5): 3,
    (2,3): 0, (2,4): 3, (2,5): 2,
    (3,4): 3, (3,5): 1,
    (4,5): 3
}

# HAND significance (Table 4)
cn_hand = {
    (0,1): 0, (0,2): 0, (0,3): 3, (0,4): 3, (0,5): 0,
    (1,2): 2, (1,3): 3, (1,4): 3, (1,5): 3,
    (2,3): 3, (2,4): 1, (2,5): 0,
    (3,4): 0, (3,5): 3,
    (4,5): 3
}

# ARM significance (from Table 5 typical values)
# diu and tou are significantly different (straight vs bend)
cn_arm = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 0,
    (1,2): 0, (1,3): 3, (1,4): 3, (1,5): 3,
    (2,3): 0, (2,4): 0, (2,5): 0,
    (3,4): 0, (3,5): 3,
    (4,5): 3
}

# HD significance (from Table 5)
# shuai 甩 is sidewise, others are forward
cn_hd = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 2,
    (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 2,
    (2,3): 0, (2,4): 0, (2,5): 2,
    (3,4): 0, (3,5): 2,
    (4,5): 2
}

# VD significance (from Table 5)
# shuai 摔 is downward, others are upward
cn_vd = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 3, (0,5): 0,
    (1,2): 0, (1,3): 0, (1,4): 3, (1,5): 0,
    (2,3): 0, (2,4): 3, (2,5): 0,
    (3,4): 3, (3,5): 0,
    (4,5): 3
}

# English data
en_force = {
    (0,1): 0, (0,2): 0, (0,3): 2, (0,4): 3, (0,5): 3,
    (1,2): 1, (1,3): 1, (1,4): 3, (1,5): 2,
    (2,3): 3, (2,4): 2, (2,5): 3,
    (3,4): 3, (3,5): 0,
    (4,5): 3
}

en_hand = {
    (0,1): 2, (0,2): 0, (0,3): 2, (0,4): 0, (0,5): 3,
    (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 2,
    (2,3): 0, (2,4): 0, (2,5): 3,
    (3,4): 1, (3,5): 1,
    (4,5): 3
}

en_arm = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 3,
    (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 3,
    (2,3): 0, (2,4): 0, (2,5): 3,
    (3,4): 0, (3,5): 3,
    (4,5): 3
}

en_hd = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 0,
    (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 0,
    (2,3): 0, (2,4): 0, (2,5): 0,
    (3,4): 0, (3,5): 0,
    (4,5): 0
}

en_vd = {
    (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 0,
    (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 0,
    (2,3): 0, (2,4): 0, (2,5): 0,
    (3,4): 0, (3,5): 0,
    (4,5): 0
}

# ============================================================
# Create long-format DataFrames
# ============================================================

def create_long_format(verbs, force, hand, arm, hd, vd):
    """Create long-format DataFrame for visualization."""
    rows = []
    pairs = list(combinations(range(len(verbs)), 2))
    
    for i, (v1, v2) in enumerate(pairs):
        pair_name = f"{verbs[v1]} vs {verbs[v2]}"
        
        # FORCE
        rows.append({
            'pair_idx': i,
            'pair': pair_name,
            'verb1': verbs[v1],
            'verb2': verbs[v2],
            'parameter': 'FORCE',
            'significance': force.get((v1, v2), 0)
        })
        
        # HAND
        rows.append({
            'pair_idx': i,
            'pair': pair_name,
            'verb1': verbs[v1],
            'verb2': verbs[v2],
            'parameter': 'HAND',
            'significance': hand.get((v1, v2), 0)
        })
        
        # ARM
        rows.append({
            'pair_idx': i,
            'pair': pair_name,
            'verb1': verbs[v1],
            'verb2': verbs[v2],
            'parameter': 'ARM',
            'significance': arm.get((v1, v2), 0)
        })
        
        # HD
        rows.append({
            'pair_idx': i,
            'pair': pair_name,
            'verb1': verbs[v1],
            'verb2': verbs[v2],
            'parameter': 'HD',
            'significance': hd.get((v1, v2), 0)
        })
        
        # VD
        rows.append({
            'pair_idx': i,
            'pair': pair_name,
            'verb1': verbs[v1],
            'verb2': verbs[v2],
            'parameter': 'VD',
            'significance': vd.get((v1, v2), 0)
        })
    
    return pd.DataFrame(rows)

cn_df = create_long_format(cn_verbs, cn_force, cn_hand, cn_arm, cn_hd, cn_vd)
en_df = create_long_format(en_verbs, en_force, en_hand, en_arm, en_hd, en_vd)

# Save CSV files
cn_df.to_csv('pairwise_long_format_chinese.csv', index=False)
en_df.to_csv('pairwise_long_format_english.csv', index=False)
print("Saved: pairwise_long_format_chinese.csv")
print("Saved: pairwise_long_format_english.csv")

# ============================================================
# Create heatmap matrices (15 pairs × 5 parameters)
# ============================================================

def create_matrix(df, verbs):
    """Create matrix from long-format DataFrame."""
    pairs = list(combinations(range(len(verbs)), 2))
    params = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']
    
    matrix = np.zeros((len(pairs), 5))
    
    for _, row in df.iterrows():
        pair_idx = row['pair_idx']
        param_idx = params.index(row['parameter'])
        matrix[pair_idx, param_idx] = row['significance']
    
    return matrix

cn_matrix = create_matrix(cn_df, cn_verbs)
en_matrix = create_matrix(en_df, en_verbs)

# Create pair labels
cn_pair_labels = [f"{cn_verbs[v1]}-{cn_verbs[v2]}" for v1, v2 in cn_pairs]
en_pair_labels = [f"{en_verbs[v1]}-{en_verbs[v2]}" for v1, v2 in en_pairs]

params = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

# ============================================================
# Figure: Side-by-side heatmaps
# ============================================================

fig, axes = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# Chinese heatmap
im1 = axes[0].imshow(cn_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[0].set_xticks(range(5))
axes[0].set_yticks(range(15))
axes[0].set_xticklabels(params, fontsize=6, fontweight='bold')
axes[0].set_yticklabels(cn_pair_labels, fontsize=4)
axes[0].set_title('Chinese', fontweight='bold', fontsize=8, pad=10)

# Add text annotations
for i in range(15):
    for j in range(5):
        val = int(cn_matrix[i, j])
        if val > 0:
            text = '*' * val
            axes[0].text(j, i, text, ha='center', va='center', 
                        fontsize=5, color='white' if val > 1 else 'black')

# English heatmap
im2 = axes[1].imshow(en_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[1].set_xticks(range(5))
axes[1].set_yticks(range(15))
axes[1].set_xticklabels(params, fontsize=6, fontweight='bold')
axes[1].set_yticklabels(en_pair_labels, fontsize=4)
axes[1].set_title('English', fontweight='bold', fontsize=8, pad=10)

for i in range(15):
    for j in range(5):
        val = int(en_matrix[i, j])
        if val > 0:
            text = '*' * val
            axes[1].text(j, i, text, ha='center', va='center', 
                        fontsize=5, color='white' if val > 1 else 'black')

# Colorbars
plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Significance')
plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Significance')

fig.suptitle('Pairwise Significance Matrix: Which Parameters Distinguish Verb Pairs?',
             fontweight='bold', fontsize=9, y=1.02)
fig.tight_layout()

ns.save_figure(fig, 'nature_fig6_pairwise_matrix.png')
ns.save_figure(fig, 'nature_fig6_pairwise_matrix.pdf')
print("Saved: nature_fig6_pairwise_matrix.png/pdf")

# ============================================================
# Figure: Summary - Count of significant pairs per parameter
# ============================================================

fig2, axes2 = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# Count significant pairs (sig >= 1) per parameter
cn_counts = [np.sum(cn_matrix[:, j] >= 1) for j in range(5)]
en_counts = [np.sum(en_matrix[:, j] >= 1) for j in range(5)]

# Count highly significant pairs (sig >= 3)
cn_high = [np.sum(cn_matrix[:, j] >= 3) for j in range(5)]
en_high = [np.sum(en_matrix[:, j] >= 3) for j in range(5)]

x = np.arange(5)
width = 0.35

# Chinese
bars1 = axes2[0].bar(x - width/2, cn_counts, width, label='Any sig. (*)',
                     color=ns.COLORS['red'], alpha=0.6, edgecolor='black', linewidth=0.3)
bars2 = axes2[0].bar(x + width/2, cn_high, width, label='Highly sig. (***)',
                     color=ns.COLORS['red'], alpha=1.0, edgecolor='black', linewidth=0.3)

axes2[0].set_xlabel('Parameter', fontsize=6)
axes2[0].set_ylabel('Number of verb pairs', fontsize=6)
axes2[0].set_title('Chinese', fontweight='bold', fontsize=8, pad=5)
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(params, fontsize=5)
axes2[0].set_ylim(0, 16)
axes2[0].legend(fontsize=5)

# Add count labels
for bar in bars1:
    height = bar.get_height()
    axes2[0].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=5)

for bar in bars2:
    height = bar.get_height()
    axes2[0].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=5)

# English
bars3 = axes2[1].bar(x - width/2, en_counts, width, label='Any sig. (*)',
                     color=ns.COLORS['blue'], alpha=0.6, edgecolor='black', linewidth=0.3)
bars4 = axes2[1].bar(x + width/2, en_high, width, label='Highly sig. (***)',
                     color=ns.COLORS['blue'], alpha=1.0, edgecolor='black', linewidth=0.3)

axes2[1].set_xlabel('Parameter', fontsize=6)
axes2[1].set_ylabel('Number of verb pairs', fontsize=6)
axes2[1].set_title('English', fontweight='bold', fontsize=8, pad=5)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(params, fontsize=5)
axes2[1].set_ylim(0, 16)
axes2[1].legend(fontsize=5)

for bar in bars3:
    height = bar.get_height()
    axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=5)

for bar in bars4:
    height = bar.get_height()
    axes2[1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{int(height)}', ha='center', va='bottom', fontsize=5)

fig2.suptitle('Parameter Discriminability: How Many Verb Pairs Are Distinguished?',
              fontweight='bold', fontsize=9, y=1.02)
fig2.tight_layout()

ns.save_figure(fig2, 'nature_fig7_param_discriminability.png')
ns.save_figure(fig2, 'nature_fig7_param_discriminability.pdf')
print("Saved: nature_fig7_param_discriminability.png/pdf")

# ============================================================
# Print summary statistics
# ============================================================

print("\n" + "="*60)
print("SUMMARY: Parameter Discriminability")
print("="*60)

print("\nChinese (6 verbs, 15 pairs):")
print(f"  FORCE: {cn_counts[0]} pairs significant, {cn_high[0]} highly significant")
print(f"  HAND:  {cn_counts[1]} pairs significant, {cn_high[1]} highly significant")
print(f"  ARM:   {cn_counts[2]} pairs significant, {cn_high[2]} highly significant")
print(f"  HD:    {cn_counts[3]} pairs significant, {cn_high[3]} highly significant")
print(f"  VD:    {cn_counts[4]} pairs significant, {cn_high[4]} highly significant")

print("\nEnglish (6 verbs, 15 pairs):")
print(f"  FORCE: {en_counts[0]} pairs significant, {en_high[0]} highly significant")
print(f"  HAND:  {en_counts[1]} pairs significant, {en_high[1]} highly significant")
print(f"  ARM:   {en_counts[2]} pairs significant, {en_high[2]} highly significant")
print(f"  HD:    {en_counts[3]} pairs significant, {en_high[3]} highly significant")
print(f"  VD:    {en_counts[4]} pairs significant, {en_high[4]} highly significant")

print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)
print(f"Chinese best discriminators: FORCE ({cn_counts[0]}), HAND ({cn_counts[1]})")
print(f"English best discriminators: FORCE ({en_counts[0]}), HAND ({en_counts[1]})")
print(f"HD/VD only useful for Chinese, not English")
