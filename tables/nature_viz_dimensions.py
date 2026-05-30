"""
Nature-style visualization of Chinese-English verb distinction dimensions
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import matplotlib.pyplot as plt
import numpy as np
from nature_style import NatureStyle
import pandas as pd

# Initialize Nature style
ns = NatureStyle()

# Load data
df1 = pd.read_csv('table1_descriptive_statistics.csv')
df3 = pd.read_csv('table3_pairwise_comparison_force.csv')
df4 = pd.read_csv('table4_pairwise_comparison_hand.csv')

# Select Chinese and English verbs only
chinese_verbs = df1[df1['Language'] == 'Chinese'].head(6)
english_verbs = df1[df1['Language'] == 'English'].head(6)

# Chinese verb names
cn_verb_names = ['reng', 'diu', 'pao', 'tou', 'shuai 摔', 'shuai 甩']
en_verb_names = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']

# Significance levels to numeric
sig_map = {'': 0, '*': 1, '**': 2, '***': 3}

# ============================================================
# Figure 1: FORCE and HAND pairwise comparison heatmaps
# ============================================================
fig1, axes = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# Chinese FORCE matrix
force_cn = np.zeros((6, 6))
force_cn_data = [
    [0, 2, 0, 0, 3, 1],
    [0, 0, 3, 3, 3, 3],
    [0, 0, 0, 0, 3, 2],
    [0, 0, 0, 0, 3, 1],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0]
]
force_cn = np.array(force_cn_data)

# Chinese HAND matrix
hand_cn_data = [
    [0, 0, 0, 3, 3, 0],
    [0, 0, 2, 3, 3, 3],
    [0, 0, 0, 3, 1, 0],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0]
]
hand_cn = np.array(hand_cn_data)

# Make symmetric
force_cn_sym = force_cn + force_cn.T
hand_cn_sym = hand_cn + hand_cn.T

# Plot FORCE heatmap
im1 = axes[0].imshow(force_cn_sym, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
axes[0].set_xticklabels(cn_verb_names, rotation=45, ha='right', fontsize=5)
axes[0].set_yticklabels(cn_verb_names, fontsize=5)
axes[0].set_title('FORCE (Chinese)', fontweight='bold', fontsize=7, pad=5)

# Add text annotations
for i in range(6):
    for j in range(6):
        val = force_cn_sym[i, j]
        if val > 0:
            text = '*' * val
            axes[0].text(j, i, text, ha='center', va='center', 
                        fontsize=6, color='white' if val > 1 else 'black')

# Plot HAND heatmap
im2 = axes[1].imshow(hand_cn_sym, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[1].set_xticks(range(6))
axes[1].set_yticks(range(6))
axes[1].set_xticklabels(cn_verb_names, rotation=45, ha='right', fontsize=5)
axes[1].set_yticklabels(cn_verb_names, fontsize=5)
axes[1].set_title('HAND (Chinese)', fontweight='bold', fontsize=7, pad=5)

for i in range(6):
    for j in range(6):
        val = hand_cn_sym[i, j]
        if val > 0:
            text = '*' * val
            axes[1].text(j, i, text, ha='center', va='center', 
                        fontsize=6, color='white' if val > 1 else 'black')

# Add colorbars
plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Significance')
plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Significance')

fig1.suptitle('Pairwise Comparison of Chinese Throw Verbs', 
              fontweight='bold', fontsize=8, y=1.02)
fig1.tight_layout()
ns.save_figure(fig1, 'nature_fig1_force_hand_heatmap.png')
ns.save_figure(fig1, 'nature_fig1_force_hand_heatmap.pdf')
print("Saved: nature_fig1_force_hand_heatmap.png/pdf")

# ============================================================
# Figure 2: English FORCE and HAND heatmaps
# ============================================================
fig2, axes2 = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# English FORCE matrix
force_en_data = [
    [0, 0, 0, 2, 3, 3],
    [0, 0, 1, 1, 3, 2],
    [0, 0, 0, 3, 2, 3],
    [0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0]
]
force_en = np.array(force_en_data)
force_en_sym = force_en + force_en.T

# English HAND matrix
hand_en_data = [
    [0, 2, 0, 2, 0, 3],
    [0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0]
]
hand_en = np.array(hand_en_data)
hand_en_sym = hand_en + hand_en.T

# Plot
im3 = axes2[0].imshow(force_en_sym, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes2[0].set_xticks(range(6))
axes2[0].set_yticks(range(6))
axes2[0].set_xticklabels(en_verb_names, rotation=45, ha='right', fontsize=5)
axes2[0].set_yticklabels(en_verb_names, fontsize=5)
axes2[0].set_title('FORCE (English)', fontweight='bold', fontsize=7, pad=5)

for i in range(6):
    for j in range(6):
        val = force_en_sym[i, j]
        if val > 0:
            text = '*' * val
            axes2[0].text(j, i, text, ha='center', va='center', 
                        fontsize=6, color='white' if val > 1 else 'black')

im4 = axes2[1].imshow(hand_en_sym, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes2[1].set_xticks(range(6))
axes2[1].set_yticks(range(6))
axes2[1].set_xticklabels(en_verb_names, rotation=45, ha='right', fontsize=5)
axes2[1].set_yticklabels(en_verb_names, fontsize=5)
axes2[1].set_title('HAND (English)', fontweight='bold', fontsize=7, pad=5)

for i in range(6):
    for j in range(6):
        val = hand_en_sym[i, j]
        if val > 0:
            text = '*' * val
            axes2[1].text(j, i, text, ha='center', va='center', 
                        fontsize=6, color='white' if val > 1 else 'black')

plt.colorbar(im3, ax=axes2[0], shrink=0.8, label='Significance')
plt.colorbar(im4, ax=axes2[1], shrink=0.8, label='Significance')

fig2.suptitle('Pairwise Comparison of English Throw Verbs', 
              fontweight='bold', fontsize=8, y=1.02)
fig2.tight_layout()
ns.save_figure(fig2, 'nature_fig2_en_force_hand_heatmap.png')
ns.save_figure(fig2, 'nature_fig2_en_force_hand_heatmap.pdf')
print("Saved: nature_fig2_en_force_hand_heatmap.png/pdf")

# ============================================================
# Figure 3: Dimensions used by each language (summary)
# ============================================================
fig3, ax3 = ns.create_figure(width=89)

dimensions = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']
cn_dims = [1, 1, 1, 1, 1]  # Chinese uses all 5
en_dims = [1, 1, 1, 0, 0]  # English uses 3

x = np.arange(len(dimensions))
width = 0.35

bars1 = ax3.bar(x - width/2, cn_dims, width, label='Chinese', 
                color=ns.COLORS['red'], alpha=0.8, edgecolor='black', linewidth=0.3)
bars2 = ax3.bar(x + width/2, en_dims, width, label='English', 
                color=ns.COLORS['blue'], alpha=0.8, edgecolor='black', linewidth=0.3)

ax3.set_xlabel('Dimensions', fontsize=6)
ax3.set_ylabel('Significant (1=Yes, 0=No)', fontsize=6)
ax3.set_title('Dimensions for Verb Distinction', fontweight='bold', fontsize=7, pad=5)
ax3.set_xticks(x)
ax3.set_xticklabels(dimensions, fontsize=5)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['No', 'Yes'], fontsize=5)
ax3.legend(fontsize=5, loc='upper right')
ax3.set_ylim(-0.1, 1.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            '√' if height > 0 else '', ha='center', va='bottom', fontsize=6)

for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            '√' if height > 0 else '', ha='center', va='bottom', fontsize=6)

fig3.tight_layout()
ns.save_figure(fig3, 'nature_fig3_dimensions_summary.png')
ns.save_figure(fig3, 'nature_fig3_dimensions_summary.pdf')
print("Saved: nature_fig3_dimensions_summary.png/pdf")

# ============================================================
# Figure 4: Typical values comparison (grouped bar chart)
# ============================================================
fig4, axes4 = ns.create_figure_with_axes(width=183, nrows=1, ncols=3)

# Data for typical values
verbs_all = cn_verb_names + en_verb_names
forces = [3.05, 2.33, 3.16, 3.21, 4.55, 3.74,  # Chinese
          3.62, 3.44, 3.91, 3.01, 4.39, 3.01]   # English

hands = [5.43, 4.63, 6.65, 9.13, 8.51, 6.27,  # Chinese
         8.58, 6.41, 7.06, 6.18, 8.0, 4.5]     # English

# ARM proportions (straight)
arm_props = [0.60, 0.80, 0.52, 0.0, 0.0, 0.28,  # Chinese
             0.17, 0.39, 0.34, 0.48, 0.28, 0.90]  # English

colors_all = [ns.COLORS['red']] * 6 + [ns.COLORS['blue']] * 6
x = np.arange(len(verbs_all))

# FORCE
axes4[0].bar(x, forces, color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.3)
axes4[0].set_ylabel('FORCE (1-5)', fontsize=6)
axes4[0].set_title('FORCE', fontweight='bold', fontsize=7, pad=5)
axes4[0].set_xticks(x)
axes4[0].set_xticklabels(verbs_all, rotation=45, ha='right', fontsize=4)
axes4[0].set_ylim(0, 5.5)

# HAND
axes4[1].bar(x, hands, color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.3)
axes4[1].set_ylabel('HAND (0-12)', fontsize=6)
axes4[1].set_title('HAND', fontweight='bold', fontsize=7, pad=5)
axes4[1].set_xticks(x)
axes4[1].set_xticklabels(verbs_all, rotation=45, ha='right', fontsize=4)
axes4[1].set_ylim(0, 11)

# ARM proportion
axes4[2].bar(x, arm_props, color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.3)
axes4[2].set_ylabel('ARM straight (proportion)', fontsize=6)
axes4[2].set_title('ARM', fontweight='bold', fontsize=7, pad=5)
axes4[2].set_xticks(x)
axes4[2].set_xticklabels(verbs_all, rotation=45, ha='right', fontsize=4)
axes4[2].set_ylim(0, 1.1)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ns.COLORS['red'], alpha=0.8, label='Chinese'),
                   Patch(facecolor=ns.COLORS['blue'], alpha=0.8, label='English')]
axes4[0].legend(handles=legend_elements, fontsize=5, loc='upper left')

fig4.suptitle('Typical Values Across Verbs', fontweight='bold', fontsize=8, y=1.02)
fig4.tight_layout()
ns.save_figure(fig4, 'nature_fig4_typical_values.png')
ns.save_figure(fig4, 'nature_fig4_typical_values.pdf')
print("Saved: nature_fig4_typical_values.png/pdf")

# ============================================================
# Figure 5: VD and HD patterns
# ============================================================
fig5, axes5 = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# VD patterns (proportion upward)
vd_up = [0.72, 0.82, 1.0, 0.97, 0.0, 0.48,  # Chinese
         0.90, 0.93, 0.72, 0.97, 0.83, 1.0]   # English

# HD patterns (proportion forward)
hd_fwd = [0.72, 0.57, 0.97, 1.0, 0.86, 0.31,  # Chinese
          1.0, 0.86, 0.93, 0.93, 0.97, 1.0]    # English

# VD bar chart
axes5[0].bar(x, vd_up, color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.3)
axes5[0].set_ylabel('VD upward (proportion)', fontsize=6)
axes5[0].set_title('Vertical Direction', fontweight='bold', fontsize=7, pad=5)
axes5[0].set_xticks(x)
axes5[0].set_xticklabels(verbs_all, rotation=45, ha='right', fontsize=4)
axes5[0].set_ylim(0, 1.15)
axes5[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# HD bar chart
axes5[1].bar(x, hd_fwd, color=colors_all, alpha=0.8, edgecolor='black', linewidth=0.3)
axes5[1].set_ylabel('HD forward (proportion)', fontsize=6)
axes5[1].set_title('Horizontal Direction', fontweight='bold', fontsize=7, pad=5)
axes5[1].set_xticks(x)
axes5[1].set_xticklabels(verbs_all, rotation=45, ha='right', fontsize=4)
axes5[1].set_ylim(0, 1.15)
axes5[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

legend_elements = [Patch(facecolor=ns.COLORS['red'], alpha=0.8, label='Chinese'),
                   Patch(facecolor=ns.COLORS['blue'], alpha=0.8, label='English')]
axes5[0].legend(handles=legend_elements, fontsize=5, loc='lower left')

fig5.suptitle('Direction Patterns Across Verbs', fontweight='bold', fontsize=8, y=1.02)
fig5.tight_layout()
ns.save_figure(fig5, 'nature_fig5_direction_patterns.png')
ns.save_figure(fig5, 'nature_fig5_direction_patterns.pdf')
print("Saved: nature_fig5_direction_patterns.png/pdf")

print("\n✅ All Nature-style figures generated successfully!")
