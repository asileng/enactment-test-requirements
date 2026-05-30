import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm

# Register CJK font
cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
fm.fontManager.addfont(cjk_font_path)
plt.rcParams['font.family'] = ['Noto Serif CJK JP', 'DejaVu Sans']

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/lulu444/Enactment-experiment/tables/table1_descriptive_statistics.csv')

# Select 6 Chinese + 6 English verbs
chinese_verbs = df[df['Language'] == 'Chinese'].head(6).copy()
english_verbs = df[df['Language'] == 'English'].head(6).copy()
verbs_df = pd.concat([chinese_verbs, english_verbs], ignore_index=True)

print("Verbs selected:")
for i, row in verbs_df.iterrows():
    print(f"  v{i+1}: {row['Language']} - {row['Verb']}")

# ── 2. Extract and normalize 5 parameters to [0,1] ───────────────────────────
# p1: FORCE = FORCE Mean (scale 1-5, normalize to 0-1)
# p2: HAND = HAND Mean (scale 0-12, normalize to 0-1)
# p3: ARM = straight / (bend + straight) → proportion of straight arm
# p4: HD = forward / (forward + sidewise) → proportion of forward
# p5: VD = upward / (upward + downward) → proportion of upward

params = np.zeros((len(verbs_df), 5))

for i, row in verbs_df.iterrows():
    # p1: FORCE (1-5 → 0-1)
    params[i, 0] = (row['FORCE Mean'] - 1) / (5 - 1)
    # p2: HAND (0-12 → 0-1)
    params[i, 1] = row['HAND Mean'] / 12
    # p3: ARM (proportion straight) - clip to minimum 0.05
    arm_total = row['ARM bend'] + row['ARM straight']
    params[i, 2] = max(row['ARM straight'] / arm_total, 0.05) if arm_total > 0 else 0.5
    # p4: HD (proportion forward) - clip to minimum 0.05
    hd_total = row['HD forward'] + row['HD sidewise']
    params[i, 3] = max(row['HD forward'] / hd_total, 0.05) if hd_total > 0 else 0.5
    # p5: VD (proportion upward) - clip to minimum 0.05
    vd_total = row['VD upward'] + row['VD downward']
    params[i, 4] = max(row['VD upward'] / vd_total, 0.05) if vd_total > 0 else 0.5

print("\nNormalized parameters (0-1):")
param_names = ['FORCE', 'HAND', 'ARM(straight)', 'HD(forward)', 'VD(upward)']
for i, row in verbs_df.iterrows():
    print(f"  {row['Verb']:12s}: {params[i]}")

# ── 3. Set up polar figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 14), facecolor='white')
ax = fig.add_subplot(111, projection='polar')

# Sector configuration
n_verbs = 12
sector_angle = 2 * np.pi / n_verbs  # 30 degrees per sector

# Single color scheme for all verbs (no language distinction)
color_human = '#808080'   # Gray for human data

# 12 unique line styles for each sector/verb
line_styles = [
    '-',    # solid
    '--',   # dashed
    '-.',   # dash-dot
    ':',    # dotted
    (0, (3, 1, 1, 1)),  # loosely dashed
    (0, (5, 2)),        # dashed
    (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    (0, (1, 1)),        # densely dotted
    (0, (5, 1, 1, 1)),  # loosely dash-dot
    (0, (2, 2)),        # loosely dotted
    (0, (5, 1, 2, 1)),  # dash-dot-dot
    (0, (1, 2)),        # densely dashed
]

# ── 4. Draw sector boundaries and labels ─────────────────────────────────────
ax.set_theta_offset(np.pi / 2)  # Start from top (12 o'clock)
ax.set_theta_direction(-1)       # Clockwise

# Draw sector boundary lines
for i in range(n_verbs):
    angle = i * sector_angle
    ax.plot([angle, angle], [0, 1.15], color='gray', linewidth=0.5, alpha=0.5)

# Add verb labels at outer edge
label_radius = 1.08
for i, row in verbs_df.iterrows():
    angle = i * sector_angle
    verb_name = row['Verb']
    lang = row['Language']

    # Rotate text to be readable
    angle_deg = np.degrees(angle)
    if 90 < angle_deg < 270:
        angle_deg += 180

    ax.text(angle, label_radius, verb_name,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=color_human, rotation=angle_deg - 90,
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

# ── 5. Plot human trajectories in each sector ────────────────────────────────
# Within each sector, distribute 5 points evenly across the sector width
point_offsets = np.linspace(-sector_angle * 0.3, sector_angle * 0.3, 5)

# Collect all points for cross-verb connections
all_thetas = np.zeros((n_verbs, 5))
all_radii = np.zeros((n_verbs, 5))

for i, row in verbs_df.iterrows():
    sector_center = i * sector_angle
    r_values = params[i]  # 5 normalized parameter values

    # Calculate (theta, r) for each of the 5 points
    thetas = sector_center + point_offsets
    radii = r_values

    all_thetas[i] = thetas
    all_radii[i] = radii

    # Gray color for human, different line style for each sector
    color = color_human
    ls = line_styles[i]

    # Plot open polyline with all 5 points clearly visible
    ax.plot(thetas, radii, '-o', color=color, linestyle=ls,
            linewidth=2.5, markersize=7, markeredgecolor='white',
            markeredgewidth=1, alpha=0.9, zorder=5)

# ── 6. Style the plot ────────────────────────────────────────────────────────
# Radial ticks
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.set_ylim(0, 1.15)

# Remove default angular ticks (we have custom labels)
ax.set_xticks([])

# Grid styling
ax.grid(True, color='lightgray', linewidth=0.5, alpha=0.7)
ax.spines['polar'].set_visible(False)

# Title
ax.set_title('Polar Sector Visualization: Human Behavioral Trajectories\n'
             'for "Throw" Verbs (Chinese + English)',
             fontsize=14, fontweight='bold', pad=30)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_human, linewidth=2, marker='o', markersize=5, label='Human trajectory (gray, different styles per verb)'),
]
ax.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(1.35, 1.1), fontsize=8)

# Add parameter order annotation
param_text = ('Parameter order (radial polyline):\n'
              'p1=FORCE → p2=HAND → p3=ARM → p4=HD → p5=VD')
fig.text(0.5, 0.02, param_text, ha='center', fontsize=9, color='gray',
         style='italic')

plt.tight_layout()

# ── 7. Save ──────────────────────────────────────────────────────────────────
output_path = '/home/lulu444/Enactment-experiment/tables/polar_sector_visualization'
plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_path}.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\nSaved: {output_path}.png")
print(f"Saved: {output_path}.pdf")
plt.close()
