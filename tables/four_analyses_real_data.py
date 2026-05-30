"""
四个分析：使用真实的Mimo-embodied-7B和RoboBrain2.0-7B数据
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import numpy as np
import pandas as pd
import json
import glob
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from nature_style import NatureStyle

ns = NatureStyle()

# ============================================================
# 读取人类基准数据
# ============================================================

df_human = pd.read_csv('table1_descriptive_statistics.csv')
cn_verbs_human = df_human[df_human['Language'] == 'Chinese'].head(6).copy().reset_index(drop=True)
en_verbs_human = df_human[df_human['Language'] == 'English'].head(6).copy().reset_index(drop=True)

def extract_human_features(verbs_df):
    """从人类数据提取30维特征向量"""
    features = []
    for _, row in verbs_df.iterrows():
        force = (row['FORCE Mean'] - 1) / 4
        hand = row['HAND Mean'] / 12
        arm_total = row['ARM bend'] + row['ARM straight']
        arm = row['ARM straight'] / arm_total if arm_total > 0 else 0.5
        hd_total = row['HD forward'] + row['HD sidewise']
        hd = row['HD forward'] / hd_total if hd_total > 0 else 0.5
        vd_total = row['VD upward'] + row['VD downward']
        vd = row['VD upward'] / vd_total if vd_total > 0 else 0.5
        features.extend([force, hand, arm, hd, vd])
    return np.array(features)

human_cn = extract_human_features(cn_verbs_human)
human_en = extract_human_features(en_verbs_human)

print("Human CN shape:", human_cn.shape)
print("Human EN shape:", human_en.shape)

# ============================================================
# 读取模型数据
# ============================================================

def extract_model_features(model_name, language):
    """从模型JSON文件提取30维特征向量"""
    verb_map_zh = {'扔': 0, '丢': 1, '抛': 2, '投': 3, '摔': 4, '甩': 5}
    verb_map_en = {'throw': 0, 'fling': 1, 'chuck': 2, 'cast': 3, 'hurl': 4, 'toss': 5}
    
    verb_map = verb_map_zh if language == 'zh' else verb_map_en
    
    # 初始化特征向量
    features = np.zeros((6, 5))
    counts = np.zeros(6)
    
    # 读取所有JSON文件
    files = glob.glob(f'pilot_results/{model_name}/task1_{language}/*.json')
    
    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
            
            if not data.get('is_valid', False):
                continue
            
            verb = data.get('verb', '')
            if verb not in verb_map:
                continue
            
            verb_idx = verb_map[verb]
            result = data.get('parsed_result', {})
            
            if result:
                # 归一化参数
                force = (result.get('FORCE', 3) - 1) / 4
                hand = result.get('HAND', 6) / 12
                arm = result.get('ARM', 0)  # 0=bend, 1=straight
                hd = result.get('HD', 1)    # 0=sideways, 1=forward
                vd = result.get('VD', 1)    # 0=downward, 1=upward
                
                features[verb_idx] += [force, hand, arm, hd, vd]
                counts[verb_idx] += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    # 计算平均值
    for i in range(6):
        if counts[i] > 0:
            features[i] /= counts[i]
    
    return features.flatten()

# 读取Mimo和RoboBrain数据
mimo_cn = extract_model_features('Mimo-embodied-7B', 'zh')
mimo_en = extract_model_features('Mimo-embodied-7B', 'en')
robobrain_cn = extract_model_features('RoboBrain2.0-7B', 'zh')
robobrain_en = extract_model_features('RoboBrain2.0-7B', 'en')

print("\nMimo CN shape:", mimo_cn.shape)
print("Mimo EN shape:", mimo_en.shape)
print("RoboBrain CN shape:", robobrain_cn.shape)
print("RoboBrain EN shape:", robobrain_en.shape)

# ============================================================
# 所有物种
# ============================================================

species_names = [
    'Human CN', 'Human EN',
    'Mimo CN', 'Mimo EN',
    'RoboBrain CN', 'RoboBrain EN'
]

all_features = np.array([
    human_cn, human_en,
    mimo_cn, mimo_en,
    robobrain_cn, robobrain_en
])

n_species = len(species_names)
print(f"\nTotal species: {n_species}")

# ============================================================
# 计算RSA矩阵
# ============================================================

distance_matrix = squareform(pdist(all_features, metric='euclidean'))

rsa_matrix = np.zeros((n_species, n_species))
for i in range(n_species):
    for j in range(n_species):
        if i == j:
            rsa_matrix[i, j] = 1.0
        else:
            corr, _ = spearmanr(distance_matrix[i], distance_matrix[j])
            rsa_matrix[i, j] = corr

print("\nRSA Matrix:")
print(pd.DataFrame(rsa_matrix, index=species_names, columns=species_names).round(3))

# ============================================================
# Figure 1: VTM - 使用真实模型数据
# ============================================================

print("\nGenerating VTM with real model data...")

fig_vtm, axes_vtm = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': 'polar'})

# 12个动词
cn_verb_names = ['扔', '丢', '抛', '投', '摔', '甩']
en_verb_names = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']
all_verb_names = cn_verb_names + en_verb_names

# 参数顺序：FORCE, HAND, ARM, HD, VD
param_names = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

# 人类数据
human_all = np.concatenate([human_cn.reshape(6, 5), human_en.reshape(6, 5)])
mimo_all = np.concatenate([mimo_cn.reshape(6, 5), mimo_en.reshape(6, 5)])
robobrain_all = np.concatenate([robobrain_cn.reshape(6, 5), robobrain_en.reshape(6, 5)])

n_verbs = 12
sector_angle = 2 * np.pi / n_verbs
point_offsets = np.linspace(-sector_angle * 0.3, sector_angle * 0.3, 5)

# 灰色线条样式
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
color_human = '#808080'
color_mimo = '#0077BB'
color_robobrain = '#EE7733'

for ax_idx, (data, color, title) in enumerate([
    (human_all, color_human, 'Human'),
    (mimo_all, color_mimo, 'Mimo-embodied-7B'),
    (robobrain_all, color_robobrain, 'RoboBrain2.0-7B')
]):
    ax = axes_vtm[ax_idx]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 绘制扇区边界
    for i in range(n_verbs):
        angle = i * sector_angle
        ax.plot([angle, angle], [0, 1.15], color='gray', linewidth=0.5, alpha=0.3)
    
    # 绘制每个动词的折线
    for i in range(n_verbs):
        sector_center = i * sector_angle
        r_values = data[i]  # 5个参数值
        
        thetas = sector_center + point_offsets
        
        if ax_idx == 0:  # Human使用不同线型
            ax.plot(thetas, r_values, 'o', color=color, markersize=4, 
                   markeredgecolor='white', markeredgewidth=0.5, alpha=0.8)
            # 连线
            ax.plot(thetas, r_values, '-', color=color, linewidth=1.5, alpha=0.7)
        else:  # 模型使用实线
            ax.plot(thetas, r_values, 'o-', color=color, linewidth=1.5, 
                   markersize=4, markeredgecolor='white', markeredgewidth=0.5, alpha=0.8)
    
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontweight='bold', fontsize=10, pad=15)
    
    # 添加动词标签
    for i in range(n_verbs):
        angle = i * sector_angle
        ax.text(angle, 1.08, all_verb_names[i], ha='center', va='center', 
               fontsize=6, rotation=np.degrees(angle) - 90)

plt.tight_layout()
plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/vtm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: /home/lulu444/Enactment-experiment/figures/approved/vtm_comparison.png")

# ============================================================
# Figure 2: RSA Heatmap
# ============================================================

fig2, ax2 = ns.create_figure(width=120)

im = ax2.imshow(rsa_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
ax2.set_xticks(range(n_species))
ax2.set_yticks(range(n_species))
ax2.set_xticklabels(species_names, fontsize=6, rotation=45, ha='right')
ax2.set_yticklabels(species_names, fontsize=6)
ax2.set_title('RSA Similarity Matrix', fontweight='bold', fontsize=9, pad=15)

for i in range(n_species):
    for j in range(n_species):
        val = rsa_matrix[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                fontsize=5, fontweight='bold', color=color)

plt.colorbar(im, ax=ax2, shrink=0.8, label='RSA (Spearman ρ)')
fig2.tight_layout()

plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/rsa_heatmap_real.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: /home/lulu444/Enactment-experiment/figures/approved/rsa_heatmap_real.png")

# ============================================================
# Figure 3: RSA Ranking
# ============================================================

fig3, ax3 = ns.create_figure(width=100, height=60)

rsa_to_human_cn = rsa_matrix[0, :]

sorted_idx = np.argsort(rsa_to_human_cn)[::-1]
sorted_names = [species_names[i] for i in sorted_idx]
sorted_scores = [rsa_to_human_cn[i] for i in sorted_idx]

color_map = {
    'Human CN': '#CC3311',
    'Human EN': '#EE7733',
    'Mimo CN': '#0077BB',
    'Mimo EN': '#33BBEE',
    'RoboBrain CN': '#009988',
    'RoboBrain EN': '#EE3377'
}
colors = [color_map[name] for name in sorted_names]

bars = ax3.barh(range(n_species), sorted_scores, color=colors,
                edgecolor='black', linewidth=0.3, height=0.7)

ax3.set_yticks(range(n_species))
ax3.set_yticklabels(sorted_names, fontsize=6)
ax3.set_xlabel('RSA Score (Spearman ρ)', fontsize=7)
ax3.set_title('RSA Ranking: Similarity to Human CN', fontweight='bold', fontsize=9, pad=10)
ax3.set_xlim(-1, 1.1)

for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    ax3.text(score + 0.02 if score >= 0 else score - 0.02, i, 
             f'{score:.3f}', va='center', fontsize=5, ha='left' if score >= 0 else 'right')

ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

fig3.tight_layout()

plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/rsa_ranking_real.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: /home/lulu444/Enactment-experiment/figures/approved/rsa_ranking_real.png")

# ============================================================
# Figure 4: Language Transfer Matrix
# ============================================================

fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))

# Mimo CN
transfer_mimo = np.array([
    [rsa_matrix[2, 0], rsa_matrix[2, 1]],
    [rsa_matrix[3, 0], rsa_matrix[3, 1]]
])

im1 = axes4[0, 0].imshow(transfer_mimo, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
axes4[0, 0].set_xticks([0, 1])
axes4[0, 0].set_xticklabels(['Human CN', 'Human EN'], fontsize=6)
axes4[0, 0].set_yticks([0, 1])
axes4[0, 0].set_yticklabels(['Mimo CN', 'Mimo EN'], fontsize=6)
axes4[0, 0].set_title('Mimo-embodied-7B', fontweight='bold', fontsize=8, pad=5)
for i in range(2):
    for j in range(2):
        val = transfer_mimo[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        axes4[0, 0].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

# RoboBrain CN
transfer_robobrain = np.array([
    [rsa_matrix[4, 0], rsa_matrix[4, 1]],
    [rsa_matrix[5, 0], rsa_matrix[5, 1]]
])

im2 = axes4[0, 1].imshow(transfer_robobrain, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
axes4[0, 1].set_xticks([0, 1])
axes4[0, 1].set_xticklabels(['Human CN', 'Human EN'], fontsize=6)
axes4[0, 1].set_yticks([0, 1])
axes4[0, 1].set_yticklabels(['RoboBrain CN', 'RoboBrain EN'], fontsize=6)
axes4[0, 1].set_title('RoboBrain2.0-7B', fontweight='bold', fontsize=8, pad=5)
for i in range(2):
    for j in range(2):
        val = transfer_robobrain[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        axes4[0, 1].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

# Language Transfer Gap
axes4[1, 0].axis('off')
axes4[1, 1].axis('off')

# 计算语言转移差
mimo_gap_cn = rsa_matrix[2, 0] - rsa_matrix[2, 1]
mimo_gap_en = rsa_matrix[3, 1] - rsa_matrix[3, 0]
robobrain_gap_cn = rsa_matrix[4, 0] - rsa_matrix[4, 1]
robobrain_gap_en = rsa_matrix[5, 1] - rsa_matrix[5, 0]

gap_text = f"""Language Transfer Gap:

Mimo-embodied-7B:
  Model CN: {mimo_gap_cn:.3f}
  Model EN: {mimo_gap_en:.3f}

RoboBrain2.0-7B:
  Model CN: {robobrain_gap_cn:.3f}
  Model EN: {robobrain_gap_en:.3f}"""

axes4[1, 0].text(0.5, 0.5, gap_text, ha='center', va='center', fontsize=8,
                transform=axes4[1, 0].transAxes, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/language_transfer_real.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: /home/lulu444/Enactment-experiment/figures/approved/language_transfer_real.png")

# ============================================================
# 打印统计信息
# ============================================================

print("\n" + "="*60)
print("Real Data RSA Analysis")
print("="*60)

print("\nRSA with Human CN:")
for i, name in enumerate(species_names):
    print(f"  {name:15s}: {rsa_matrix[0, i]:.3f}")

print("\nLanguage Transfer Gap:")
print(f"  Mimo CN:   {mimo_gap_cn:.3f}")
print(f"  Mimo EN:   {mimo_gap_en:.3f}")
print(f"  RoboBrain CN: {robobrain_gap_cn:.3f}")
print(f"  RoboBrain EN: {robobrain_gap_en:.3f}")
