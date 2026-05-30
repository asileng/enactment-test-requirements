"""
四个分析：人类基准 + Mock模型数据
正确理解：每个物种 = 6动词 × 5参数 = 30维特征向量
RSA比较的是物种间的特征相似度
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from nature_style import NatureStyle

ns = NatureStyle()

# 读取数据
df = pd.read_csv('table1_descriptive_statistics.csv')

# 选择中文和英文
cn_verbs = df[df['Language'] == 'Chinese'].head(6).copy().reset_index(drop=True)
en_verbs = df[df['Language'] == 'English'].head(6).copy().reset_index(drop=True)

# ============================================================
# 提取特征向量（每个物种 = 30维）
# ============================================================

def extract_feature_vector(verbs_df):
    """提取6动词×5参数 = 30维特征向量"""
    features = []
    for _, row in verbs_df.iterrows():
        # FORCE: 归一化到0-1
        force = (row['FORCE Mean'] - 1) / 4
        
        # HAND: 归一化到0-1
        hand = row['HAND Mean'] / 12
        
        # ARM: straight比例
        arm_total = row['ARM bend'] + row['ARM straight']
        arm = row['ARM straight'] / arm_total if arm_total > 0 else 0.5
        
        # HD: forward比例
        hd_total = row['HD forward'] + row['HD sidewise']
        hd = row['HD forward'] / hd_total if hd_total > 0 else 0.5
        
        # VD: upward比例
        vd_total = row['VD upward'] + row['VD downward']
        vd = row['VD upward'] / vd_total if vd_total > 0 else 0.5
        
        features.extend([force, hand, arm, hd, vd])
    
    return np.array(features)  # 30维

# 人类数据
human_cn = extract_feature_vector(cn_verbs)
human_en = extract_feature_vector(en_verbs)

print("Human CN feature vector shape:", human_cn.shape)
print("Human EN feature vector shape:", human_en.shape)

# ============================================================
# Mock模型数据
# ============================================================

np.random.seed(42)

# Mock LLM模型（与人类有差异）
llm_cn = human_cn + np.random.normal(0, 0.15, 30)
llm_en = human_en + np.random.normal(0, 0.15, 30)

# Mock VLM模型（与人类更接近）
vlm_cn = human_cn + np.random.normal(0, 0.08, 30)
vlm_en = human_en + np.random.normal(0, 0.08, 30)

# Mock VLA模型（介于LLM和VLM之间）
vla_cn = human_cn + np.random.normal(0, 0.12, 30)
vla_en = human_en + np.random.normal(0, 0.12, 30)

# 所有物种
species_names = [
    'Human CN', 'Human EN',
    'LLM CN', 'LLM EN',
    'VLM CN', 'VLM EN',
    'VLA CN', 'VLA EN'
]

all_features = np.array([
    human_cn, human_en,
    llm_cn, llm_en,
    vlm_cn, vlm_en,
    vla_cn, vla_en
])

n_species = len(species_names)
print(f"\nTotal species: {n_species}")
print("Species:", species_names)

# ============================================================
# 计算RSA矩阵（基于欧氏距离的相关性）
# ============================================================

# 计算每个物种的距离向量（与其他所有物种的距离）
distance_matrix = squareform(pdist(all_features, metric='euclidean'))

# RSA: 计算距离向量之间的Spearman相关
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
# Figure 1: VTM (已有)
# ============================================================
print("\nVTM: 使用已有的 polar_sector_visualization.png/pdf")

# ============================================================
# Figure 2: RSA Heatmap
# ============================================================

fig2, ax2 = ns.create_figure(width=140)

im = ax2.imshow(rsa_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
ax2.set_xticks(range(n_species))
ax2.set_yticks(range(n_species))
ax2.set_xticklabels(species_names, fontsize=5, rotation=45, ha='right')
ax2.set_yticklabels(species_names, fontsize=5)
ax2.set_title('RSA Similarity Matrix', fontweight='bold', fontsize=8, pad=15)

# 添加数值标注
for i in range(n_species):
    for j in range(n_species):
        val = rsa_matrix[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', 
                fontsize=4, fontweight='bold', color=color)

plt.colorbar(im, ax=ax2, shrink=0.8, label='RSA (Spearman ρ)')
fig2.tight_layout()

ns.save_figure(fig2, 'analysis_v2_fig2_rsa_heatmap.png')
ns.save_figure(fig2, 'analysis_v2_fig2_rsa_heatmap.pdf')
print("\nSaved: analysis_v2_fig2_rsa_heatmap.png/pdf")

# ============================================================
# Figure 3: RSA Ranking（与Human CN的相似度）
# ============================================================

fig3, ax3 = ns.create_figure(width=89, height=60)

# 与Human CN的RSA
rsa_to_human_cn = rsa_matrix[0, :]  # Human CN行

# 排序（按RSA降序）
sorted_idx = np.argsort(rsa_to_human_cn)[::-1]
sorted_names = [species_names[i] for i in sorted_idx]
sorted_scores = [rsa_to_human_cn[i] for i in sorted_idx]

# 颜色
color_map = {
    'Human CN': ns.COLORS['red'],
    'Human EN': ns.COLORS['orange'],
    'LLM CN': ns.COLORS['blue'],
    'LLM EN': ns.COLORS['cyan'],
    'VLM CN': ns.COLORS['teal'],
    'VLM EN': ns.COLORS['magenta'],
    'VLA CN': '#999999',
    'VLA EN': ns.COLORS['grey']
}
colors = [color_map[name] for name in sorted_names]

# 绘制条形图
bars = ax3.barh(range(n_species), sorted_scores, color=colors,
                edgecolor='black', linewidth=0.3, height=0.7)

ax3.set_yticks(range(n_species))
ax3.set_yticklabels(sorted_names, fontsize=5)
ax3.set_xlabel('RSA Score (Spearman ρ)', fontsize=6)
ax3.set_title('RSA Ranking: Similarity to Human CN', fontweight='bold', fontsize=8, pad=10)
ax3.set_xlim(-0.5, 1.1)

# 添加数值标注
for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    ax3.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=5)

# 添加参考线
ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax3.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

fig3.tight_layout()

ns.save_figure(fig3, 'analysis_v2_fig3_rsa_ranking.png')
ns.save_figure(fig3, 'analysis_v2_fig3_rsa_ranking.pdf')
print("Saved: analysis_v2_fig3_rsa_ranking.png/pdf")

# ============================================================
# Figure 4: Language Transfer Matrix
# ============================================================

fig4, axes4 = ns.create_figure_with_axes(width=183, nrows=2, ncols=3)

# 第一行：Model CN的语言转移
# 子图1: LLM CN
transfer_llm_cn = np.array([
    [rsa_matrix[2, 0], rsa_matrix[2, 1]]  # LLM CN ↔ Human CN, LLM CN ↔ Human EN
])

im1 = axes4[0, 0].imshow(transfer_llm_cn, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[0, 0].set_xticks([0, 1])
axes4[0, 0].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[0, 0].set_yticks([0])
axes4[0, 0].set_yticklabels(['LLM CN'], fontsize=5)
axes4[0, 0].set_title('LLM', fontweight='bold', fontsize=7, pad=5)
for j in range(2):
    val = transfer_llm_cn[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[0, 0].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 子图2: VLM CN
transfer_vlm_cn = np.array([
    [rsa_matrix[4, 0], rsa_matrix[4, 1]]
])
im2 = axes4[0, 1].imshow(transfer_vlm_cn, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[0, 1].set_xticks([0, 1])
axes4[0, 1].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[0, 1].set_yticks([0])
axes4[0, 1].set_yticklabels(['VLM CN'], fontsize=5)
axes4[0, 1].set_title('VLM', fontweight='bold', fontsize=7, pad=5)
for j in range(2):
    val = transfer_vlm_cn[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[0, 1].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 子图3: VLA CN
transfer_vla_cn = np.array([
    [rsa_matrix[6, 0], rsa_matrix[6, 1]]
])
im3 = axes4[0, 2].imshow(transfer_vla_cn, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[0, 2].set_xticks([0, 1])
axes4[0, 2].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[0, 2].set_yticks([0])
axes4[0, 2].set_yticklabels(['VLA CN'], fontsize=5)
axes4[0, 2].set_title('VLA', fontweight='bold', fontsize=7, pad=5)
for j in range(2):
    val = transfer_vla_cn[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[0, 2].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 第二行：Model EN的语言转移
# 子图4: LLM EN
transfer_llm_en = np.array([
    [rsa_matrix[3, 0], rsa_matrix[3, 1]]  # LLM EN ↔ Human CN, LLM EN ↔ Human EN
])
im4 = axes4[1, 0].imshow(transfer_llm_en, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[1, 0].set_xticks([0, 1])
axes4[1, 0].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[1, 0].set_yticks([0])
axes4[1, 0].set_yticklabels(['LLM EN'], fontsize=5)
for j in range(2):
    val = transfer_llm_en[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[1, 0].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 子图5: VLM EN
transfer_vlm_en = np.array([
    [rsa_matrix[5, 0], rsa_matrix[5, 1]]
])
im5 = axes4[1, 1].imshow(transfer_vlm_en, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[1, 1].set_xticks([0, 1])
axes4[1, 1].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[1, 1].set_yticks([0])
axes4[1, 1].set_yticklabels(['VLM EN'], fontsize=5)
for j in range(2):
    val = transfer_vlm_en[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[1, 1].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 子图6: VLA EN
transfer_vla_en = np.array([
    [rsa_matrix[7, 0], rsa_matrix[7, 1]]
])
im6 = axes4[1, 2].imshow(transfer_vla_en, cmap='RdYlBu_r', vmin=-0.8, vmax=1, aspect='auto')
axes4[1, 2].set_xticks([0, 1])
axes4[1, 2].set_xticklabels(['Human CN', 'Human EN'], fontsize=5)
axes4[1, 2].set_yticks([0])
axes4[1, 2].set_yticklabels(['VLA EN'], fontsize=5)
for j in range(2):
    val = transfer_vla_en[0, j]
    color = 'white' if abs(val) > 0.5 else 'black'
    axes4[1, 2].text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# 共享colorbar
fig4.subplots_adjust(right=0.85)
cbar_ax = fig4.add_axes([0.88, 0.15, 0.02, 0.7])
fig4.colorbar(im6, cax=cbar_ax, label='RSA (Spearman ρ)')

fig4.suptitle('Language Transfer Matrix', fontweight='bold', fontsize=9, y=1.02)
fig4.tight_layout(rect=[0, 0, 0.85, 1])

ns.save_figure(fig4, 'analysis_v2_fig4_language_transfer.png')
ns.save_figure(fig4, 'analysis_v2_fig4_language_transfer.pdf')
print("Saved: analysis_v2_fig4_language_transfer.png/pdf")

# ============================================================
# 打印统计信息
# ============================================================

print("\n" + "="*60)
print("RSA Analysis Results")
print("="*60)

print("\nRSA with Human CN:")
for i, name in enumerate(species_names):
    print(f"  {name:12s}: {rsa_matrix[0, i]:.3f}")

print("\nKey Metrics:")
# 语言内RSA（模型CN vs Human CN）
print("Language-specific RSA (Model CN ↔ Human CN):")
print(f"  LLM:  {rsa_matrix[2, 0]:.3f}")
print(f"  VLM:  {rsa_matrix[4, 0]:.3f}")
print(f"  VLA:  {rsa_matrix[6, 0]:.3f}")

# 跨语言RSA（模型CN vs Human EN）
print("\nCross-lingual RSA (Model CN ↔ Human EN):")
print(f"  LLM:  {rsa_matrix[2, 1]:.3f}")
print(f"  VLM:  {rsa_matrix[4, 1]:.3f}")
print(f"  VLA:  {rsa_matrix[6, 1]:.3f}")

# 语言转移差异
print("Language Transfer Gap:")
print("  For Model CN: RSA(Model CN, Human CN) - RSA(Model CN, Human EN)")
print(f"    LLM:  {rsa_matrix[2, 0] - rsa_matrix[2, 1]:.3f}")
print(f"    VLM:  {rsa_matrix[4, 0] - rsa_matrix[4, 1]:.3f}")
print(f"    VLA:  {rsa_matrix[6, 0] - rsa_matrix[6, 1]:.3f}")

print("\n  For Model EN: RSA(Model EN, Human EN) - RSA(Model EN, Human CN)")
print(f"    LLM:  {rsa_matrix[3, 1] - rsa_matrix[3, 0]:.3f}")
print(f"    VLM:  {rsa_matrix[5, 1] - rsa_matrix[5, 0]:.3f}")
print(f"    VLA:  {rsa_matrix[7, 1] - rsa_matrix[7, 0]:.3f}")
