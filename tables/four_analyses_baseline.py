"""
四个分析的人类基准数据可视化
1. VTM (Verb Trajectory Map) - 极坐标行为轨迹
2. RSA Heatmap - 表征相似性分析热图
3. RSA Ranking - 表征相似性排序
4. Language Transfer Matrix - 语言迁移矩阵
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

print("中文动词:", cn_verbs['Verb'].tolist())
print("英文动词:", en_verbs['Verb'].tolist())

# ============================================================
# 提取5个参数的特征向量
# ============================================================

def extract_features(verbs_df):
    """从DataFrame提取5维特征向量"""
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
        
        features.append([force, hand, arm, hd, vd])
    
    return np.array(features)

cn_features = extract_features(cn_verbs)
en_features = extract_features(en_verbs)

# ============================================================
# Figure 1: VTM (Verb Trajectory Map) - 已有
# ============================================================
print("\nVTM: 使用已有的 polar_sector_visualization.png/pdf")

# ============================================================
# Figure 2: RSA Heatmap
# ============================================================

def compute_rsa_matrix(features_list, labels):
    """计算RSA矩阵（基于Spearman相关）"""
    n = len(features_list)
    rsa_matrix = np.zeros((n, n))
    
    # 计算每个组的特征向量之间的距离
    for i in range(n):
        for j in range(n):
            if i == j:
                rsa_matrix[i, j] = 1.0
            else:
                # 计算两个特征矩阵之间的相关性
                # 将每个组的6个动词×5个参数展平
                vec_i = features_list[i].flatten()
                vec_j = features_list[j].flatten()
                
                # 计算Spearman相关
                corr, _ = spearmanr(vec_i, vec_j)
                rsa_matrix[i, j] = corr
    
    return rsa_matrix

# 人类数据：Human CN和Human EN
# 为了RSA，我们需要比较动词间的距离结构
# 使用相同的6个动词位置（12个位置，但中英文动词不同）

# 方法：比较动词间的相对距离模式
def compute_distance_structure(features):
    """计算动词间的距离结构（欧氏距离的上三角）"""
    dists = squareform(pdist(features, metric='euclidean'))
    # 取上三角（不包括对角线）
    return dists[np.triu_indices_from(dists, k=1)]

# Human CN的距离结构
cn_dist_struct = compute_distance_structure(cn_features)
# Human EN的距离结构
en_dist_struct = compute_distance_structure(en_features)

# 计算RSA（两个距离结构之间的相关）
rsa_cn_en, _ = spearmanr(cn_dist_struct, en_dist_struct)
print(f"\nRSA(Human CN, Human EN) = {rsa_cn_en:.3f}")

# 创建RSA热图（基准数据只有2×2）
fig2, ax2 = ns.create_figure(width=89)

# 对于基准数据，我们展示：
# 1. Human CN内部一致性（自相关=1）
# 2. Human EN内部一致性（自相关=1）
# 3. Human CN与Human EN之间的RSA
rsa_data = np.array([[1.0, rsa_cn_en],
                     [rsa_cn_en, 1.0]])

im = ax2.imshow(rsa_data, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Human CN', 'Human EN'], fontsize=6)
ax2.set_yticklabels(['Human CN', 'Human EN'], fontsize=6)
ax2.set_title('RSA Similarity Matrix', fontweight='bold', fontsize=8, pad=10)

# 添加数值标注
for i in range(2):
    for j in range(2):
        val = rsa_data[i, j]
        color = 'white' if val < 0.5 else 'black'
        ax2.text(j, i, f'{val:.3f}', ha='center', va='center', 
                fontsize=7, fontweight='bold', color=color)

plt.colorbar(im, ax=ax2, shrink=0.8, label='RSA (Spearman ρ)')
fig2.tight_layout()

ns.save_figure(fig2, 'analysis_fig2_rsa_heatmap.png')
ns.save_figure(fig2, 'analysis_fig2_rsa_heatmap.pdf')
print("Saved: analysis_fig2_rsa_heatmap.png/pdf")

# ============================================================
# Figure 3: RSA Ranking
# ============================================================

fig3, ax3 = ns.create_figure(width=89)

# 基准数据：只有Human CN和Human EN
# RSA ranking显示与人类的相似度
species = ['Human CN', 'Human EN']
rsa_scores = [1.0, rsa_cn_en]  # Human CN与自己的RSA=1，与Human EN的RSA

# 排序（按RSA降序）
sorted_idx = np.argsort(rsa_scores)[::-1]
sorted_species = [species[i] for i in sorted_idx]
sorted_scores = [rsa_scores[i] for i in sorted_idx]

# 绘制条形图
colors = [ns.COLORS['red'], ns.COLORS['blue']]
bars = ax3.barh(range(2), sorted_scores, color=[colors[i] for i in sorted_idx],
                edgecolor='black', linewidth=0.3, height=0.6)

ax3.set_yticks(range(2))
ax3.set_yticklabels(sorted_species, fontsize=6)
ax3.set_xlabel('RSA Score (Spearman ρ)', fontsize=6)
ax3.set_title('RSA Ranking: Similarity to Human', fontweight='bold', fontsize=8, pad=10)
ax3.set_xlim(0, 1.1)

# 添加数值标注
for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    ax3.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=6)

# 添加参考线
ax3.axvline(x=rsa_cn_en, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax3.text(rsa_cn_en + 0.01, 1.5, f'Cross-lingual RSA\n= {rsa_cn_en:.3f}', 
         fontsize=5, color='gray')

fig3.tight_layout()

ns.save_figure(fig3, 'analysis_fig3_rsa_ranking.png')
ns.save_figure(fig3, 'analysis_fig3_rsa_ranking.pdf')
print("Saved: analysis_fig3_rsa_ranking.png/pdf")

# ============================================================
# Figure 4: Language Transfer Matrix
# ============================================================

fig4, ax4 = ns.create_figure(width=89)

# 语言迁移矩阵：
# 展示 Model CN ↔ Human CN 与 Model CN ↔ Human EN 的差异
# 对于基准数据，我们展示：
# - Human CN ↔ Human CN (自身)
# - Human CN ↔ Human EN (跨语言)
# - Human EN ↔ Human CN (跨语言)
# - Human EN ↔ Human EN (自身)

# 计算特征相似度（余弦相似度）
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# 计算各组之间的相似度
# Human CN的平均特征
cn_mean = cn_features.mean(axis=0)
en_mean = en_features.mean(axis=0)

sim_cn_cn = cosine_similarity(cn_mean, cn_mean)  # = 1.0
sim_cn_en = cosine_similarity(cn_mean, en_mean)
sim_en_cn = sim_cn_en
sim_en_en = cosine_similarity(en_mean, en_mean)  # = 1.0

# 语言转移矩阵
transfer_data = np.array([[sim_cn_cn, sim_cn_en],
                          [sim_en_cn, sim_en_en]])

im4 = ax4.imshow(transfer_data, cmap='RdYlBu_r', vmin=0.5, vmax=1, aspect='equal')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Human CN', 'Human EN'], fontsize=6)
ax4.set_yticklabels(['Human CN', 'Human EN'], fontsize=6)
ax4.set_title('Language Transfer Matrix', fontweight='bold', fontsize=8, pad=10)

# 添加数值标注
for i in range(2):
    for j in range(2):
        val = transfer_data[i, j]
        color = 'white' if val < 0.7 else 'black'
        ax4.text(j, i, f'{val:.3f}', ha='center', va='center', 
                fontsize=7, fontweight='bold', color=color)

plt.colorbar(im4, ax=ax4, shrink=0.8, label='Cosine Similarity')
fig4.tight_layout()

ns.save_figure(fig4, 'analysis_fig4_language_transfer.png')
ns.save_figure(fig4, 'analysis_fig4_language_transfer.pdf')
print("Saved: analysis_fig4_language_transfer.png/pdf")

# ============================================================
# 打印统计信息
# ============================================================

print("\n" + "="*60)
print("四个分析的基准数据统计")
print("="*60)

print("\n1. VTM (Verb Trajectory Map)")
print("   - 12个扇区（6中文 + 6英文）")
print("   - 每个扇区5个点（FORCE, HAND, ARM, HD, VD）")
print("   - 文件: polar_sector_visualization.png/pdf")

print("\n2. RSA Heatmap")
print(f"   - RSA(Human CN, Human EN) = {rsa_cn_en:.3f}")
print("   - 表示中英文人类行为结构的相似度")
print("   - 文件: analysis_fig2_rsa_heatmap.png/pdf")

print("\n3. RSA Ranking")
print("   - Human CN: 1.000 (self)")
print(f"   - Human EN: {rsa_cn_en:.3f} (cross-lingual)")
print("   - 文件: analysis_fig3_rsa_ranking.png/pdf")

print("\n4. Language Transfer Matrix")
print(f"   - Human CN ↔ Human CN: {sim_cn_cn:.3f}")
print(f"   - Human CN ↔ Human EN: {sim_cn_en:.3f}")
print(f"   - Human EN ↔ Human CN: {sim_en_cn:.3f}")
print(f"   - Human EN ↔ Human EN: {sim_en_en:.3f}")
print("   - 文件: analysis_fig4_language_transfer.png/pdf")

print("\n" + "="*60)
print("关键发现")
print("="*60)
print(f"跨语言RSA = {rsa_cn_en:.3f}，说明中英文人类行为结构")
if rsa_cn_en > 0.7:
    print("具有较高的相似性（>0.7），表明存在跨语言的行为共性")
elif rsa_cn_en > 0.4:
    print("具有中等相似性（0.4-0.7），表明存在部分跨语言共性")
else:
    print("相似性较低（<0.4），表明中英文行为结构差异较大")
