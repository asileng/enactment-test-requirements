"""
从Table 1计算所有5个参数的pairwise显著性检验
使用正确的统计方法：
- FORCE, HAND: 使用Mean和SD做t-test
- ARM, HD, VD: 使用频次数据做chi-square或proportion test
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from nature_style import NatureStyle

ns = NatureStyle()

# 读取Table 1
df = pd.read_csv('table1_descriptive_statistics.csv')

# 选择中文和英文
cn_verbs = df[df['Language'] == 'Chinese'].head(6).copy().reset_index(drop=True)
en_verbs = df[df['Language'] == 'English'].head(6).copy().reset_index(drop=True)

print("中文动词:", cn_verbs['Verb'].tolist())
print("英文动词:", en_verbs['Verb'].tolist())

N = 30  # 每组样本量

def pairwise_ttest(mean1, sd1, mean2, sd2, n=N):
    """使用Mean和SD做两独立样本t-test"""
    se1 = sd1 / np.sqrt(n)
    se2 = sd2 / np.sqrt(n)
    se_diff = np.sqrt(se1**2 + se2**2)
    t_stat = (mean1 - mean2) / se_diff
    df = 2 * n - 2
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return p_value

def pairwise_proportion_test(count1, count2, n=N):
    """使用频次数据做proportion test (chi-square)"""
    # count1和count2是"是"的频次
    p1 = count1 / n
    p2 = count2 / n
    p_pool = (count1 + count2) / (2 * n)
    
    if p_pool == 0 or p_pool == 1:
        return 1.0
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n + 1/n))
    if se == 0:
        return 1.0
    
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return p_value

def get_significance(p_value):
    """将p值转换为显著性标记"""
    if p_value < 0.001:
        return 3  # ***
    elif p_value < 0.01:
        return 2  # **
    elif p_value < 0.05:
        return 1  # *
    else:
        return 0  # ns

def compute_pairwise_for_language(verbs_df, lang_name):
    """为一种语言计算所有5个参数的pairwise显著性"""
    n_verbs = len(verbs_df)
    pairs = list(combinations(range(n_verbs), 2))
    
    # 结果矩阵: 15 pairs × 5 parameters
    results = np.zeros((len(pairs), 5))
    p_values = {}
    
    for idx, (i, j) in enumerate(pairs):
        v1 = verbs_df.iloc[i]
        v2 = verbs_df.iloc[j]
        pair_name = f"{v1['Verb']} vs {v2['Verb']}"
        
        # FORCE: t-test
        p_force = pairwise_ttest(v1['FORCE Mean'], v1['FORCE SD'], 
                                 v2['FORCE Mean'], v2['FORCE SD'])
        
        # HAND: t-test
        p_hand = pairwise_ttest(v1['HAND Mean'], v1['HAND SD'],
                                v2['HAND Mean'], v2['HAND SD'])
        
        # ARM: proportion test (straight vs bend)
        p_arm = pairwise_proportion_test(v1['ARM straight'], v2['ARM straight'])
        
        # HD: proportion test (forward vs sidewise)
        p_hd = pairwise_proportion_test(v1['HD forward'], v2['HD forward'])
        
        # VD: proportion test (upward vs downward)
        p_vd = pairwise_proportion_test(v1['VD upward'], v2['VD upward'])
        
        results[idx, 0] = get_significance(p_force)
        results[idx, 1] = get_significance(p_hand)
        results[idx, 2] = get_significance(p_arm)
        results[idx, 3] = get_significance(p_hd)
        results[idx, 4] = get_significance(p_vd)
        
        p_values[pair_name] = {
            'FORCE': p_force,
            'HAND': p_hand,
            'ARM': p_arm,
            'HD': p_hd,
            'VD': p_vd
        }
    
    return results, p_values, pairs

# 计算中文和英文
cn_results, cn_pvalues, cn_pairs = compute_pairwise_for_language(cn_verbs, 'Chinese')
en_results, en_pvalues, en_pairs = compute_pairwise_for_language(en_verbs, 'English')

# 创建动词对标签
cn_pair_labels = [f"{cn_verbs.iloc[v1]['Verb']} vs {cn_verbs.iloc[v2]['Verb']}" for v1, v2 in cn_pairs]
en_pair_labels = [f"{en_verbs.iloc[v1]['Verb']} vs {en_verbs.iloc[v2]['Verb']}" for v1, v2 in en_pairs]

params = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

# ============================================================
# 保存长格式CSV
# ============================================================

def create_long_format(verbs_df, results, pairs, lang_name):
    """创建长格式DataFrame"""
    rows = []
    for idx, (i, j) in enumerate(pairs):
        v1 = verbs_df.iloc[i]['Verb']
        v2 = verbs_df.iloc[j]['Verb']
        for p_idx, param in enumerate(params):
            rows.append({
                'language': lang_name,
                'verb1': v1,
                'verb2': v2,
                'pair': f"{v1} vs {v2}",
                'parameter': param,
                'significance': int(results[idx, p_idx])
            })
    return pd.DataFrame(rows)

cn_long = create_long_format(cn_verbs, cn_results, cn_pairs, 'Chinese')
en_long = create_long_format(en_verbs, en_results, en_pairs, 'English')

cn_long.to_csv('pairwise_statistical_chinese.csv', index=False)
en_long.to_csv('pairwise_statistical_english.csv', index=False)
print("\nSaved: pairwise_statistical_chinese.csv")
print("Saved: pairwise_statistical_english.csv")

# ============================================================
# Figure 1: Pairwise significance matrix
# ============================================================

fig, axes = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

# Chinese heatmap
im1 = axes[0].imshow(cn_results, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[0].set_xticks(range(5))
axes[0].set_yticks(range(15))
axes[0].set_xticklabels(params, fontsize=6, fontweight='bold')
axes[0].set_yticklabels(cn_pair_labels, fontsize=4)
axes[0].set_title('Chinese', fontweight='bold', fontsize=8, pad=10)

for i in range(15):
    for j in range(5):
        val = int(cn_results[i, j])
        if val > 0:
            text = '*' * val
            axes[0].text(j, i, text, ha='center', va='center', 
                        fontsize=5, color='white' if val > 1 else 'black')

# English heatmap
im2 = axes[1].imshow(en_results, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[1].set_xticks(range(5))
axes[1].set_yticks(range(15))
axes[1].set_xticklabels(params, fontsize=6, fontweight='bold')
axes[1].set_yticklabels(en_pair_labels, fontsize=4)
axes[1].set_title('English', fontweight='bold', fontsize=8, pad=10)

for i in range(15):
    for j in range(5):
        val = int(en_results[i, j])
        if val > 0:
            text = '*' * val
            axes[1].text(j, i, text, ha='center', va='center', 
                        fontsize=5, color='white' if val > 1 else 'black')

fig.colorbar(im1, ax=axes[0], shrink=0.8)
fig.colorbar(im2, ax=axes[1], shrink=0.8)

fig.suptitle('Pairwise Significance Matrix (Statistical Tests from Table 1)',
             fontweight='bold', fontsize=9, y=1.02)
fig.tight_layout()

ns.save_figure(fig, 'nature_fig8_statistical_matrix.png')
ns.save_figure(fig, 'nature_fig8_statistical_matrix.pdf')
print("Saved: nature_fig8_statistical_matrix.png/pdf")

# ============================================================
# Figure 2: Parameter discriminability
# ============================================================

fig2, axes2 = ns.create_figure_with_axes(width=183, nrows=1, ncols=2)

cn_counts = [np.sum(cn_results[:, j] >= 1) for j in range(5)]
en_counts = [np.sum(en_results[:, j] >= 1) for j in range(5)]
cn_high = [np.sum(cn_results[:, j] >= 3) for j in range(5)]
en_high = [np.sum(en_results[:, j] >= 3) for j in range(5)]

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

fig2.suptitle('Parameter Discriminability (Statistical Tests)',
              fontweight='bold', fontsize=9, y=1.02)
fig2.tight_layout()

ns.save_figure(fig2, 'nature_fig9_statistical_discriminability.png')
ns.save_figure(fig2, 'nature_fig9_statistical_discriminability.pdf')
print("Saved: nature_fig9_statistical_discriminability.png/pdf")

# ============================================================
# 打印统计结果
# ============================================================

print("\n" + "="*60)
print("STATISTICAL RESULTS (from Table 1)")
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
print(f"Chinese best discriminators: ", end="")
cn_sorted = sorted(zip(params, cn_counts), key=lambda x: -x[1])
print(", ".join([f"{p}({c})" for p, c in cn_sorted]))

print(f"English best discriminators: ", end="")
en_sorted = sorted(zip(params, en_counts), key=lambda x: -x[1])
print(", ".join([f"{p}({c})" for p, c in en_sorted]))
