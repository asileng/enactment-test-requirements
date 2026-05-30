"""
分析模型对中文动词的区分能力
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import numpy as np
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
from itertools import combinations
from nature_style import NatureStyle

ns = NatureStyle()

# ============================================================
# 读取数据
# ============================================================

verb_names = ['扔', '丢', '抛', '投', '摔', '甩']
param_names = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

def read_model_data(model_name):
    """读取模型中文数据"""
    files = glob.glob(f'pilot_results/{model_name}/task1_zh/*.json')
    
    data_dict = {}
    for f in sorted(files):
        with open(f, 'r') as fh:
            data = json.load(fh)
        
        if data.get('is_valid', False):
            verb = data.get('verb', '')
            result = data.get('parsed_result', {})
            if verb and result:
                data_dict[verb] = result
    
    # 构建特征矩阵
    features = np.zeros((6, 5))
    for i, verb in enumerate(verb_names):
        if verb in data_dict:
            r = data_dict[verb]
            features[i] = [r.get('FORCE', 0), r.get('HAND', 0), r.get('ARM', 0), r.get('HD', 0), r.get('VD', 0)]
    
    return features

# 读取人类基准
df_human = pd.read_csv('table1_descriptive_statistics.csv')
cn_human = df_human[df_human['Language'] == 'Chinese'].head(6)

human_features = np.zeros((6, 5))
for i, (_, row) in enumerate(cn_human.iterrows()):
    force = row['FORCE Mean']
    hand = row['HAND Mean']
    arm_total = row['ARM bend'] + row['ARM straight']
    arm = row['ARM straight'] / arm_total if arm_total > 0 else 0.5
    hd_total = row['HD forward'] + row['HD sidewise']
    hd = row['HD forward'] / hd_total if hd_total > 0 else 0.5
    vd_total = row['VD upward'] + row['VD downward']
    vd = row['VD upward'] / vd_total if vd_total > 0 else 0.5
    human_features[i] = [force, hand, arm * 5, hd * 5, vd * 5]  # 缩放到0-5范围

# 读取模型数据
mimo_features = read_model_data('Mimo-embodied-7B')
robobrain_features = read_model_data('RoboBrain2.0-7B')

# ============================================================
# 打印数据对比
# ============================================================

print("="*70)
print("Chinese Verb Features Comparison (6 verbs × 5 parameters)")
print("="*70)

print(f"\n{'Verb':6s} | {'Human':>20s} | {'Mimo':>20s} | {'RoboBrain':>20s}")
print(f"{'':6s} | {'FORCE HAND ARM HD VD':>20s} | {'FORCE HAND ARM HD VD':>20s} | {'FORCE HAND ARM HD VD':>20s}")
print("-" * 90)

for i, verb in enumerate(verb_names):
    h = human_features[i]
    m = mimo_features[i]
    r = robobrain_features[i]
    print(f"{verb:6s} | {h[0]:5.1f} {h[1]:5.1f} {h[2]:5.2f} {h[3]:5.2f} {h[4]:5.2f} | {m[0]:5.1f} {m[1]:5.1f} {m[2]:5.2f} {m[3]:5.2f} {m[4]:5.2f} | {r[0]:5.1f} {r[1]:5.1f} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f}")

# ============================================================
# Pairwise差异分析
# ============================================================

print("\n" + "="*70)
print("Pairwise Analysis: Which verbs are most different?")
print("="*70)

def compute_pairwise_diffs(features, verb_names, param_names):
    """计算pairwise差异"""
    pairs = list(combinations(range(6), 2))
    results = []
    
    for i, j in pairs:
        total_diff = 0
        param_diffs = {}
        for p_idx, param in enumerate(param_names):
            diff = abs(features[i, p_idx] - features[j, p_idx])
            param_diffs[param] = diff
            total_diff += diff
        results.append({
            'pair': f"{verb_names[i]} vs {verb_names[j]}",
            'total_diff': total_diff,
            **param_diffs
        })
    
    return pd.DataFrame(results)

# Mimo pairwise
df_mimo = compute_pairwise_diffs(mimo_features, verb_names, param_names)
df_mimo = df_mimo.sort_values('total_diff', ascending=False)

print("\nMimo-embodied-7B: Top 10 most different verb pairs:")
print(df_mimo.head(10).to_string(index=False))

# RoboBrain pairwise
df_rb = compute_pairwise_diffs(robobrain_features, verb_names, param_names)
df_rb = df_rb.sort_values('total_diff', ascending=False)

print("\nRoboBrain2.0-7B: Top 10 most different verb pairs:")
print(df_rb.head(10).to_string(index=False))

# ============================================================
# 可视化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 参数范围
param_ranges = {
    'FORCE': (1, 5),
    'HAND': (0, 12),
    'ARM': (0, 1),
    'HD': (0, 1),
    'VD': (0, 1)
}

x = np.arange(6)
width = 0.15

# Human
ax = axes[0]
for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, human_features[:, p_idx], width, label=param)
ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('Human (Chinese)', fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend(fontsize=8)
ax.set_ylim(0, 12)

# Mimo
ax = axes[1]
for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, mimo_features[:, p_idx], width, label=param)
ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('Mimo-embodied-7B', fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend(fontsize=8)
ax.set_ylim(0, 12)

# RoboBrain
ax = axes[2]
for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, robobrain_features[:, p_idx], width, label=param)
ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('RoboBrain2.0-7B', fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend(fontsize=8)
ax.set_ylim(0, 12)

plt.tight_layout()
plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/chinese_verb_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: figures/approved/chinese_verb_comparison.png")

# ============================================================
# 计算方差分析
# ============================================================

print("\n" + "="*70)
print("Variance Analysis: How much do verbs differ within each species?")
print("="*70)

for name, features in [('Human', human_features), ('Mimo', mimo_features), ('RoboBrain', robobrain_features)]:
    variance = np.var(features, axis=0)
    print(f"\n{name}:")
    for p_idx, param in enumerate(param_names):
        print(f"  {param:5s}: variance = {variance[p_idx]:.3f}")
    print(f"  Total variance: {variance.sum():.3f}")
