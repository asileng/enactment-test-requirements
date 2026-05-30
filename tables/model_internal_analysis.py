"""
分析模型内部的动词区分能力
检查模型能否区分6个中文动词和6个英文动词
"""
import sys
sys.path.insert(0, '/home/lulu444/.claude/skills/nature-viz/scripts')

import numpy as np
import pandas as pd
import json
import glob
from scipy import stats
from itertools import combinations
from nature_style import NatureStyle

ns = NatureStyle()

# ============================================================
# 读取模型数据
# ============================================================

def extract_model_data(model_name, task_dir):
    """提取模型数据"""
    verb_map_zh = {'扔': 0, '丢': 1, '抛': 2, '投': 3, '摔': 4, '甩': 5}
    
    # 初始化
    features = np.zeros((6, 5))  # 6 verbs, 5 params
    counts = np.zeros(6)
    
    files = glob.glob(f'pilot_results/{model_name}/{task_dir}/*.json')
    
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
        
        if not data.get('is_valid', False):
            continue
        
        verb = data.get('verb', '')
        if verb not in verb_map_zh:
            continue
        
        verb_idx = verb_map_zh[verb]
        result = data.get('parsed_result', {})
        
        if result:
            force = result.get('FORCE', 3)
            hand = result.get('HAND', 6)
            arm = result.get('ARM', 0)
            hd = result.get('HD', 1)
            vd = result.get('VD', 1)
            
            features[verb_idx] += [force, hand, arm, hd, vd]
            counts[verb_idx] += 1
    
    # 计算平均值
    for i in range(6):
        if counts[i] > 0:
            features[i] /= counts[i]
    
    return features, counts

# 读取Mimo和RoboBrain数据
models = {
    'Mimo-embodied-7B': {},
    'RoboBrain2.0-7B': {}
}

verb_names = ['扔', '丢', '抛', '投', '摔', '甩']
param_names = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

for model in models:
    for task in ['task1_zh', 'task1_en']:
        features, counts = extract_model_data(model, task)
        models[model][task] = {
            'features': features,
            'counts': counts
        }
        print(f"\n{model} {task}:")
        print(f"  Valid trials per verb: {counts}")
        print(f"  Features shape: {features.shape}")

# ============================================================
# Pairwise分析
# ============================================================

def pairwise_analysis(features, verb_names, param_names):
    """对每个参数做pairwise t-test"""
    n_verbs = len(verb_names)
    pairs = list(combinations(range(n_verbs), 2))
    
    results = []
    
    for i, j in pairs:
        for p_idx, param in enumerate(param_names):
            # 由于每个动词只有1个观测值，无法做t-test
            # 改为计算差异
            diff = features[i, p_idx] - features[j, p_idx]
            results.append({
                'pair': f"{verb_names[i]} vs {verb_names[j]}",
                'param': param,
                'verb1': verb_names[i],
                'verb2': verb_names[j],
                'value1': features[i, p_idx],
                'value2': features[j, p_idx],
                'diff': diff
            })
    
    return pd.DataFrame(results)

# 分析每个模型的task1_zh
print("\n" + "="*60)
print("Model Internal Analysis: Can models distinguish 6 Chinese verbs?")
print("="*60)

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    
    for task in ['task1_zh', 'task1_en']:
        features = models[model][task]['features']
        counts = models[model][task]['counts']
        
        print(f"\n--- {task} ---")
        print("\nVerb features (raw values):")
        print(f"{'Verb':8s} {'FORCE':>6s} {'HAND':>6s} {'ARM':>6s} {'HD':>6s} {'VD':>6s}")
        print("-" * 40)
        
        for i, verb in enumerate(verb_names):
            if counts[i] > 0:
                print(f"{verb:8s} {features[i,0]:6.2f} {features[i,1]:6.2f} {features[i,2]:6.2f} {features[i,3]:6.2f} {features[i,4]:6.2f}")
            else:
                print(f"{verb:8s} {'N/A':>6s} {'N/A':>6s} {'N/A':>6s} {'N/A':>6s} {'N/A':>6s}")
        
        # Pairwise差异
        print("\nPairwise differences (top 10 by absolute difference):")
        df = pairwise_analysis(features, verb_names, param_names)
        df['abs_diff'] = df['diff'].abs()
        df_sorted = df.sort_values('abs_diff', ascending=False).head(10)
        
        for _, row in df_sorted.iterrows():
            print(f"  {row['pair']:15s} {row['param']:5s}: {row['value1']:.2f} vs {row['value2']:.2f} (diff={row['diff']:.2f})")

# ============================================================
# 可视化：模型内部差异
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mimo task1_zh
ax = axes[0, 0]
features_mimo_zh = models['Mimo-embodied-7B']['task1_zh']['features']
x = np.arange(6)
width = 0.15

for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, features_mimo_zh[:, p_idx], width, label=param)

ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('Mimo-embodied-7B: Chinese verbs (task1_zh)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend()
ax.set_ylim(0, 12)

# Mimo task1_en
ax = axes[0, 1]
features_mimo_en = models['Mimo-embodied-7B']['task1_en']['features']

for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, features_mimo_en[:, p_idx], width, label=param)

ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('Mimo-embodied-7B: Chinese verbs (task1_en)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend()
ax.set_ylim(0, 12)

# RoboBrain task1_zh
ax = axes[1, 0]
features_rb_zh = models['RoboBrain2.0-7B']['task1_zh']['features']

for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, features_rb_zh[:, p_idx], width, label=param)

ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('RoboBrain2.0-7B: Chinese verbs (task1_zh)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend()
ax.set_ylim(0, 12)

# RoboBrain task1_en
ax = axes[1, 1]
features_rb_en = models['RoboBrain2.0-7B']['task1_en']['features']

for p_idx, param in enumerate(param_names):
    ax.bar(x + p_idx * width, features_rb_en[:, p_idx], width, label=param)

ax.set_xlabel('Verb')
ax.set_ylabel('Value')
ax.set_title('RoboBrain2.0-7B: Chinese verbs (task1_en)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(verb_names)
ax.legend()
ax.set_ylim(0, 12)

plt.tight_layout()
plt.savefig('/home/lulu444/Enactment-experiment/figures/approved/model_internal_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: figures/approved/model_internal_comparison.png")

# ============================================================
# 计算模型内部方差
# ============================================================

print("\n" + "="*60)
print("Model Internal Variance Analysis")
print("="*60)

for model in models:
    print(f"\n{model}:")
    for task in ['task1_zh', 'task1_en']:
        features = models[model][task]['features']
        counts = models[model][task]['counts']
        
        # 只计算有数据的动词
        valid_mask = counts > 0
        if valid_mask.sum() > 1:
            valid_features = features[valid_mask]
            variance = np.var(valid_features, axis=0)
            print(f"  {task}: variance = {variance.round(3)}")
            print(f"    Total variance: {variance.sum():.3f}")
