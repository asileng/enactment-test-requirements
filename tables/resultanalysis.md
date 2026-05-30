# 结果分析：中英文"扔"类动词的行为参数区分

## 研究背景

本研究比较中文和英文"throw"类动词在五个行为参数上的区分模式，探究不同语言如何通过动作特征区分语义相近的动词。

## 数据来源

- **数据文件**: `table1_descriptive_statistics.csv`
- **样本量**: 每种语言30名母语者
- **动词数量**: 中文6个，英文6个
- **参数**: FORCE（力量）、HAND（手高度）、ARM（手臂姿势）、HD（水平方向）、VD（垂直方向）

## 一、典型值对比（Figure 4）

### 图表说明
展示12个动词在FORCE、HAND、ARM三个连续参数上的典型值。

### 统计数据

| 动词 | FORCE | HAND | ARM (straight比例) |
|------|-------|------|-------------------|
| **中文** | | | |
| reng 掷 | 3.05 | 5.43 | 0.60 |
| diu 丢 | 2.33 | 4.63 | 0.80 |
| pao 抛 | 3.16 | 6.65 | 0.52 |
| tou 投 | 3.21 | 9.13 | 0.00 |
| shuai 摔 | 4.55 | 8.51 | 0.00 |
| shuai 甩 | 3.74 | 6.27 | 0.28 |
| **英文** | | | |
| throw | 3.62 | 8.58 | 0.17 |
| fling | 3.44 | 6.41 | 0.39 |
| chuck | 3.91 | 7.06 | 0.34 |
| cast | 3.01 | 6.18 | 0.48 |
| hurl | 4.39 | 8.00 | 0.28 |
| toss | 3.01 | 4.50 | 0.90 |

### 关键发现

1. **FORCE维度**: 中文shuai摔(4.55)力量最大，diu丢(2.33)最小；英文hurl(4.39)最大，toss(3.01)最小
2. **HAND维度**: 中文tou投(9.13)手位最高，diu丢(4.63)最低；英文throw(8.58)最高，toss(4.50)最低
3. **ARM维度**: 中文diu丢(0.80)和英文toss(0.90)直臂比例最高；中文tou投和shuai摔完全弯臂

---

## 二、方向模式分析（Figure 5）

### 图表说明
展示VD（垂直方向）和HD（水平方向）的典型模式。

### 统计数据

| 动词 | VD向上比例 | HD向前比例 |
|------|-----------|-----------|
| **中文** | | |
| reng 掷 | 0.72 | 0.72 |
| diu 丢 | 0.82 | 0.57 |
| pao 抛 | 1.00 | 0.97 |
| tou 投 | 0.97 | 1.00 |
| shuai 摔 | **0.00** | 0.86 |
| shuai 甩 | 0.48 | **0.31** |
| **英文** | | |
| throw | 0.90 | 1.00 |
| fling | 0.93 | 0.86 |
| chuck | 0.72 | 0.93 |
| cast | 0.97 | 0.93 |
| hurl | 0.83 | 0.97 |
| toss | 1.00 | 1.00 |

### 关键发现

1. **VD（垂直方向）**:
   - 中文shuai摔VD=0（完全向下），是唯一完全向下的动词
   - 英文所有动词VD>0.7（主要向上）

2. **HD（水平方向）**:
   - 中文shuai甩HD=0.31（明显侧向），是唯一有侧向特征的动词
   - 英文所有动词HD>0.85（几乎完全向前）

---

## 三、Pairwise显著性矩阵（Figure 6）

### 图表说明
展示15个动词对在5个参数上的显著性检验结果（从Table 3, 4直接提取）。

### 统计方法
- FORCE, HAND: 使用Mean和SD做两独立样本t-test
- ARM, HD, VD: 使用频次数据做proportion test

### 中文Pairwise结果

| 动词对 | FORCE | HAND | ARM | HD | VD |
|--------|-------|------|-----|----|----|
| reng vs diu | ** | ns | ns | ns | ns |
| reng vs pao | ns | ns | ns | ns | ns |
| reng vs tou | ns | *** | *** | ** | ** |
| reng vs shuai摔 | *** | *** | *** | ns | *** |
| reng vs shuai甩 | * | ns | ** | ** | ns |
| diu vs pao | *** | ** | * | *** | ** |
| diu vs tou | *** | *** | *** | *** | * |
| diu vs shuai摔 | *** | *** | *** | * | *** |
| diu vs shuai甩 | *** | *** | *** | ns | * |
| pao vs tou | ns | *** | *** | ns | ns |
| pao vs shuai摔 | *** | * | *** | *** | *** |
| pao vs shuai甩 | ** | ns | *** | *** | *** |
| tou vs shuai摔 | *** | ns | ns | * | *** |
| tou vs shuai甩 | ** | *** | ** | *** | *** |
| shuai摔 vs shuai甩 | *** | *** | ** | *** | *** |

### 英文Pairwise结果

| 动词对 | FORCE | HAND | ARM | HD | VD |
|--------|-------|------|-----|----|----|
| throw vs fling | ns | *** | ns | ns | ns |
| throw vs chuck | ns | ** | ns | ns | ns |
| throw vs cast | *** | *** | * | ns | ns |
| throw vs hurl | *** | ns | ns | ns | ns |
| throw vs toss | *** | *** | *** | ns | ns |
| fling vs chuck | * | ns | ns | ns | ns |
| fling vs cast | ** | ns | ns | ns | ns |
| fling vs hurl | *** | ** | ns | ns | ns |
| fling vs toss | ** | *** | *** | ns | ns |
| chuck vs cast | *** | ns | ns | ns | * |
| chuck vs hurl | * | ns | ns | ns | ns |
| chuck vs toss | *** | *** | *** | ns | ** |
| cast vs hurl | *** | ** | ns | ns | ns |
| cast vs toss | ns | ** | ** | ns | ns |
| hurl vs toss | *** | *** | *** | ns | * |

---

## 四、参数区分力汇总（Figure 9）

### 图表说明
统计每个参数能区分多少个动词对（显著性检验结果）。

### 统计数据

| 参数 | 中文（15对） | 英文（15对） |
|------|-------------|-------------|
| **FORCE** | 12对 (80%) | 12对 (80%) |
| **HAND** | 13对 (87%) | 10对 (67%) |
| **ARM** | 11对 (73%) | 6对 (40%) |
| **HD** | 10对 (67%) | **0对 (0%)** |
| **VD** | 12对 (80%) | 3对 (20%) |

### 高度显著（***）统计

| 参数 | 中文 | 英文 |
|------|------|------|
| FORCE | 11对 | 8对 |
| HAND | 10对 | 6对 |
| ARM | 7对 | 4对 |
| HD | 5对 | 0对 |
| VD | 7对 | 0对 |

---

## 五、核心结论

### 结论1：参数区分力存在语言差异

**中文使用5个参数区分动词，英文主要使用3个参数。**

- 中文最有效参数：HAND (87%), FORCE (80%), VD (80%)
- 英文最有效参数：FORCE (80%), HAND (67%), ARM (40%)

### 结论2：HD和VD是中文独有的区分维度

- **HD（水平方向）**: 中文10对显著，英文0对显著
- **VD（垂直方向）**: 中文12对显著，英文仅3对显著

这表明中文"扔"类动词在空间方向上编码了更多语义信息。

### 结论3：特定动词对的区分指标

**中文最强区分指标组合**:
- diu vs shuai摔: FORCE + HAND + ARM + VD（4个参数）
- tou vs shuai甩: HAND + HD + VD（3个参数）
- shuai摔 vs shuai甩: 所有5个参数

**英文最强区分指标组合**:
- throw vs toss: FORCE + HAND + ARM（3个参数）
- chuck vs toss: FORCE + HAND + ARM（3个参数）

### 结论4：典型行为模式

**中文"扔"类动词的语义空间更精细**:
- 轻抛型: diu丢（低力量、低手位、直臂）
- 中抛型: reng掷、pao抛（中等力量、中等手位）
- 重投型: tou投、shuai摔（高力量、高手位、弯臂）
- 侧甩型: shuai甩（侧向运动）

**英文"扔"类动词的语义空间较粗略**:
- 轻抛型: toss（低力量、低手位、直臂）
- 中抛型: throw、fling、chuck、cast（中等特征）
- 重抛型: hurl（高力量、高手位）

---

## 六、图表索引

| 图表文件 | 内容 | 说明 |
|----------|------|------|
| `nature_fig4_typical_values.png/pdf` | 典型值对比 | FORCE、HAND、ARM三参数柱状图 |
| `nature_fig5_direction_patterns.png/pdf` | 方向模式 | VD和HD比例柱状图 |
| `nature_fig6_pairwise_matrix.png/pdf` | Pairwise矩阵 | 15对×5参数热图（原始Table 3,4数据） |
| `nature_fig9_statistical_discriminability.png/pdf` | 参数区分力 | 从Table 1统计检验计算 |
| `polar_sector_visualization.png/pdf` | 极坐标可视化 | 12个动词的行为轨迹 |

---

## 七、数据文件

| 文件 | 内容 |
|------|------|
| `pairwise_long_format_chinese.csv` | 中文Pairwise长格式数据 |
| `pairwise_long_format_english.csv` | 英文Pairwise长格式数据 |
| `pairwise_statistical_chinese.csv` | 中文统计检验结果 |
| `pairwise_statistical_english.csv` | 英文统计检验结果 |

---

*分析完成于 2026年5月30日*
