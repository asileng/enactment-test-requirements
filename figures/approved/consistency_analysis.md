# 人类-模型一致性分析

## 分析概述

本分析比较 Mimo-embodied-7B 和 RoboBrain2.0-7B 两个模型与人类在中文"扔"类动词编码任务上的一致性。

## 数据

- **人类基准**: 30名中文母语者对6个动词的5维编码（来自Gao 2016）
- **模型数据**: 每个模型对6个中文动词的5维编码（task1_zh）

### 5个行为参数

| 参数 | 描述 | 取值范围 |
|------|------|---------|
| FORCE | 力量强度 | 1-5 |
| HAND | 手部高度 | 0-12 |
| ARM | 手臂姿势 | 0=弯曲, 1=伸直 |
| HD | 水平方向 | 0=侧向, 1=向前 |
| VD | 垂直方向 | 0=向下, 1=向上 |

---

## 1. RSA (Representational Similarity Analysis)

### 公式

$$
\text{RSA}(X, Y) = \text{Spearman}(\text{vec}(D_X), \text{vec}(D_Y))
$$

其中：
- $D_X$ = 特征矩阵 $X$ 的欧氏距离矩阵
- $\text{vec}(\cdot)$ = 取上三角向量

### 计算过程

1. 对人类特征矩阵 $H \in \mathbb{R}^{6 \times 5}$ 和模型特征矩阵 $M \in \mathbb{R}^{6 \times 5}$ 分别计算距离矩阵
2. 提取距离矩阵的上三角元素
3. 计算两个距离向量的 Spearman 相关系数

### 结果

$$
\text{RSA}(H, \text{Mimo}) = 0.4607 \quad (p = 0.084)
$$

$$
\text{RSA}(H, \text{RoboBrain}) = 0.4251 \quad (p = 0.114)
$$

### 解读

- RSA 值范围：[-1, 1]，越高表示表征结构越相似
- Mimo (0.46) 略高于 RoboBrain (0.43)
- 两者均未达到统计显著性 (p > 0.05)

---

## 2. CKA (Centered Kernel Alignment)

### 公式

$$
\text{CKA}(X, Y) = \frac{\text{HSIC}(K_X, K_Y)}{\sqrt{\text{HSIC}(K_X, K_X) \cdot \text{HSIC}(K_Y, K_Y)}}
$$

其中 HSIC (Hilbert-Schmidt Independence Criterion)：

$$
\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(KHLH)
$$

- $K = XX^T$, $L = YY^T$ 为线性核矩阵
- $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ 为中心化矩阵

### 计算过程

1. 中心化特征矩阵：$X_c = X - \bar{X}$
2. 计算核矩阵：$K = X_c X_c^T$
3. 计算 HSIC 值
4. 归一化得到 CKA

### 结果

$$
\text{CKA}(H, \text{Mimo}) = 0.4954
$$

$$
\text{CKA}(H, \text{RoboBrain}) = 0.5571
$$

### 解读

- CKA 值范围：[0, 1]，越高表示表征越相似
- RoboBrain (0.56) 高于 Mimo (0.50)
- RoboBrain 与人类的表征相似度更高

---

## 3. 动词作用一致性 (Verb-wise Functional Consistency)

### 公式

$$
C_v = \| T_{\text{human}}(v) - T_{\text{model}}(v) \|_2
$$

其中：
- $T_s(v)$ = 物种 $s$ 在动词 $v$ 下的5维特征向量
- $\|\cdot\|_2$ = 欧氏距离

### 计算过程

对每个动词 $v \in \{扔, 丢, 抛, 投, 摔, 甩\}$：
1. 提取人类特征向量 $h_v \in \mathbb{R}^5$
2. 提取模型特征向量 $m_v \in \mathbb{R}^5$
3. 计算欧氏距离 $C_v = \|h_v - m_v\|_2$

### 结果

| 动词 | Mimo $C_v$ | RoboBrain $C_v$ |
|------|-----------|-----------------|
| 扔 | 1.1143 | 1.2325 |
| 丢 | 1.4306 | 1.5778 |
| 抛 | 1.3570 | 0.9950 |
| 投 | 1.4269 | 1.1084 |
| 摔 | 1.6833 | 1.6981 |
| 甩 | 1.0220 | 0.9626 |
| **Mean** | **1.3390** | **1.2624** |
| **Std** | **0.2183** | **0.2815** |

### 解读

- 距离越小表示模型与人类越一致
- RoboBrain 平均距离 (1.26) 小于 Mimo (1.34)
- 两个模型在"摔"上差异最大，在"甩"上差异最小

---

## 4. 参数敏感性一致性 (Parameter Sensitivity Consistency)

### 公式

$$
\text{SC} = \text{Corr}(\sigma^2_{\text{human}}, \sigma^2_{\text{model}})
$$

其中：
- $\sigma^2_p = \text{Var}(\{f_v(p)\}_{v=1}^6)$ = 参数 $p$ 在所有动词上的方差

### 计算过程

对每个参数 $p \in \{FORCE, HAND, ARM, HD, VD\}$：
1. 计算人类在该参数上的方差 $\sigma^2_{h,p}$
2. 计算模型在该参数上的方差 $\sigma^2_{m,p}$
3. 计算两组方差的 Pearson 相关系数

### 结果

| 参数 | Human方差 | Mimo方差 | RoboBrain方差 |
|------|----------|---------|--------------|
| FORCE | 0.463 | 0.139 | 0.333 |
| HAND | 2.539 | 3.889 | 11.250 |
| ARM | 2.300 | 0.139 | 0.250 |
| HD | 1.446 | 0.000 | 0.222 |
| VD | 2.944 | 0.222 | 0.000 |

$$
\text{SC}(H, \text{Mimo}) = 0.3603
$$

$$
\text{SC}(H, \text{RoboBrain}) = 0.3191
$$

### 解读

- 相关性越接近1表示参数敏感性越一致
- 两个模型与人类的参数敏感性均较弱相关 (~0.3)
- 模型对参数的利用方式与人类不同

---

## 综合对比

| 指标 | Mimo-embodied-7B | RoboBrain2.0-7B | 更优 |
|------|------------------|-----------------|------|
| RSA | 0.4607 | 0.4251 | Mimo |
| CKA | 0.4954 | 0.5571 | RoboBrain |
| Verb Consistency (mean) | 1.3390 | 1.2624 | RoboBrain |
| Param Sensitivity Corr | 0.3603 | 0.3191 | Mimo |

## 结论

1. **表征相似性**: 两个模型与人类的表征结构相似度中等 (RSA 0.43-0.46)
2. **表征对齐**: RoboBrain 与人类的表征对齐度更高 (CKA 0.56 vs 0.50)
3. **动词一致性**: RoboBrain 在动词编码上与人类更接近 (距离 1.26 vs 1.34)
4. **参数敏感性**: 两个模型与人类的参数利用方式均不一致 (相关 ~0.3)

**总体**: RoboBrain 在多数指标上略优于 Mimo，但两者与人类均存在显著差异。

---

*分析完成于 2026年5月31日*
