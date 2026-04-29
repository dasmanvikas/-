# Kolmogorov-Smirnov 检验实验日志

**导出日期**：2026-04-27  
**相关文件**：`statistical_tests.py`

## 1. K-S 检验理论基础

### 1.1 检验原理

Kolmogorov-Smirnov 检验是一种非参数检验方法，用于比较两个经验累积分布函数（ECDF）之间的最大垂直距离。

```text
K-S 统计量：D = max |F1(x) - F2(x)|
```

其中：

- `F1(x)`：第一组（干净样本）的经验累积分布函数
- `F2(x)`：第二组（投毒样本）的经验累积分布函数
- `D`：两组 ECDF 之间的最大垂直距离

### 1.2 D 值解释

| D 值范围 | 解释 |
|---|---|
| `D < 0.2` | 两组分布高度重叠 |
| `0.2 ≤ D < 0.4` | 轻度差异 |
| `0.4 ≤ D < 0.6` | 中等差异 |
| `0.6 ≤ D < 0.8` | 显著差异 |
| `D ≥ 0.8` | 极大差异 |

### 1.3 假设设置

| 假设 | 内容 |
|---|---|
| `H0` | 干净样本与投毒样本来自同一分布 |
| `H1` | 干净样本与投毒样本来自不同分布 |

判定标准：`p < 0.05` 时拒绝 `H0`。

## 2. K-S 检验代码实现

```python
print()
print("=" * 70)
print("6. Kolmogorov-Smirnov 检验")
print("=" * 70)
print("H0：两组样本来自同一分布")
print("p < 0.05：拒绝 H0，认为分布不同")
print()

# CIFAR-10 K-S 检验
ks_c, p_c_ks = stats.ks_2samp(clean_c6, poison_c6_feat)
print("CIFAR-10:")
print(f"  KS = {ks_c:.4f}")
print(f"  p = {p_c_ks:.2e}")
print(f"  结论：{'分布不同' if p_c_ks < 0.05 else '分布相同'}")

# PathMNIST K-S 检验
ks_p, p_p_ks = stats.ks_2samp(clean_p6, poison_p6_feat)
print("\nPathMNIST:")
print(f"  KS = {ks_p:.4f}")
print(f"  p = {p_p_ks:.2e}")
print(f"  结论：{'分布不同' if p_p_ks < 0.05 else '分布相同'}")
```

## 3. 实验输出

```text
======================================================================
6. Kolmogorov-Smirnov Test
======================================================================
H0: Two groups come from the same distribution
p < 0.05 -> Reject H0, different distributions

CIFAR-10:
  KS = 0.6007
  p = 0.00e+00
  Conclusion: Different distributions

PathMNIST:
  KS = 0.6466
  p = 0.00e+00
  Conclusion: Different distributions
======================================================================
```

## 4. 辅助统计结果

### 4.1 描述性统计

```text
CIFAR-10 Clean (Label 6):
  n       = 4474
  Mean    = 0.4536
  Median  = 0.4253
  Std     = 0.2308
  Min     = 0.0000
  Max     = 1.0000
  IQR     = 0.2996

CIFAR-10 Poison (Label 6):
  n       = 5000
  Mean    = 0.7143
  Median  = 0.7034
  Std     = 0.1304
  Min     = 0.4444
  Max     = 1.0000
  IQR     = 0.1799

PathMNIST Clean (Label 6):
  n       = 7084
  Mean    = 0.6507
  Median  = 0.6558
  Std     = 0.1118
  Min     = 0.3124
  Max     = 0.9744
  IQR     = 0.1562

PathMNIST Poison (Label 6):
  n       = 8999
  Mean    = 0.8107
  Median  = 0.8092
  Std     = 0.0709
  Min     = 0.4450
  Max     = 0.9843
  IQR     = 0.0836
```

### 4.2 正态性检验

```text
======================================================================
2. Normality Test (Shapiro-Wilk)
======================================================================
H0: Data follows normal distribution
p < 0.05 -> Reject H0, data is NOT normal

CIFAR-10:
  Clean:   W = 0.9704, p = 1.85e-29 -> Non-normal
  Poison:  W = 0.9829, p = 3.87e-24 -> Non-normal

PathMNIST:
  Clean:   W = 0.9963, p = 7.68e-10 -> Non-normal
  Poison:  W = 0.9888, p = 1.68e-19 -> Non-normal
```

结论：由于数据不服从正态分布，因此使用 K-S 检验这类非参数方法更合适。

### 4.3 其他非参数检验

```text
======================================================================
5. Mann-Whitney U Test
======================================================================
CIFAR-10:
  U = 3671462
  p = 0.00e+00
  Conclusion: Distributions differ significantly

PathMNIST:
  U = 7218789
  p = 0.00e+00
  Conclusion: Distributions differ significantly
```

## 5. 核心结果汇总

### 5.1 K-S 检验结果

| 数据集 | K-S D | p 值 | 显著性 | 分布差异比例 |
|---|---:|---:|---|---:|
| CIFAR-10 | 0.6007 | 约等于 0 | `p < 0.001` | 60.07% |
| PathMNIST | 0.6466 | 约等于 0 | `p < 0.001` | 64.66% |

### 5.2 效应量对比

| 指标 | CIFAR-10 | PathMNIST | 解释 |
|---|---:|---:|---|
| K-S D | 0.6007 | 0.6466 | 均达到显著差异水平 |
| Cohen's d | 1.4105 | 1.7543 | 均为大效应量 |
| Cliff's δ | 0.6718 | 0.7735 | 均为大效应量 |
| AUC | 0.8359 | 0.8868 | 区分能力良好 |

## 6. 实践意义

### 6.1 对后门检测的意义

- `K-S D > 0.6` 表明两组样本的分布重叠较少。
- 即使采用简单阈值，也有较大概率实现有效区分。
- 统计结果可为 GBSD 类方法提供分布层面的证据支撑。
- 触发器特征在特征空间中形成了可检测偏移。

### 6.2 与其他检验的一致性

| 检验方法 | CIFAR-10 | PathMNIST | 结论 |
|---|---|---|---|
| Welch's t | `p ≈ 0` | `p ≈ 0` | 均值显著不同 |
| Mann-Whitney U | `p ≈ 0` | `p ≈ 0` | 秩次分布显著不同 |
| K-S 检验 | `p ≈ 0` | `p ≈ 0` | 整体分布显著不同 |

三类检验的结论一致：干净样本与投毒样本之间存在极显著差异。

## 7. 统计检验汇总表

| Statistical Test | CIFAR-10 | PathMNIST |
|---|---|---|
| Sample Size (Clean) | 4474 | 7084 |
| Sample Size (Poison) | 5000 | 8999 |
| Mean Diff | 0.2606 | 0.1600 |
| 95% CI | `[0.253, 0.268]` | `[0.157, 0.163]` |
| Shapiro-Wilk | Non-normal | Non-normal |
| Levene Test | Unequal | Unequal |
| Welch's t | `t = -66.60` | `t = -104.98` |
| Mann-Whitney U | `U = 3,671,462` | `U = 7,218,789` |
| K-S D | `0.6007` | `0.6466` |
| Cohen's d | `1.4105` | `1.7543` |
| Cliff's δ | `0.6718` | `0.7735` |
| AUC | `0.8359` | `0.8868` |

## 8. 相关文件

| 文件 | 说明 |
|---|---|
| `statistical_tests.py` | 统计检验主代码 |
| `KS检验实验日志.md` | 本文档 |

## 9. 结论

- CIFAR-10 的 `K-S D = 0.6007`，PathMNIST 的 `K-S D = 0.6466`。
- 两个数据集的 `p` 值都接近 0，均应拒绝原假设 `H0`。
- 这说明干净样本与投毒样本并非来自同一分布。
- 触发器相关特征能够有效区分两类样本，可作为后门检测的重要依据。
