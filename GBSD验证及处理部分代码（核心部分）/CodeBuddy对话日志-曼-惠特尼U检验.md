# 后门投毒检测统计学检验对话日志

**导出日期**：2026-04-25  
**对话主题**：Mann-Whitney U 检验、PCA 分析、效应量计算

## 1. Mann-Whitney U 检验（全样本）

### 1.1 代码实现

```python
"""
Mann-Whitney U 检验：基于全样本进行计算
"""
import torch
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("Mann-Whitney U 检验：全样本计算")
print("=" * 70)

# 加载全部数据
data_cifar = torch.load(
    "E:/26统计建模/generated_pts/generated_pts/cifar10_patch.pt",
    map_location="cpu",
)
data_path = torch.load(
    "E:/26统计建模/generated_pts/generated_pts/pathmnist_patch.pt",
    map_location="cpu",
)


def extract_features(images):
    features_list = []
    for i in range(len(images)):
        x = images[i].float() / 255.0
        br_region = x[:, -3:, -3:]
        br_flat = br_region.reshape(-1)
        features_list.append(br_flat.mean().item())
    return np.array(features_list)


# CIFAR-10 全量数据
feat_c = extract_features(data_cifar["images"])
poison_c = data_cifar["poison_flags"].numpy()
clean_c = feat_c[poison_c == 0]
poison_c_arr = feat_c[poison_c == 1]

# PathMNIST 全量数据
feat_p = extract_features(data_path["images"])
poison_p = data_path["poison_flags"].numpy()
clean_p = feat_p[poison_p == 0]
poison_p_arr = feat_p[poison_p == 1]


def calc_mann_whitney(data1, data2, name):
    n1, n2 = len(data1), len(data2)
    N = n1 + n2

    # 合并后排序并计算秩次
    combined = np.concatenate([data1, data2])
    sorted_indices = np.argsort(combined)
    sorted_data = combined[sorted_indices]

    ranks = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and sorted_data[j] == sorted_data[j + 1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1

    R1 = np.sum(ranks[:n1])
    R2 = np.sum(ranks[n1:])

    U1 = n1 * n2 + n1 * (n1 + 1) / 2 - R1
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    mu_U = n1 * n2 / 2
    sigma_U = np.sqrt(n1 * n2 * (N + 1) / 12)
    z = (U - mu_U) / sigma_U
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    # 秩二分相关系数
    r = (2 * U) / (n1 * n2) - 1

    print(f"\n{name}:")
    print(f"  n1（干净） = {n1:,}, n2（投毒） = {n2:,}, N = {N:,}")
    print(f"  R1 = {R1:.2f}, R2 = {R2:.2f}")
    print(f"  U1 = {U1:.2f}, U2 = {U2:.2f}")
    print(f"  U = {U:.2f}")
    print(f"  E[U] = {mu_U:.2f}")
    print(f"  sigma_U = {sigma_U:.2f}")
    print(f"  z = {z:.4f}")
    print(f"  p = {p:.2e}")
    print(f"  r（秩二分相关） = {r:.4f}")

    return U, p, z, r, n1, n2


# 执行检验
U_c, p_c, z_c, r_c, n1_c, n2_c = calc_mann_whitney(clean_c, poison_c_arr, "CIFAR-10")
U_p, p_p, z_p, r_p, n1_p, n2_p = calc_mann_whitney(clean_p, poison_p_arr, "PathMNIST")
```

### 1.2 计算结果

| 数据集 | n_clean | n_poison | N | U | z | p | r（秩二分相关） |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 45,000 | 5,000 | 50,000 | 41,624,882 | -73.20 | 约等于 0 | -0.6300 |
| PathMNIST | 80,997 | 8,999 | 89,996 | 109,735,492 | -108.94 | 约等于 0 | -0.6989 |

### 1.3 分布对比可视化

```python
"""
分布对比：横向排列展示
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings("ignore")

# 加载数据
data = np.load("mann_whitney_data.npz")
clean_c = data["clean_c"]
poison_c = data["poison_c"]
clean_p = data["clean_p"]
poison_p = data["poison_p"]

# 两个数据集并排展示
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CIFAR-10 直方图
ax1 = axes[0]
ax1.hist(clean_c, bins=60, alpha=0.6, label=f"干净样本（n={len(clean_c):,}）",
         color="#87CEEB", density=True, edgecolor="white")
ax1.hist(poison_c, bins=60, alpha=0.7, label=f"投毒样本（n={len(poison_c):,}）",
         color="#FF6B6B", density=True, edgecolor="white")
ax1.axvline(np.mean(clean_c), color="#4169E1", linestyle="--", linewidth=2.5,
            label=f"干净样本均值 = {np.mean(clean_c):.3f}")
ax1.axvline(np.mean(poison_c), color="#DC143C", linestyle="--", linewidth=2.5,
            label=f"投毒样本均值 = {np.mean(poison_c):.3f}")
ax1.set_xlabel("BR_3x3 特征值", fontsize=12)
ax1.set_ylabel("密度", fontsize=12)
ax1.set_title("CIFAR-10：分布对比（全部 50,000 个样本）", fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

# PathMNIST 直方图
ax2 = axes[1]
ax2.hist(clean_p, bins=60, alpha=0.6, label=f"干净样本（n={len(clean_p):,}）",
         color="#90EE90", density=True, edgecolor="white")
ax2.hist(poison_p, bins=60, alpha=0.7, label=f"投毒样本（n={len(poison_p):,}）",
         color="#FFA500", density=True, edgecolor="white")
ax2.axvline(np.mean(clean_p), color="#228B22", linestyle="--", linewidth=2.5,
            label=f"干净样本均值 = {np.mean(clean_p):.3f}")
ax2.axvline(np.mean(poison_p), color="#FF4500", linestyle="--", linewidth=2.5,
            label=f"投毒样本均值 = {np.mean(poison_p):.3f}")
ax2.set_xlabel("BR_3x3 特征值", fontsize=12)
ax2.set_ylabel("密度", fontsize=12)
ax2.set_title("PathMNIST：分布对比（全部 89,996 个样本）", fontsize=13, fontweight="bold")
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mann_whitney_distributions_full.png", dpi=150, bbox_inches="tight", facecolor="white")
```

生成图像：`mann_whitney_distributions_full.png`

## 2. PCA 空间可视化

### 2.1 代码实现

```python
"""
PCA 空间可视化：全样本版本
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 加载数据
data_cifar = torch.load("E:/26统计建模/generated_pts/generated_pts/cifar10_patch.pt", map_location="cpu")
data_path = torch.load("E:/26统计建模/generated_pts/generated_pts/pathmnist_patch.pt", map_location="cpu")


def extract_features_full(images):
    """提取完整特征"""
    features = []
    for i in range(len(images)):
        x = images[i].float() / 255.0

        # 触发器区域特征
        br_region = x[:, -3:, -3:]
        br_mean = br_region.mean().item()
        br_std = br_region.std().item()

        # 中心区域特征
        center_region = x[:, 10:22, 10:22]
        center_mean = center_region.mean().item()
        center_std = center_region.std().item()

        # 全图统计量
        full_mean = x.mean().item()
        full_std = x.std().item()

        features.append([br_mean, br_std, center_mean, center_std, full_mean, full_std])

    return np.array(features)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CIFAR-10 PCA
ax1 = axes[0]
scaler_c = StandardScaler()
feat_c_scaled = scaler_c.fit_transform(feat_c)
pca_c = PCA(n_components=2)
feat_c_pca = pca_c.fit_transform(feat_c_scaled)

# 抽样绘制
np.random.seed(42)
sample_size = 5000
clean_sample_idx = np.random.choice(np.where(clean_idx_c)[0], sample_size, replace=False)
poison_sample_idx = np.random.choice(np.where(poison_idx_c)[0], sample_size, replace=False)

ax1.scatter(
    feat_c_pca[clean_sample_idx, 0],
    feat_c_pca[clean_sample_idx, 1],
    c="steelblue",
    alpha=0.4,
    s=10,
    label=f"干净样本（n={clean_idx_c.sum():,}）",
)
ax1.scatter(
    feat_c_pca[poison_sample_idx, 0],
    feat_c_pca[poison_sample_idx, 1],
    c="crimson",
    alpha=0.6,
    s=20,
    label=f"投毒样本（n={poison_idx_c.sum():,}）",
)
ax1.set_xlabel(f"PC1（{pca_c.explained_variance_ratio_[0] * 100:.1f}%）", fontsize=11)
ax1.set_ylabel(f"PC2（{pca_c.explained_variance_ratio_[1] * 100:.1f}%）", fontsize=11)
ax1.set_title("CIFAR-10：PCA 空间", fontsize=14, fontweight="bold")
```

生成图像：`pca_space_full.png`

### 2.2 3D PCA 可视化（浅色版）

```python
"""
PCA 三维可视化：浅色版本
"""
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 7), facecolor="white")

# CIFAR-10 3D
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_facecolor("#f8f9fa")

pca_c3 = PCA(n_components=3)
feat_c_pca3 = pca_c3.fit_transform(feat_c_scaled)

ax1.scatter(
    feat_c_pca3[clean_sample_idx, 0],
    feat_c_pca3[clean_sample_idx, 1],
    feat_c_pca3[clean_sample_idx, 2],
    c="#87CEEB",
    alpha=0.5,
    s=15,
    label=f"干净样本（n={clean_idx_c.sum():,}）",
    edgecolors="white",
    linewidth=0.3,
)

ax1.scatter(
    feat_c_pca3[poison_sample_idx, 0],
    feat_c_pca3[poison_sample_idx, 1],
    feat_c_pca3[poison_sample_idx, 2],
    c="#FF6B6B",
    alpha=0.7,
    s=25,
    label=f"投毒样本（n={poison_idx_c.sum():,}）",
    edgecolors="white",
    linewidth=0.3,
)

ax1.set_xlabel(f"PC1（{pca_c3.explained_variance_ratio_[0] * 100:.1f}%）", fontsize=10)
ax1.set_ylabel(f"PC2（{pca_c3.explained_variance_ratio_[1] * 100:.1f}%）", fontsize=10)
ax1.set_zlabel(f"PC3（{pca_c3.explained_variance_ratio_[2] * 100:.1f}%）", fontsize=10)
ax1.set_title("CIFAR-10：三维 PCA 空间", fontsize=13, fontweight="bold", pad=10)
ax1.legend(loc="upper left", fontsize=9)
ax1.view_init(elev=20, azim=45)

# PathMNIST 3D
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_facecolor("#f8f9fa")

ax2.scatter(
    feat_p_pca3[clean_sample_idx_p, 0],
    feat_p_pca3[clean_sample_idx_p, 1],
    feat_p_pca3[clean_sample_idx_p, 2],
    c="#90EE90",
    alpha=0.5,
    s=15,
    label=f"干净样本（n={clean_idx_p.sum():,}）",
    edgecolors="white",
    linewidth=0.3,
)

ax2.scatter(
    feat_p_pca3[poison_sample_idx_p, 0],
    feat_p_pca3[poison_sample_idx_p, 1],
    feat_p_pca3[poison_sample_idx_p, 2],
    c="#FFA500",
    alpha=0.7,
    s=25,
    label=f"投毒样本（n={poison_idx_p.sum():,}）",
    edgecolors="white",
    linewidth=0.3,
)

ax2.set_xlabel(f"PC1（{pca_p3.explained_variance_ratio_[0] * 100:.1f}%）", fontsize=10)
ax2.set_ylabel(f"PC2（{pca_p3.explained_variance_ratio_[1] * 100:.1f}%）", fontsize=10)
ax2.set_zlabel(f"PC3（{pca_p3.explained_variance_ratio_[2] * 100:.1f}%）", fontsize=10)
ax2.set_title("PathMNIST：三维 PCA 空间", fontsize=13, fontweight="bold", pad=10)
ax2.legend(loc="upper left", fontsize=9)
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("pca_space_3d_light.png", dpi=150, bbox_inches="tight", facecolor="white")
```

生成图像：`pca_space_3d_light.png`、`pca_3d_cifar10_light.png`、`pca_3d_pathmnist_light.png`

## 3. PCA 主成分含义解释

### 3.1 代码实现

```python
"""
PCA 主成分解释
"""
feature_names = ["BR_mean", "BR_std", "Center_mean", "Center_std", "Full_mean", "Full_std"]

for name, feat in [("CIFAR-10", feat_c), ("PathMNIST", feat_p)]:
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    pca = PCA(n_components=6)
    pca.fit(feat_scaled)

    print(f"\n{name} 的主成分载荷：")
    for i in range(6):
        loadings = pca.components_[i]
        explained = pca.explained_variance_ratio_[i] * 100
        sorted_idx = np.argsort(np.abs(loadings))[::-1]

        print(f"\nPC{i + 1}（解释方差 {explained:.2f}%）：")
        print(f"  载荷：{[f'{loadings[j]:.3f}' for j in range(6)]}")
        print("  主要贡献项：")
        for j in sorted_idx[:3]:
            sign = "+" if loadings[j] > 0 else "-"
            print(f"    {sign} {feature_names[j]}: {abs(loadings[j]):.3f}")
```

### 3.2 主成分含义

#### CIFAR-10

| 主成分 | 方差解释 | 主要含义 |
|---|---|---|
| PC1（32.4%） | Full_mean（0.59）+ BR_mean（0.45） | 整体亮度：反映图像平均亮度水平 |
| PC2（22.0%） | Center_std（0.62）+ Full_std（0.57） | 局部变异性：反映中心区域纹理复杂度 |
| PC3（18.3%） | BR_std（0.76）+ BR_mean（0.49） | 触发器特征：反映右下角区域异常变化 |

#### PathMNIST

| 主成分 | 方差解释 | 主要含义 |
|---|---|---|
| PC1（58.3%） | Full_mean（0.47）+ Center_mean（0.46） | 整体亮度 |
| PC2（23.8%） | Full_std（0.54）+ Center_std（0.50） | 纹理复杂度 |
| PC3（10.3%） | BR_std（0.90） | 触发器信号，是关键后门检测特征 |
| PC4（5.4%） | BR_mean（0.81） | 触发器强度，反映触发器亮度水平 |

## 4. 效应量计算

### 4.1 常用语言效应量（CLES）

```text
CLES = P(Poison > Clean) = U / (n_clean × n_poison)
```

| 数据集 | n_clean | n_poison | U 统计量 | CLES |
|---|---:|---:|---:|---:|
| CIFAR-10 | 45,000 | 5,000 | 183,375,118 | 0.8150（81.5%） |
| PathMNIST | 80,997 | 8,999 | 619,156,511 | 0.8494（84.9%） |

### 4.2 CLES 解释

| CLES 值 | 效应量大小 | 解释 |
|---|---|---|
| 0.50 | 无 | 随机水平 |
| 0.56 | 小 | 56% 的概率高于对照组 |
| 0.64 | 中 | 64% 的概率高于对照组 |
| 0.81 至 0.85 | 大 | 超过 80% 的概率高于对照组 |

## 5. 生成文件清单

### 5.1 代码文件

| 文件名 | 描述 |
|---|---|
| `mann_whitney_full.py` | Mann-Whitney U 全样本计算 |
| `distribution_comparison.py` | 分布对比可视化 |
| `pca_visualization.py` | PCA 空间可视化 |
| `pca_3d_light.py` | 三维 PCA 浅色版 |
| `pca_components.py` | PCA 主成分解释 |
| `mann_whitney_data.npz` | 预处理后的中间数据 |

### 5.2 可视化图像

| 文件名 | 描述 |
|---|---|
| `mann_whitney_distributions_full.png` | 分布对比图 |
| `pca_space_full.png` | 二维 PCA 空间图 |
| `pca_space_3d.png` | 三维 PCA 空间图 |
| `pca_space_3d_light.png` | 三维 PCA 浅色版 |
| `pca_3d_cifar10_light.png` | CIFAR-10 单独三维图 |
| `pca_3d_pathmnist_light.png` | PathMNIST 单独三维图 |

## 6. 结论

- CIFAR-10 与 PathMNIST 的干净样本和投毒样本在 Mann-Whitney U 检验下均呈现极显著差异，`p` 值接近 0。
- 两个数据集的效应量都很大，`|r| > 0.5`，说明差异不仅统计显著，而且具有明显的实际意义。
- CLES 分别达到 81.5% 和 84.9%，说明投毒样本在目标特征上的数值显著高于干净样本。
- PCA 结果表明，触发器区域相关特征能够在特征空间中形成可分离结构，可为后门检测提供支持。
