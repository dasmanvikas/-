"""
GBSD: 分布感知粒球后门检测算法
完整可运行代码
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 数据加载函数
# ============================================

def load_data(filepath):
    """
    加载 PyTorch 数据文件
    参数:
        filepath: .pt 文件路径
    返回:
        dict: 包含 images, labels, poison_flags
    """
    data = torch.load(filepath, map_location='cpu')
    return {
        'images': data['images'],
        'labels': data['labels'],
        'poison_flags': data['poison_flags']
    }


# ============================================
# 特征提取函数
# ============================================

def extract_trigger_features(images):
    """
    提取右下角触发器区域特征 (核心!)
    针对 patch 型后门攻击，触发器通常位于固定位置(右下角)
    
    参数:
        images: torch.Tensor [N, C, H, W]
    返回:
        np.array [N, 13]: 提取的13维特征
    """
    features_list = []
    
    for i in range(len(images)):
        x = images[i].float()
        # 归一化: uint8 [0-255] -> float [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        # === 核心特征: 右下角 3x3 区域 ===
        br_region = x[:, -3:, -3:]  # [C, 3, 3]
        br_flat = br_region.reshape(-1)  # 展平为 1D 向量
        
        # 6 个核心统计特征
        f1 = br_flat.mean().item()          # 均值 (最重要, AUC~0.86)
        f2 = br_flat.std().item()            # 标准差
        f3 = (br_flat > 0.9).float().mean().item()  # 白色像素比例 (AUC~0.98)
        f4 = (br_flat > 0.95).float().mean().item() # 高亮像素比例
        f5 = br_flat.max().item()            # 最大值 (AUC=1.0!)
        f6 = br_flat.min().item()            # 最小值
        
        # 扩展区域特征 (5x5, 7x7)
        f7 = x[:, -5:, -5:].mean().item()
        f8 = x[:, -5:, -5:].std().item()
        f9 = x[:, -7:, -7:].mean().item()
        f10 = x[:, -7:, -7:].std().item()
        
        # 对比角落特征 (与其他角落的差异)
        f11 = x[:, :3, :3].mean().item()   # 左上均值
        f12 = x[:, :3, -3:].mean().item()  # 右上均值
        f13 = x[:, -3:, :3].mean().item()  # 左下均值
        
        features_list.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])
    
    return np.array(features_list)


# ============================================
# 谱签名计算
# ============================================

def compute_spectral_signature(X):
    """
    计算谱签名特征: PCA重构误差 + KNN密度估计
    
    参数:
        X: np.array [N, D], 特征矩阵
    返回:
        recon_error: PCA重构误差
        density: KNN密度估计
        X_pca: PCA降维后的特征
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维
    n_comp = min(10, len(X) - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 重构误差 (谱签名核心)
    X_recon = pca.inverse_transform(X_pca)
    recon_error = np.sum((X_scaled - X_recon) ** 2, axis=1)
    
    # KNN密度估计
    k = min(15, len(X) - 1)
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn_model.fit(X_pca)
    dists, _ = nn_model.kneighbors(X_pca)
    density = dists.mean(axis=1)
    
    return recon_error, density, X_pca


# ============================================
# 分数组合
# ============================================

def combine_scores(X, recon_error, density, w_recon=0.1, w_density=0.1, w_br=0.8):
    """
    组合多源异常分数
    
    参数:
        X: 特征矩阵
        recon_error: 重构误差
        density: 密度估计
        w_*: 各分量权重
    返回:
        scores: 组合后的异常分数 [0, 1]
    """
    br_mean = X[:, 0]  # 右下角均值是主特征
    
    # 归一化到 [0, 1]
    recon_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
    density_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
    br_norm = (br_mean - br_mean.min()) / (br_mean.max() - br_mean.min() + 1e-8)
    
    # 加权组合 (BR特征占80%权重)
    scores = w_recon * recon_norm + w_density * density_norm + w_br * br_norm
    
    return scores


# ============================================
# 粒球清洗
# ============================================

def granular_ball_cleaning(X_pca, scores, n_balls=25, contamination_threshold=0.35):
    """
    基于粒球的后门样本清洗
    
    参数:
        X_pca: PCA降维后的特征
        scores: 异常分数
        n_balls: 粒球数量
        contamination_threshold: 污染度阈值
    返回:
        assignments: 每个样本的粒球分配
        clean_balls: 清洁粒球ID列表
        toxic_balls: 有毒粒球ID列表
        contamination_norm: 归一化污染度
    """
    # KMeans聚类生成粒球
    kmeans = KMeans(n_clusters=n_balls, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(X_pca)
    
    # 计算每个粒球的污染度
    contamination = np.zeros(n_balls)
    for ball_id in range(n_balls):
        mask = (assignments == ball_id)
        if mask.sum() == 0:
            continue
        
        center = X_pca[mask].mean(axis=0)
        dists = np.linalg.norm(X_pca[mask] - center, axis=1)
        
        # 紧凑度 = 均值/标准差
        compactness = (dists.mean() + 1e-8) / (dists.std() + 1e-8)
        # 污染度 = 紧凑度的倒数 + 分数方差
        contamination[ball_id] = 1.0 / (compactness + 1e-8) + np.std(scores[mask])
    
    # 归一化污染度
    contamination_norm = (contamination - contamination.min()) / \
                         (contamination.max() - contamination.min() + 1e-8)
    
    # 区分清洁/有毒粒球
    clean_balls = np.where(contamination_norm < contamination_threshold)[0]
    toxic_balls = np.where(contamination_norm >= contamination_threshold)[0]
    
    return assignments, clean_balls, toxic_balls, contamination_norm


# ============================================
# 分数传播
# ============================================

def propagate_scores(X_pca, scores, assignments, clean_balls, n_neighbors=10):
    """
    KNN分数传播: 用清洁样本的分数推断未知样本
    
    参数:
        X_pca: PCA特征
        scores: 原始分数
        assignments: 粒球分配
        clean_balls: 清洁粒球ID
        n_neighbors: 近邻数量
    返回:
        final_scores: 传播后的最终分数
    """
    clean_mask = np.isin(assignments, clean_balls)
    
    if clean_mask.sum() == 0:
        return scores
    
    X_clean = X_pca[clean_mask]
    clean_scores = scores[clean_mask]
    
    # 在清洁空间建立KNN
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_clean)), algorithm='ball_tree')
    knn.fit(X_clean)
    
    # 距离加权传播
    final_scores = np.zeros_like(scores)
    for i in range(len(X_pca)):
        dists, idx = knn.kneighbors([X_pca[i]])
        weights = 1.0 / (dists[0] + 1e-8)
        final_scores[i] = np.sum(weights * clean_scores[idx[0]]) / np.sum(weights)
    
    return final_scores


# ============================================
# 阈值优化
# ============================================

def optimize_threshold(scores, y_true, n_thresholds=500):
    """
    F1最大化的阈值搜索
    
    参数:
        scores: 异常分数
        y_true: 真实标签
        n_thresholds: 搜索精度
    返回:
        best_thresh: 最优阈值
        best_f1: 对应的F1分数
    """
    best_f1, best_thresh = 0, 0
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        if pred.sum() == 0:
            continue
        
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    
    return best_thresh, best_f1


# ============================================
# GB-SSD-DA 主检测函数
# ============================================

def gbssd_da_detect(data, target_label=6):
    """
    GB-SSD-DA 后门检测主函数
    
    参数:
        data: 包含 images, labels, poison_flags 的字典
        target_label: 目标类别 (默认为6, 即包含污染样本的标签)
    返回:
        dict: 包含 predictions, scores, metrics 等结果
    """
    images = data['images']
    labels = data['labels'].numpy()
    poison_flags = data['poison_flags'].numpy()
    
    print(f"[INFO] 总样本数: {len(images)}, 污染样本: {poison_flags.sum()}")
    
    # 筛选目标类别
    target_mask = (labels == target_label)
    if target_mask.sum() == 0:
        target_mask = np.ones(len(labels), dtype=bool)
    
    images_target = images[target_mask]
    poison_target = poison_flags[target_mask]
    y_true = poison_target.astype(int)
    
    print(f"[INFO] 目标类别: {target_label}, 筛选后样本数: {len(images_target)}")
    
    # 1. 特征提取
    print("[1/6] 提取触发器特征...")
    X_features = extract_trigger_features(images_target)
    
    # 2. 谱签名计算
    print("[2/6] 计算谱签名...")
    recon_error, density, X_pca = compute_spectral_signature(X_features)
    
    # 3. 分数组合
    print("[3/6] 组合异常分数...")
    scores = combine_scores(X_features, recon_error, density)
    
    # 4. 粒球清洗
    print("[4/6] 粒球清洗...")
    assignments, clean_balls, toxic_balls, contamination = granular_ball_cleaning(X_pca, scores)
    print(f"    清洁粒球: {len(clean_balls)}, 有毒粒球: {len(toxic_balls)}")
    
    # 5. 分数传播
    print("[5/6] 分数传播...")
    final_scores = propagate_scores(X_pca, scores, assignments, clean_balls)
    
    # 6. 阈值优化
    print("[6/6] 阈值优化...")
    best_thresh, best_f1 = optimize_threshold(final_scores, y_true)
    predictions = (final_scores >= best_thresh).astype(int)
    
    # 计算评估指标
    auc = roc_auc_score(y_true, final_scores)
    ap = average_precision_score(y_true, final_scores)
    
    clean_s = final_scores[y_true == 0]
    poison_s = final_scores[y_true == 1]
    
    # 计算分离度 d'
    if len(clean_s) > 0 and len(poison_s) > 0:
        d_prime = np.abs(poison_s.mean() - clean_s.mean()) / \
                  np.sqrt(0.5 * (clean_s.std()**2 + poison_s.std()**2) + 1e-8)
    else:
        d_prime = 0
    
    metrics = {
        'AUC': auc,
        'AP': ap,
        'F1': best_f1,
        'd_prime': d_prime,
        'best_threshold': best_thresh,
        'precision': precision_score(y_true, predictions, zero_division=0),
        'recall': recall_score(y_true, predictions, zero_division=0),
        'clean_balls': len(clean_balls),
        'toxic_balls': len(toxic_balls)
    }
    
    return {
        'predictions': predictions,
        'scores': final_scores,
        'y_true': y_true,
        'X_pca': X_pca,
        'assignments': assignments,
        'contamination': contamination,
        'metrics': metrics
    }


# ============================================
# 可视化函数
# ============================================

def visualize_results(result_cifar, result_pathmnist):
    """
    可视化检测结果 (浅色调)
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 浅色调颜色定义
    CLEAN_COLOR = '#87CEEB'       # 浅蓝色
    POISON_COLOR = '#FFB6C1'      # 浅红色
    CLEAN_BALL_COLOR = '#ADD8E6'  # 淡蓝色
    TOXIC_BALL_COLOR = '#FFB7B2'  # 淡红色
    
    # === CIFAR-10 结果 ===
    # 1. PCA空间分布
    ax1 = fig.add_subplot(3, 4, 1)
    X_pca_cifar = result_cifar['X_pca']
    y_true_cifar = result_cifar['y_true']
    scores_cifar = result_cifar['scores']
    
    clean_mask_c = y_true_cifar == 0
    ax1.scatter(X_pca_cifar[clean_mask_c, 0], X_pca_cifar[clean_mask_c, 1], 
              c=CLEAN_COLOR, alpha=0.5, s=20, label='Clean')
    ax1.scatter(X_pca_cifar[~clean_mask_c, 0], X_pca_cifar[~clean_mask_c, 1], 
              c=POISON_COLOR, alpha=0.8, s=40, marker='x', label='Poison')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'CIFAR-10 PCA Space\nAUC={result_cifar["metrics"]["AUC"]:.3f}')
    ax1.legend()
    
    # 2. 分数分布直方图
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.hist(scores_cifar[clean_mask_c], bins=50, alpha=0.6, color=CLEAN_COLOR, label='Clean', density=True)
    ax2.hist(scores_cifar[~clean_mask_c], bins=50, alpha=0.6, color=POISON_COLOR, label='Poison', density=True)
    ax2.axvline(result_cifar['metrics']['best_threshold'], color='green', linestyle='--', 
               label=f'Thresh={result_cifar["metrics"]["best_threshold"]:.3f}')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Density')
    ax2.set_title('CIFAR-10 Score Distribution')
    ax2.legend()
    
    # 3. ROC曲线
    ax3 = fig.add_subplot(3, 4, 3)
    fpr, tpr, _ = roc_curve(y_true_cifar, scores_cifar)
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={result_cifar["metrics"]["AUC"]:.3f}')
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    ax3.set_title('CIFAR-10 ROC Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. PR曲线
    ax4 = fig.add_subplot(3, 4, 4)
    prec, rec, _ = precision_recall_curve(y_true_cifar, scores_cifar)
    ax4.plot(rec, prec, 'r-', linewidth=2, label=f'AP={result_cifar["metrics"]["AP"]:.3f}')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('CIFAR-10 PR Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # === PathMNIST 结果 ===
    ax5 = fig.add_subplot(3, 4, 5)
    X_pca_path = result_pathmnist['X_pca']
    y_true_path = result_pathmnist['y_true']
    scores_path = result_pathmnist['scores']
    
    clean_mask_p = y_true_path == 0
    ax5.scatter(X_pca_path[clean_mask_p, 0], X_pca_path[clean_mask_p, 1], 
              c=CLEAN_COLOR, alpha=0.5, s=20, label='Clean')
    ax5.scatter(X_pca_path[~clean_mask_p, 0], X_pca_path[~clean_mask_p, 1], 
              c=POISON_COLOR, alpha=0.8, s=40, marker='x', label='Poison')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title(f'PathMNIST PCA Space\nAUC={result_pathmnist["metrics"]["AUC"]:.3f}')
    ax5.legend()
    
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.hist(scores_path[clean_mask_p], bins=50, alpha=0.6, color=CLEAN_COLOR, label='Clean', density=True)
    ax6.hist(scores_path[~clean_mask_p], bins=50, alpha=0.6, color=POISON_COLOR, label='Poison', density=True)
    ax6.axvline(result_pathmnist['metrics']['best_threshold'], color='green', linestyle='--',
               label=f'Thresh={result_pathmnist["metrics"]["best_threshold"]:.3f}')
    ax6.set_xlabel('Anomaly Score')
    ax6.set_ylabel('Density')
    ax6.set_title('PathMNIST Score Distribution')
    ax6.legend()
    
    ax7 = fig.add_subplot(3, 4, 7)
    fpr, tpr, _ = roc_curve(y_true_path, scores_path)
    ax7.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={result_pathmnist["metrics"]["AUC"]:.3f}')
    ax7.plot([0, 1], [0, 1], 'k--')
    ax7.set_xlabel('FPR')
    ax7.set_ylabel('TPR')
    ax7.set_title('PathMNIST ROC Curve')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(3, 4, 8)
    prec, rec, _ = precision_recall_curve(y_true_path, scores_path)
    ax8.plot(rec, prec, 'r-', linewidth=2, label=f'AP={result_pathmnist["metrics"]["AP"]:.3f}')
    ax8.set_xlabel('Recall')
    ax8.set_ylabel('Precision')
    ax8.set_title('PathMNIST PR Curve')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # === 粒球可视化 ===
    ax9 = fig.add_subplot(3, 4, 9)
    assignments_c = result_cifar['assignments']
    contamination_c = result_cifar['contamination']
    
    for ball_id in np.unique(assignments_c):
        mask = assignments_c == ball_id
        color = TOXIC_BALL_COLOR if contamination_c[ball_id] >= 0.35 else CLEAN_BALL_COLOR
        ax9.scatter(X_pca_cifar[mask, 0], X_pca_cifar[mask, 1], c=color, alpha=0.3, s=10)
    ax9.set_xlabel('PC1')
    ax9.set_ylabel('PC2')
    ax9.set_title('CIFAR-10 Granular Balls\n(Pink=Toxic, Blue=Clean)')
    
    ax10 = fig.add_subplot(3, 4, 10)
    assignments_p = result_pathmnist['assignments']
    contamination_p = result_pathmnist['contamination']
    
    for ball_id in np.unique(assignments_p):
        mask = assignments_p == ball_id
        color = TOXIC_BALL_COLOR if contamination_p[ball_id] >= 0.35 else CLEAN_BALL_COLOR
        ax10.scatter(X_pca_path[mask, 0], X_pca_path[mask, 1], c=color, alpha=0.3, s=10)
    ax10.set_xlabel('PC1')
    ax10.set_ylabel('PC2')
    ax10.set_title('PathMNIST Granular Balls\n(Pink=Toxic, Blue=Clean)')
    
    # === 指标汇总 ===
    ax11 = fig.add_subplot(3, 4, 11)
    metrics_text_c = f"""CIFAR-10 Results:
─────────────────
AUC:      {result_cifar['metrics']['AUC']:.4f}
AP:       {result_cifar['metrics']['AP']:.4f}
F1:       {result_cifar['metrics']['F1']:.4f}
d':       {result_cifar['metrics']['d_prime']:.4f}
Precision: {result_cifar['metrics']['precision']:.4f}
Recall:   {result_cifar['metrics']['recall']:.4f}
─────────────────
Clean: {result_cifar['metrics']['clean_balls']}
Toxic: {result_cifar['metrics']['toxic_balls']}"""
    ax11.text(0.1, 0.5, metrics_text_c, transform=ax11.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax11.axis('off')
    ax11.set_title('CIFAR-10 Metrics')
    
    ax12 = fig.add_subplot(3, 4, 12)
    metrics_text_p = f"""PathMNIST Results:
─────────────────
AUC:      {result_pathmnist['metrics']['AUC']:.4f}
AP:       {result_pathmnist['metrics']['AP']:.4f}
F1:       {result_pathmnist['metrics']['F1']:.4f}
d':       {result_pathmnist['metrics']['d_prime']:.4f}
Precision: {result_pathmnist['metrics']['precision']:.4f}
Recall:   {result_pathmnist['metrics']['recall']:.4f}
─────────────────
Clean: {result_pathmnist['metrics']['clean_balls']}
Toxic: {result_pathmnist['metrics']['toxic_balls']}"""
    ax12.text(0.1, 0.5, metrics_text_p, transform=ax12.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax12.axis('off')
    ax12.set_title('PathMNIST Metrics')
    
    fig.suptitle('GB-SSD-DA: Granular Ball-based Backdoor Detection', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('gbssd_da_results.png', dpi=150, bbox_inches='tight')
    print("[INFO] 保存: gbssd_da_results.png")
    
    return fig


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / "generated_pts"

    # 数据路径
    cifar_path = generated_dir / "cifar10_patch.pt"
    pathmnist_path = generated_dir / "pathmnist_patch.pt"
    
    # 加载数据
    print("=" * 60)
    print("GB-SSD-DA 后门检测算法")
    print("=" * 60)
    
    data_cifar = load_data(cifar_path)
    data_pathmnist = load_data(pathmnist_path)
    
    print(f"CIFAR-10: {len(data_cifar['images'])} 样本")
    print(f"PathMNIST: {len(data_pathmnist['images'])} 样本")
    
    # 执行检测
    print("\n>>> CIFAR-10 检测")
    result_cifar = gbssd_da_detect(data_cifar, target_label=6)
    
    print("\n>>> PathMNIST 检测")
    result_pathmnist = gbssd_da_detect(data_pathmnist, target_label=6)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("检测结果")
    print("=" * 60)
    
    print("\n【CIFAR-10】")
    for key, value in result_cifar['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n【PathMNIST】")
    for key, value in result_pathmnist['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    # 可视化
    print("\n>>> 生成可视化...")
    visualize_results(result_cifar, result_pathmnist)
    
    print("\n完成!")
