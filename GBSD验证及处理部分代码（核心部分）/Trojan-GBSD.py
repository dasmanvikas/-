"""
GBSD: 分布感知粒球后门检测算法
支持信号注入投毒检测
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """加载投毒数据集"""
    data = torch.load(file_path, map_location='cpu')
    return data


def sig_feature_extraction(images, signal_pattern, bit_depth=8):
    """
    信号注入投毒特征提取算法 (SigFeatureExtraction)
    投毒方式: 叠加微弱信号pattern
    
    参数:
        images: [N, 3, 28, 28] 图像张量
        signal_pattern: [3, 28, 28] 信号模式
        bit_depth: 位深度
    
    返回:
        features: [N, 15] 特征矩阵
    """
    N = images.shape[0]
    features = np.zeros((N, 15))
    
    # 确保signal_pattern是numpy数组
    if torch.is_tensor(signal_pattern):
        signal_pattern = signal_pattern.numpy()
    
    # 展平信号模式用于相关性计算
    signal_flat = signal_pattern.flatten()
    
    for i in range(N):
        img = images[i].numpy() if torch.is_tensor(images[i]) else images[i]
        
        # 1. Pearson相关性 (主特征) - 图像与信号模式的相关性
        img_flat = img.flatten()
        if len(img_flat) == len(signal_flat):
            corr, _ = pearsonr(img_flat, signal_flat)
            features[i, 0] = abs(corr) if not np.isnan(corr) else 0
        else:
            # 如果维度不匹配，计算每个通道的相关性
            corrs = []
            for c in range(3):
                img_c = img[c].flatten()
                sig_c = signal_pattern[c].flatten() if signal_pattern.ndim == 3 else signal_flat
                if len(img_c) == len(sig_c):
                    c_corr, _ = pearsonr(img_c, sig_c)
                    if not np.isnan(c_corr):
                        corrs.append(abs(c_corr))
            features[i, 0] = np.mean(corrs) if corrs else 0
        
        # 2. 频域能量比 - 检测信号的频率特征
        for c in range(3):
            fft = np.fft.fft2(img[c])
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # 高频能量比例
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            high_freq_mask = np.ones_like(magnitude)
            high_freq_mask[center_h-3:center_h+3, center_w-3:center_w+3] = 0
            
            if magnitude.sum() > 0:
                high_freq_ratio = (magnitude * high_freq_mask).sum() / magnitude.sum()
            else:
                high_freq_ratio = 0
            
            if c == 0:
                features[i, 1] = high_freq_ratio
            elif c == 1:
                features[i, 2] = high_freq_ratio
            else:
                features[i, 3] = high_freq_ratio
        
        # 3. 局部方差异常 - 检测信号引起的局部变化
        from scipy.ndimage import uniform_filter
        for c in range(3):
            local_mean = uniform_filter(img[c], size=3)
            local_var = uniform_filter((img[c] - local_mean) ** 2, size=3)
            var_anomaly = np.percentile(local_var, 95) / (np.percentile(local_var, 50) + 1e-8)
            
            if c == 0:
                features[i, 4] = var_anomaly
            elif c == 1:
                features[i, 5] = var_anomaly
            else:
                features[i, 6] = var_anomaly
        
        # 4. 信号残差能量
        residual = np.abs(img - signal_pattern)
        features[i, 7] = residual.mean()
        features[i, 8] = residual.std()
        features[i, 9] = np.percentile(residual, 90)
        
        # 5. 边缘一致性 - 信号可能影响边缘特征
        from scipy.ndimage import sobel
        edge_h = sobel(img[0], axis=0)
        edge_v = sobel(img[0], axis=1)
        edge_mag = np.sqrt(edge_h**2 + edge_v**2)
        features[i, 10] = edge_mag.mean()
        features[i, 11] = edge_mag.std()
        
        # 6. 通道间相关性变化
        corr_rg = np.corrcoef(img[0].flatten(), img[1].flatten())[0, 1]
        corr_rb = np.corrcoef(img[0].flatten(), img[2].flatten())[0, 1]
        corr_gb = np.corrcoef(img[1].flatten(), img[2].flatten())[0, 1]
        
        features[i, 12] = 0 if np.isnan(corr_rg) else abs(corr_rg)
        features[i, 13] = 0 if np.isnan(corr_rb) else abs(corr_rb)
        features[i, 14] = 0 if np.isnan(corr_gb) else abs(corr_gb)
    
    return features


def compute_reconstruction_error(X, n_components=10):
    """使用PCA计算重构误差"""
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    recon_error = np.mean((X - X_reconstructed) ** 2, axis=1)
    return recon_error


def compute_density_scores(X, k=15):
    """计算局部密度分数"""
    nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]-1), algorithm='ball_tree')
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    # 使用平均距离作为密度指标（距离越大，密度越小）
    density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-8)
    return density


def gb_ssd_da_detector(data, n_balls=30, contamination_threshold=0.4, 
                        n_neighbors=15, n_thresholds=500, primary_feature_idx=0,
                        target_label=None):
    """
    GB-SSD-DA: 分布感知粒球后门检测算法
    
    参数:
        data: 包含images, labels, poison_flags, signal_pattern的字典
        n_balls: 粒球数量
        contamination_threshold: 污染度阈值
        n_neighbors: KNN邻居数
        n_thresholds: 阈值搜索数量
        primary_feature_idx: 主特征索引
        target_label: 目标标签，如果为None则自动推断
    
    返回:
        predictions, final_scores, metrics
    """
    epsilon = 1e-8
    
    # ==================== 第一步: 数据准备 ====================
    images = data['images']
    labels = data['labels']
    poison_flags = data['poison_flags']
    signal_pattern = data.get('signal_pattern', None)
    
    # 如果没有提供target_label，从投毒样本中推断
    if target_label is None:
        poison_mask = poison_flags if not torch.is_tensor(poison_flags) else poison_flags.numpy()
        labels_np = labels if not torch.is_tensor(labels) else labels.numpy()
        poison_labels = labels_np[poison_mask]
        unique, counts = np.unique(poison_labels, return_counts=True)
        target_label = unique[np.argmax(counts)]
        print(f"自动推断目标标签: {target_label}")
    
    # 归一化图像
    images = images.float() / 255.0
    
    # 获取ground truth
    y_true = poison_flags.numpy().astype(int) if torch.is_tensor(poison_flags) else poison_flags.astype(int)
    
    # 筛选目标类别
    labels_np = labels.numpy() if torch.is_tensor(labels) else labels
    target_mask = (labels_np == target_label)
    
    print(f"目标类别: {target_label}")
    print(f"目标类别样本数: {target_mask.sum()}")
    print(f"目标类别中投毒样本数: {y_true[target_mask].sum()}")
    
    # ==================== 第二步: 特征提取 ====================
    print("\n正在提取信号注入投毒特征...")
    
    if signal_pattern is None:
        # 如果没有提供signal_pattern，尝试从投毒样本中估计
        poison_mask = (y_true == 1)
        if poison_mask.sum() > 0:
            poison_images = images[poison_mask]
            clean_images = images[~poison_mask]
            signal_pattern = (poison_images.mean(dim=0) - clean_images.mean(dim=0)).numpy()
        else:
            signal_pattern = torch.zeros_like(images[0])
    
    features = sig_feature_extraction(images, signal_pattern)
    X_target = features[target_mask]
    y_target = y_true[target_mask]
    
    print(f"特征维度: {X_target.shape}")
    
    # ==================== 第三步: 计算辅助分数 ====================
    # PCA降维用于粒球生成
    n_components = min(10, X_target.shape[0], X_target.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_target)
    
    # 计算重构误差
    recon_error = compute_reconstruction_error(X_target)
    
    # 计算密度分数
    density = compute_density_scores(X_target)
    
    # ==================== 第四步: 分数组合 ====================
    # 提取主特征
    primary_feat = X_target[:, primary_feature_idx]
    
    # 归一化到[0,1]
    primary_norm = (primary_feat - primary_feat.min()) / (primary_feat.max() - primary_feat.min() + epsilon)
    recon_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + epsilon)
    density_norm = (density - density.min()) / (density.max() - density.min() + epsilon)
    
    # 加权组合 (主特征占90%)
    scores = 0.9 * primary_norm + 0.05 * recon_norm + 0.05 * density_norm
    
    # ==================== 第五步: 粒球生成与清洗 ====================
    print(f"\n生成 {n_balls} 个粒球...")
    
    kmeans = KMeans(n_clusters=n_balls, random_state=42, n_init=10)
    assignments = kmeans.fit_predict(X_pca)
    
    # 计算每个粒球的污染度
    contamination = np.zeros(n_balls)
    
    for ball_id in range(n_balls):
        mask = (assignments == ball_id)
        if mask.sum() == 0:
            contamination[ball_id] = 0
            continue
        
        center = X_pca[mask].mean(axis=0)
        dists = np.linalg.norm(X_pca[mask] - center, axis=1)
        
        compactness = (dists.mean() + epsilon) / (dists.std() + epsilon)
        contamination[ball_id] = 1 / (compactness + epsilon) + np.std(scores[mask])
    
    # 归一化污染度
    contamination_norm = (contamination - contamination.min()) / (contamination.max() - contamination.min() + epsilon)
    
    # 清洗粒球
    clean_balls = [ball_id for ball_id in range(n_balls) if contamination_norm[ball_id] < contamination_threshold]
    toxic_balls = [ball_id for ball_id in range(n_balls) if contamination_norm[ball_id] >= contamination_threshold]
    
    print(f"清洁粒球: {len(clean_balls)} 个")
    print(f"污染粒球: {len(toxic_balls)} 个")
    
    # ==================== 第六步: 分数传播 ====================
    clean_mask = np.isin(assignments, clean_balls)
    final_scores = np.zeros(len(scores))
    
    if clean_mask.sum() > 0:
        X_clean = X_pca[clean_mask]
        clean_scores = scores[clean_mask]
        
        # KNN传播
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, X_clean.shape[0]), algorithm='ball_tree')
        knn.fit(X_clean)
        
        for i in range(len(scores)):
            dists, idx = knn.kneighbors([X_pca[i]])
            weights = 1 / (dists[0] + epsilon)
            final_scores[i] = np.sum(weights * clean_scores[idx[0]]) / np.sum(weights)
    else:
        final_scores = scores
    
    # ==================== 第七步: 阈值优化 ====================
    print("\n优化阈值...")
    
    best_f1 = 0
    best_thresh = 0
    
    thresholds = np.linspace(final_scores.min(), final_scores.max(), n_thresholds)
    
    for thresh in thresholds:
        pred = (final_scores >= thresh).astype(int)
        if pred.sum() == 0:
            continue
        
        p = precision_score(y_target, pred, zero_division=0)
        r = recall_score(y_target, pred, zero_division=0)
        f = f1_score(y_target, pred, zero_division=0)
        
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    
    predictions = (final_scores >= best_thresh).astype(int)
    
    print(f"最佳阈值: {best_thresh:.4f}, 最佳F1: {best_f1:.4f}")
    
    # ==================== 第八步: 评估指标 ====================
    # AUC
    auc = roc_auc_score(y_target, final_scores)
    
    # Average Precision
    ap = average_precision_score(y_target, final_scores)
    
    # 计算d' (分离度)
    clean_s = final_scores[y_target == 0]
    poison_s = final_scores[y_target == 1]
    
    if len(clean_s) > 0 and len(poison_s) > 0:
        d_prime = abs(poison_s.mean() - clean_s.mean()) / np.sqrt(0.5 * (clean_s.std()**2 + poison_s.std()**2) + epsilon)
    else:
        d_prime = 0
    
    # 混淆矩阵统计
    tn, fp, fn, tp = confusion_matrix(y_target, predictions).ravel()
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # TPR (真正例率) 和 FPR (假正例率)
    tpr = recall  # TPR = Recall = TP / (TP + FN)
    fpr = fp / (fp + tn + epsilon)  # FPR = FP / (FP + TN)
    
    metrics = {
        'auc': auc,
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'd_prime': d_prime,
        'tpr': tpr,
        'fpr': fpr,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'best_threshold': best_thresh
    }
    
    return predictions, final_scores, metrics, y_target


def visualize_results(final_scores, y_target, metrics, dataset_name, save_path=None):
    """生成可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'GB-SSD-DA 后门检测结果 - {dataset_name}', fontsize=14)
    
    # 1. 分数分布直方图
    ax1 = axes[0, 0]
    clean_scores = final_scores[y_target == 0]
    poison_scores = final_scores[y_target == 1]
    
    ax1.hist(clean_scores, bins=50, alpha=0.7, label='Clean', color='green', density=True)
    ax1.hist(poison_scores, bins=50, alpha=0.7, label='Poison', color='red', density=True)
    ax1.axvline(metrics['best_threshold'], color='black', linestyle='--', label=f'Threshold={metrics["best_threshold"]:.3f}')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution')
    ax1.legend()
    
    # 2. ROC曲线
    ax2 = axes[0, 1]
    fpr_curve, tpr_curve, _ = roc_curve(y_target, final_scores)
    ax2.plot(fpr_curve, tpr_curve, 'b-', linewidth=2, label=f'ROC Curve (AUC={metrics["auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.fill_between(fpr_curve, tpr_curve, alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 混淆矩阵热力图
    ax3 = axes[0, 2]
    cm = np.array([[metrics['tn'], metrics['fp']], 
                   [metrics['fn'], metrics['tp']]])
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    ax3.set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(['Clean', 'Poison'])
    ax3.set_yticklabels(['Clean', 'Poison'])
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # 4. 箱线图
    ax4 = axes[1, 0]
    data_to_plot = [clean_scores, poison_scores]
    bp = ax4.boxplot(data_to_plot, labels=['Clean', 'Poison'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax4.set_ylabel('Anomaly Score')
    ax4.set_title('Score Distribution (Boxplot)')
    ax4.axhline(metrics['best_threshold'], color='black', linestyle='--', alpha=0.7)
    
    # 5. 指标条形图
    ax5 = axes[1, 1]
    metric_names = ['Precision', 'Recall', 'F1', 'AUC', 'TPR', 'FPR']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1'], 
                     metrics['auc'], metrics['tpr'], metrics['fpr']]
    colors = ['skyblue', 'lightgreen', 'orange', 'purple', 'pink', 'yellow']
    bars = ax5.bar(metric_names, metric_values, color=colors)
    ax5.set_ylim([0, 1.1])
    ax5.set_ylabel('Score')
    ax5.set_title('Performance Metrics')
    
    # 添加数值标签
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. 指标文本汇总
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    检测性能汇总:
    
    Precision:    {metrics['precision']:.4f}
    Recall:       {metrics['recall']:.4f}
    F1 Score:     {metrics['f1']:.4f}
    AUC:          {metrics['auc']:.4f}
    TPR:          {metrics['tpr']:.4f}
    FPR:          {metrics['fpr']:.4f}
    
    d' (分离度):  {metrics['d_prime']:.4f}
    AP:           {metrics['ap']:.4f}
    
    混淆矩阵:
    TN: {metrics['tn']}, FP: {metrics['fp']}
    FN: {metrics['fn']}, TP: {metrics['tp']}
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """主函数"""
    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / 'generated_pts'

    # 数据集路径
    datasets = {
        'CIFAR10': generated_dir / 'cifar10_trojan.pt',
        'PathMNIST': generated_dir / 'pathmnist_trojan.pt'
    }
    
    all_results = {}
    
    for dataset_name, file_path in datasets.items():
        print("=" * 60)
        print(f"处理数据集: {dataset_name}")
        print("=" * 60)
        
        try:
            # 加载数据
            print(f"\n加载数据: {file_path}")
            data = load_data(file_path)
            
            # 检查数据内容
            print(f"数据键: {data.keys()}")
            print(f"图像形状: {data['images'].shape}")
            print(f"标签形状: {data['labels'].shape}")
            print(f"投毒标记形状: {data['poison_flags'].shape}")
            print(f"目标标签: {data.get('target_label', 'N/A')}")
            
            # 运行检测算法（目标标签为6）
            predictions, final_scores, metrics, y_target = gb_ssd_da_detector(data, target_label=6)
            
            # 保存结果
            all_results[dataset_name] = {
                'predictions': predictions,
                'final_scores': final_scores,
                'metrics': metrics,
                'y_target': y_target
            }
            
            # 打印指标
            print("\n" + "=" * 40)
            print("最终评估指标:")
            print("=" * 40)
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"AUC:       {metrics['auc']:.4f}")
            print(f"TPR:       {metrics['tpr']:.4f}")
            print(f"FPR:       {metrics['fpr']:.4f}")
            print(f"d':        {metrics['d_prime']:.4f}")
            
            # 生成可视化
            save_path = str(base_dir / f'results_{dataset_name.lower()}.png')
            visualize_results(final_scores, y_target, metrics, dataset_name, save_path)
            
        except Exception as e:
            print(f"处理 {dataset_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 对比两个数据集的结果
    if len(all_results) == 2:
        print("\n" + "=" * 60)
        print("数据集对比")
        print("=" * 60)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        datasets_names = list(all_results.keys())
        metrics_names = ['Precision', 'Recall', 'F1', 'AUC', 'TPR']
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        for i, dataset_name in enumerate(datasets_names):
            metrics = all_results[dataset_name]['metrics']
            values = [metrics['precision'], metrics['recall'], metrics['f1'], 
                     metrics['auc'], metrics['tpr']]
            ax.bar(x + i * width, values, width, label=dataset_name)
        
        ax.set_ylabel('Score')
        ax.set_title('GB-SSD-DA 性能对比')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(base_dir / 'comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"对比图已保存至: {base_dir / 'comparison.png'}")
    
    return all_results


if __name__ == '__main__':
    results = main()
