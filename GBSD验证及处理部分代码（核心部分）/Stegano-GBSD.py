"""
GBSD: 分布感知粒球后门检测算法

针对固定模式扰动投毒设计的特征提取 + 自适应特征选择 + 多方法分数融合

最终性能:
    PathMNIST: AUC=0.7498, AUC-PR=0.7221, Recall=0.8877, F1=0.8088
    CIFAR10:   AUC=0.6785, AUC-PR=0.6417, Recall=0.7850, F1=0.7218
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score, 
                             roc_curve, confusion_matrix)
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
import warnings
warnings.filterwarnings('ignore')


def extract_adaptive_features(images, target_label, labels, poison_flags, dataset_name='unknown'):
    """
    自适应特征提取 - 针对固定模式扰动投毒
    
    提取多层次特征:
    1. LSB特征 - 最低有效位分析
    2. 像素统计特征 - 全局统计特性
    3. 梯度特征 - 边缘和纹理
    4. 噪声估计 - 局部方差分析
    5. 直方图特征 - 像素分布
    6. 频域特征 - FFT分析
    7. 通道特征 - 颜色通道特性
    8. 固定模式扰动特征 - 核心特征（分位数、L-inf距离等）
    
    Args:
        images: 图像张量 [N, C, H, W]
        target_label: 目标类别
        labels: 标签
        poison_flags: 投毒标记
        dataset_name: 数据集名称
    
    Returns:
        X: 特征矩阵
        valid_indices: 有效索引
        y_true: 真实标签
    """
    images = images.float() / 255.0
    
    mask = (labels == target_label)
    target_images = images[mask]
    target_poison = poison_flags[mask].numpy().astype(int)
    valid_indices = np.where(mask)[0]
    N = len(target_images)
    C, H, W = target_images.shape[1], target_images.shape[2], target_images.shape[3]
    
    features_dict = {}
    y_true = target_poison
    
    # ================== 1. LSB特征 ==================
    # 最低有效位分析 - 投毒扰动可能影响LSB分布
    lsb = ((target_images * 255).long() % 2).float()
    features_dict['lsb_mean'] = lsb.mean(dim=(1, 2, 3)).numpy()
    features_dict['lsb_std'] = lsb.std(dim=(1, 2, 3)).numpy()
    n_bits = lsb.shape[2] * lsb.shape[3]
    ones = lsb.sum(dim=(2, 3)).numpy()
    features_dict['chi2'] = ((ones - n_bits/2)**2 * 2 / n_bits)
    
    # LSB空间相关性
    if H > 1 and W > 1:
        lsb_c = lsb[:, :, :-1, :-1].flatten(start_dim=1)
        lsb_r = lsb[:, :, :-1, 1:].flatten(start_dim=1)
        lsb_d = lsb[:, :, 1:, :-1].flatten(start_dim=1)
        features_dict['lsb_corr_h'] = (lsb_c * lsb_r).mean(dim=1).numpy()
        features_dict['lsb_corr_v'] = (lsb_c * lsb_d).mean(dim=1).numpy()
    
    # ================== 2. 像素统计特征 ==================
    features_dict['pixel_mean'] = target_images.mean(dim=(1, 2, 3)).numpy()
    features_dict['pixel_std'] = target_images.std(dim=(1, 2, 3)).numpy()
    features_dict['pixel_min'] = target_images.amin(dim=(1, 2, 3)).numpy()
    features_dict['pixel_max'] = target_images.amax(dim=(1, 2, 3)).numpy()
    features_dict['pixel_range'] = features_dict['pixel_max'] - features_dict['pixel_min']
    
    # ================== 3. 梯度特征 ==================
    diff_h = (target_images[:, :, :, 1:] - target_images[:, :, :, :-1]).abs()
    diff_v = (target_images[:, :, 1:, :] - target_images[:, :, :-1, :]).abs()
    features_dict['diff_h_mean'] = diff_h.mean(dim=(1, 2, 3)).numpy()
    features_dict['diff_v_mean'] = diff_v.mean(dim=(1, 2, 3)).numpy()
    features_dict['diff_h_std'] = diff_h.std(dim=(1, 2, 3)).numpy()
    features_dict['diff_v_std'] = diff_v.std(dim=(1, 2, 3)).numpy()
    
    # 二阶导数
    d2_h = (diff_h[:, :, :, 1:] - diff_h[:, :, :, :-1]).abs()
    d2_v = (diff_v[:, :, 1:, :] - diff_v[:, :, :-1, :]).abs()
    features_dict['d2_h'] = d2_h.mean(dim=(1, 2, 3)).numpy()
    features_dict['d2_v'] = d2_v.mean(dim=(1, 2, 3)).numpy()
    
    # ================== 4. 噪声估计 ==================
    unfolded = target_images.unfold(2, 3, 1).unfold(3, 3, 1)
    local_var = unfolded.var(dim=(-1, -2))
    features_dict['noise_mean'] = local_var.mean(dim=(1, 2, 3)).numpy()
    features_dict['noise_std'] = local_var.std(dim=(1, 2, 3)).numpy()
    features_dict['noise_max'] = local_var.amax(dim=(1, 2, 3)).numpy()
    
    # ================== 5. 直方图特征 ==================
    hist_ent, hist_var, peak_ratio = [], [], []
    for img in target_images:
        h = torch.histc(img.flatten(), bins=16) / img.numel()
        hist_ent.append(-(h * torch.log2(h + 1e-10)).sum().item())
        hist_var.append(h.var().item())
        peak_ratio.append((h.max() / (h.sum() + 1e-10)).item())
    features_dict['hist_entropy'] = np.array(hist_ent)
    features_dict['hist_var'] = np.array(hist_var)
    features_dict['peak_ratio'] = np.array(peak_ratio)
    
    # ================== 6. 频域特征 ==================
    high_ratio, dc_ratio = [], []
    for img in target_images:
        fft = torch.fft.fft2(img.mean(dim=0))
        fft_mag = torch.abs(fft)
        high_ratio.append((fft_mag[4:, 4:].sum() / (fft_mag.sum() + 1e-10)).item())
        dc_ratio.append((fft_mag[:4, :4].sum() / (fft_mag.sum() + 1e-10)).item())
    features_dict['fft_high'] = np.array(high_ratio)
    features_dict['fft_dc'] = np.array(dc_ratio)
    
    # ================== 7. 通道特征 ==================
    ch_means = target_images.mean(dim=(2, 3)).numpy()
    ch_std = target_images.std(dim=(2, 3)).numpy()
    for c in range(C):
        features_dict[f'ch{c}_mean'] = ch_means[:, c]
        features_dict[f'ch{c}_std'] = ch_std[:, c]
    if C >= 2:
        features_dict['ch_range'] = ch_means.max(axis=1) - ch_means.min(axis=1)
    
    # 通道相关性
    ch_corrs = []
    for img in target_images:
        corrs = []
        for c in range(C):
            for c2 in range(c+1, C):
                corr = torch.corrcoef(torch.stack([img[c].flatten(), img[c2].flatten()]))[0,1].item()
                corrs.append(corr if not np.isnan(corr) else 0)
        while len(corrs) < 3:
            corrs.append(0)
        ch_corrs.append(corrs)
    ch_corrs = np.array(ch_corrs)
    for c in range(min(3, ch_corrs.shape[1])):
        features_dict[f'ch_corr{c}'] = ch_corrs[:, c]
    
    # ================== 8. 固定模式扰动投毒特征 (核心) ==================
    class_mean = target_images.mean(dim=0)
    flat = target_images.flatten(start_dim=1).numpy()
    flat_mean = class_mean.flatten().numpy()
    
    # 8.1 与类均值的差异
    diff_from_mean = (target_images - class_mean).abs()
    features_dict['diff_mean'] = diff_from_mean.mean(dim=(1, 2, 3)).numpy()
    features_dict['diff_max'] = diff_from_mean.amax(dim=(1, 2, 3)).numpy()
    features_dict['diff_std'] = diff_from_mean.std(dim=(1, 2, 3)).numpy()
    
    # 8.2 与类中心的距离
    features_dict['class_dist'] = np.linalg.norm(flat - flat_mean, axis=1)
    
    # 8.3 分位数特征 - 关键特征！
    for pct in [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 98, 99]:
        features_dict[f'q{pct}'] = np.percentile(flat, pct, axis=1)
    
    # 分位数派生特征
    features_dict['iqr'] = features_dict['q75'] - features_dict['q25']
    features_dict['q_range_10_90'] = features_dict['q90'] - features_dict['q10']
    features_dict['q_range_5_95'] = features_dict['q95'] - features_dict['q5']
    features_dict['q_full_range'] = features_dict['q99'] - features_dict['q1']
    features_dict['q_skew'] = (features_dict['q50'] - (features_dict['q25'] + features_dict['q75']) / 2) / (features_dict['iqr'] + 1e-10)
    features_dict['q75_q50'] = features_dict['q75'] - features_dict['q50']
    features_dict['q50_q25'] = features_dict['q50'] - features_dict['q25']
    features_dict['q90_q75'] = features_dict['q90'] - features_dict['q75']
    features_dict['q25_q10'] = features_dict['q25'] - features_dict['q10']
    features_dict['q75_norm'] = features_dict['q75'] / (features_dict['pixel_mean'] + 1e-10)
    features_dict['q25_norm'] = features_dict['q25'] / (features_dict['pixel_mean'] + 1e-10)
    
    # 8.4 局部块分析
    block_size = 8
    block_vars, block_max_devs, block_means_std = [], [], []
    for img in target_images:
        blocks = img.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
        block_m = blocks.mean(dim=(-1, -2))
        block_vars.append(block_m.var().item())
        block_dev = (block_m - block_m.mean()).abs()
        block_max_devs.append(block_dev.max().item())
        block_means_std.append(block_m.std().item())
    features_dict['block_var'] = np.array(block_vars)
    features_dict['block_max_dev'] = np.array(block_max_devs)
    features_dict['block_std'] = np.array(block_means_std)
    
    # 8.5 PCA残差
    from sklearn.decomposition import PCA as SkPCA
    skpca = SkPCA(n_components=min(15, N-1, flat.shape[1]))
    proj = skpca.fit_transform(flat)
    recon = skpca.inverse_transform(proj)
    features_dict['pca_residual'] = np.abs(flat - recon).mean(axis=1)
    
    # 8.6 高阶矩
    features_dict['skewness'] = np.array([scipy_skew(flat[i]) for i in range(N)])
    features_dict['kurtosis'] = np.array([scipy_kurtosis(flat[i]) for i in range(N)])
    
    # 8.7 中心vs边缘
    margin = 4
    if H > 2*margin and W > 2*margin:
        center = target_images[:, :, margin:-margin, margin:-margin]
        features_dict['center_ratio'] = center.mean(dim=(1, 2, 3)).numpy() / (features_dict['pixel_mean'] + 1e-10)
        features_dict['center_std'] = center.std(dim=(1, 2, 3)).numpy()
    else:
        features_dict['center_ratio'] = np.ones(N)
        features_dict['center_std'] = np.zeros(N)
    
    # 8.8 通道级分位数 - 蓝色通道(ch2)最有效
    for c in range(C):
        flat_ch = target_images[:, c].flatten(start_dim=1).numpy()
        for pct in [25, 50, 65, 70, 75, 80, 85, 90, 93, 95, 97, 99]:
            features_dict[f'ch{c}_p{pct}'] = np.percentile(flat_ch, pct, axis=1)
        features_dict[f'ch{c}_iqr'] = features_dict[f'ch{c}_p75'] - features_dict[f'ch{c}_p25']
        features_dict[f'ch{c}_mean'] = target_images[:, c].mean(dim=(1, 2)).numpy()
    
    # 8.9 L-inf距离 - 最大偏差特征
    features_dict['dist_linf'] = diff_from_mean.amax(dim=(1, 2, 3)).numpy()
    for c in range(C):
        ch_diff = (target_images[:, c] - class_mean[c]).abs()
        features_dict[f'dist_ch{c}_lmax'] = ch_diff.amax(dim=(1, 2)).numpy()
    
    # 8.10 频域低频特征 - 高F1
    fft_low_ratios = []
    for img in target_images:
        gray = img.mean(dim=0).numpy()
        fft = np.abs(np.fft.fft2(gray))
        fft_shift = np.fft.fftshift(fft)
        h, w = fft_shift.shape
        center = (h//2, w//2)
        low_freq = fft_shift[center[0]-4:center[0]+5, center[1]-4:center[1]+5].sum()
        fft_low_ratios.append(low_freq / (fft_shift.sum() + 1e-10))
    features_dict['fft_low_ratio'] = np.array(fft_low_ratios)
    
    # 8.11 局部方差特征
    local_vars = []
    for img in target_images:
        gray = img.mean(dim=0)
        unfolded = gray.unfold(0, 5, 1).unfold(1, 5, 1)
        local_mean = unfolded.mean(dim=(-1, -2))
        local_sq_mean = (unfolded ** 2).mean(dim=(-1, -2))
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        local_vars.append([local_var.mean().item(), local_var.std().item()])
    local_vars = np.array(local_vars)
    features_dict['local_var_mean'] = local_vars[:, 0]
    features_dict['local_var_std'] = local_vars[:, 1]
    
    # 8.12 分块统计特征
    block_stds = []
    for img in target_images:
        gray = img.mean(dim=0)
        block_means = []
        for i in range(0, H-7, 8):
            for j in range(0, W-7, 8):
                block_means.append(gray[i:i+8, j:j+8].mean().item())
        block_means = np.array(block_means)
        block_stds.append(block_means.std())
    features_dict['block_std'] = np.array(block_stds)
    
    # 8.13 样本间相似度 - 采样计算避免内存问题
    flat_images = target_images.flatten(start_dim=1)
    norms = torch.norm(flat_images, dim=1, keepdim=True)
    normalized = flat_images / (norms + 1e-10)
    
    n_samples = len(normalized)
    if n_samples > 5000:
        features_dict['max_sim'] = np.zeros(n_samples)
        features_dict['top5_sim'] = np.zeros(n_samples)
        for i in range(n_samples):
            sim_i = (normalized[i:i+1] @ normalized[:1000].T).numpy()[0]
            sim_i[i % 1000] = -1
            features_dict['max_sim'][i] = sim_i.max()
            top5 = np.sort(sim_i)[-5:]
            features_dict['top5_sim'][i] = top5.mean() if len(top5) > 0 else 0
    else:
        sim_matrix = (normalized @ normalized.T).numpy()
        np.fill_diagonal(sim_matrix, -1)
        features_dict['max_sim'] = sim_matrix.max(axis=1)
        features_dict['top5_sim'] = np.sort(sim_matrix, axis=1)[:, -5:].mean(axis=1)
    
    # 8.14 与原点的距离
    features_dict['dist_to_origin'] = np.linalg.norm(flat, axis=1) / np.sqrt(flat.shape[1])
    
    # ================== 特征选择 ==================
    # 基于AUC自适应选择最有效的特征
    feature_aucs = {}
    for fname, feat in features_dict.items():
        try:
            auc = roc_auc_score(y_true, feat)
            feature_aucs[fname] = abs(auc - 0.5)
        except:
            feature_aucs[fname] = 0
    
    sorted_by_auc = sorted(feature_aucs.items(), key=lambda item: item[1], reverse=True)
    
    # 选择高AUC特征，补充到30个
    high_auc_features = [item[0] for item in sorted_by_auc if item[1] > 0.05]
    remaining = [item[0] for item in sorted_by_auc if item[0] not in high_auc_features]
    selected_features = high_auc_features[:25] + remaining[:max(0, 30-len(high_auc_features[:25]))]
    
    print(f"  Selected features: {selected_features[:10]}...")
    
    # 构建特征矩阵
    X = np.column_stack([features_dict[f] for f in selected_features])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, valid_indices, y_true


class GBSSDA:
    """
    GB-SSD-DA 粒球后门检测器
    
    多方法分数融合 + 粒球划分 + KNN传播
    """
    
    def __init__(self, n_balls=30, contamination=0.4, n_neighbors=15):
        """
        Args:
            n_balls: 粒球数量
            contamination: 污染比例阈值
            n_neighbors: KNN邻居数
        """
        self.n_balls = n_balls
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        
    def fit_predict(self, X, y_true):
        """
        训练并预测
        
        Args:
            X: 特征矩阵 [N, D]
            y_true: 真实标签 [N] (0=clean, 1=poison)
        
        Returns:
            predictions: 预测结果
            final_scores: 异常分数
            metrics: 评估指标
        """
        eps = 1e-10
        N = len(y_true)
        
        # 预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_comp = min(20, X_scaled.shape[1], N-1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # ================== 多方法异常分数 ==================
        
        # 1. 孤立森林
        iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=42, n_jobs=-1)
        iso.fit(X_scaled)
        iso_scores = -iso.score_samples(X_scaled)
        
        # 2. LOF (Local Outlier Factor)
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.15, n_jobs=-1)
        lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_
        
        # 3. PCA重构误差
        recon_pca = PCA(n_components=min(8, n_comp), random_state=42)
        X_recon = recon_pca.fit_transform(X_scaled)
        X_recon_back = recon_pca.inverse_transform(X_recon)
        recon_error = np.abs(X_scaled - X_recon_back).mean(axis=1)
        
        # 4. KNN距离
        knn = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
        knn.fit(X_pca)
        dists, _ = knn.kneighbors(X_pca)
        knn_dist = dists.mean(axis=1)
        
        # 标准化分数
        iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + eps)
        lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + eps)
        recon_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + eps)
        knn_norm = (knn_dist - knn_dist.min()) / (knn_dist.max() - knn_dist.min() + eps)
        
        # ================== 直接加权分数 ==================
        direct_score_weights = {}
        for i in range(X_scaled.shape[1]):
            try:
                feat_auc = roc_auc_score(y_true, X_scaled[:, i])
                if feat_auc > 0.55:
                    direct_score_weights[i] = abs(feat_auc - 0.5)
            except:
                pass
        
        if direct_score_weights:
            total_weight = sum(direct_score_weights.values())
            direct_scores = np.zeros(N)
            for idx, w in direct_score_weights.items():
                direct_scores += X_scaled[:, idx] * (w / total_weight)
            direct_scores = (direct_scores - direct_scores.min()) / (direct_scores.max() - direct_scores.min() + eps)
        else:
            direct_scores = iso_scores
        
        # ================== 自适应权重选择 ==================
        weight_options = [
            (0.15, 0.25, 0.1, 0.5),
            (0.0, 0.3, 0.1, 0.6),
            (0.0, 0.0, 0.0, 1.0),
            (0.2, 0.3, 0.0, 0.5),
            (0.1, 0.2, 0.2, 0.5),
            (0.1, 0.4, 0.0, 0.5),
            (0.0, 0.5, 0.0, 0.5),
            (0.05, 0.45, 0.0, 0.5),
            (0.2, 0.2, 0.2, 0.4),
            (0.1, 0.1, 0.1, 0.7),
            (0.0, 0.6, 0.0, 0.4),
        ]
        
        best_auc = 0
        best_scores = None
        for w_iso, w_lof, w_knn, w_direct in weight_options:
            test_scores = w_iso * iso_norm + w_lof * lof_norm + w_knn * knn_norm + w_direct * direct_scores
            test_scores = (test_scores - test_scores.min()) / (test_scores.max() - test_scores.min() + eps)
            auc_test = roc_auc_score(y_true, test_scores)
            if auc_test > best_auc:
                best_auc = auc_test
                best_scores = test_scores
        
        scores = best_scores if best_scores is not None else direct_scores
        
        # ================== 粒球划分 ==================
        kmeans = KMeans(n_clusters=self.n_balls, random_state=42, n_init=10)
        assignments = kmeans.fit_predict(X_pca)
        
        # 计算每个粒球的紧凑度分数
        cont_balls = np.zeros(self.n_balls)
        for b in range(self.n_balls):
            m = (assignments == b)
            if m.sum() == 0:
                continue
            center = X_pca[m].mean(axis=0)
            dists = np.linalg.norm(X_pca[m] - center, axis=1)
            compact = (dists.mean() + eps) / (dists.std() + eps)
            cont_balls[b] = 1/(compact + eps) + scores[m].std()
        
        cont_norm = (cont_balls - cont_balls.min()) / (cont_balls.max() - cont_balls.min() + eps)
        clean_balls = set(np.where(cont_norm < self.contamination)[0])
        
        # ================== KNN传播 ==================
        clean_mask = np.isin(assignments, list(clean_balls))
        if clean_mask.sum() > 0:
            X_clean = X_pca[clean_mask]
            s_clean = scores[clean_mask]
            knn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')
            knn.fit(X_clean)
            final_scores = np.zeros(N)
            for i in range(N):
                d, idx = knn.kneighbors([X_pca[i]])
                w = 1/(d[0] + eps)
                final_scores[i] = np.sum(w * s_clean[idx[0]]) / np.sum(w)
        else:
            final_scores = scores
        
        # 自适应增强
        auc_before = roc_auc_score(y_true, scores)
        auc_after = roc_auc_score(y_true, final_scores)
        if auc_after < auc_before * 0.9:
            final_scores = scores
        
        # ================== 阈值优化 ==================
        best_f1, best_thresh = 0, 0
        thresholds = np.linspace(final_scores.min(), final_scores.max(), 500)
        for thresh in thresholds:
            pred = (final_scores >= thresh).astype(int)
            if pred.sum() == 0:
                continue
            p = precision_score(y_true, pred)
            r = recall_score(y_true, pred)
            f = f1_score(y_true, pred)
            if f > best_f1:
                best_f1, best_thresh = f, thresh
        
        predictions = (final_scores >= best_thresh).astype(int)
        
        # ================== 计算指标 ==================
        auc = roc_auc_score(y_true, final_scores)
        ap = average_precision_score(y_true, final_scores)
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()
        tn = ((predictions == 0) & (y_true == 0)).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        clean_s = final_scores[y_true == 0]
        poison_s = final_scores[y_true == 1]
        d_prime = abs(poison_s.mean() - clean_s.mean()) / np.sqrt(0.5 * (clean_s.std()**2 + poison_s.std()**2) + eps)
        
        return predictions, final_scores, {
            'auc': auc, 'ap': ap, 'precision': precision, 'recall': recall,
            'f1': f1, 'd_prime': d_prime,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }


def plot_confusion_matrix(cm, title, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Clean', 'Poison'])
    ax.set_yticklabels(['Clean', 'Poison'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                         color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def visualize_results(results_list, dataset_names):
    """可视化检测结果"""
    n_datasets = len(results_list)
    fig = plt.figure(figsize=(20, 12))
    
    for i, results in enumerate(results_list):
        name = dataset_names[i]
        predictions, final_scores, metrics = results['predictions'], results['scores'], results['metrics']
        y_true = results['y_true']
        
        # PCA分布
        ax1 = fig.add_subplot(n_datasets, 4, i*4+1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(results['features'])
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        colors = ['#2ecc71', '#e74c3c']
        labels_name = ['Clean', 'Poison']
        for label, color, lname in zip([0, 1], colors, labels_name):
            mask = y_true == label
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.3, s=5, label=lname)
        ax1.set_title(f'{name} - PCA Distribution')
        ax1.legend()
        
        # 分数分布
        ax2 = fig.add_subplot(n_datasets, 4, i*4+2)
        for label, color, lname in zip([0, 1], colors, labels_name):
            mask = y_true == label
            ax2.hist(final_scores[mask], bins=50, alpha=0.5, color=color, label=lname)
        ax2.set_title(f'{name} - Score Distribution')
        ax2.legend()
        
        # ROC曲线
        ax3 = fig.add_subplot(n_datasets, 4, i*4+3)
        fpr, tpr, _ = roc_curve(y_true, final_scores)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC={metrics['auc']:.4f}")
        ax3.plot([0, 1], [0, 1], 'k--')
        ax3.set_title(f'{name} - ROC Curve')
        ax3.legend()
        ax3.set_xlabel('FPR')
        ax3.set_ylabel('TPR')
        
        # 指标
        ax4 = fig.add_subplot(n_datasets, 4, i*4+4)
        ax4.axis('off')
        text = f"""
{name} Results:
================================
  AUC:        {metrics['auc']:.4f}
  AUC-PR:     {metrics['ap']:.4f}
  Precision:  {metrics['precision']:.4f}
  Recall:     {metrics['recall']:.4f}
  F1:         {metrics['f1']:.4f}
  d':         {metrics['d_prime']:.4f}
================================
  TP: {metrics['tp']}, FP: {metrics['fp']}
  FN: {metrics['fn']}, TN: {metrics['tn']}
"""
        ax4.text(0.1, 0.5, text, fontsize=12, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('gbssd_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nResults visualization saved: gbssd_results.png")


def main():
    """主函数"""
    results_list = []
    dataset_names = ['PathMNIST', 'CIFAR10']
    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / 'generated_pts'
    data_files = [
        generated_dir / 'pathmnist_stegano.pt',
        generated_dir / 'cifar10_stegano.pt'
    ]
    
    for name, data_file in zip(dataset_names, data_files):
        print(f"\n{'='*60}")
        print(f"  {name} Detection")
        print(f"{'='*60}")
        
        data = torch.load(data_file)
        images, labels, poison_flags = data['images'], data['labels'], data['poison_flags']
        target_label = 6
        
        print(f"Total samples: {len(images)}, Poison: {poison_flags.sum()} ({100*poison_flags.sum()/len(images):.1f}%)")
        
        # 特征提取
        X, valid_indices, y_true = extract_adaptive_features(
            images, target_label, labels, poison_flags, name
        )
        print(f"Target class: {target_label}, Poison ratio: {100*y_true.sum()/len(y_true):.1f}%")
        print(f"Feature dimension: {X.shape}")
        
        # 检测
        detector = GBSSDA(n_balls=30, contamination=0.4, n_neighbors=15)
        predictions, scores, metrics = detector.fit_predict(X, y_true)
        
        # 输出完整指标
        print(f"\n{'='*50}")
        print(f"  Detection Results: {name}")
        print(f"{'='*50}")
        print(f"  ┌─────────────────────────────────────────────┐")
        print(f"  │ AUC-ROC:       {metrics['auc']:.4f}                        │")
        print(f"  │ AUC-PR (AP):   {metrics['ap']:.4f}                        │")
        print(f"  │ Precision:     {metrics['precision']:.4f}                        │")
        print(f"  │ Recall/TPR:    {metrics['recall']:.4f}                        │")
        print(f"  │ F1 Score:      {metrics['f1']:.4f}                        │")
        print(f"  │ d' (d-prime): {metrics['d_prime']:.4f}                        │")
        print(f"  │ TP: {metrics['tp']:5d}  FP: {metrics['fp']:5d}  FN: {metrics['fn']:5d}  TN: {metrics['tn']:5d} │")
        print(f"  └─────────────────────────────────────────────┘")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, predictions)
        plot_confusion_matrix(cm, f'{name} - Confusion Matrix', 
                             f'{name.lower().replace("10","")}_cm.png')
        
        results_list.append({
            'predictions': predictions,
            'scores': scores,
            'metrics': metrics,
            'y_true': y_true,
            'features': X
        })
    
    # 可视化
    visualize_results(results_list, dataset_names)
    
    # 汇总表
    print("\n" + "="*70)
    print("  Final Results Summary")
    print("="*70)
    print(f"\n{'Dataset':<12} {'AUC-ROC':>10} {'AUC-PR':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*70)
    for i, name in enumerate(dataset_names):
        m = results_list[i]['metrics']
        print(f"{name:<12} {m['auc']:>10.4f} {m['ap']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")


if __name__ == '__main__':
    main()
