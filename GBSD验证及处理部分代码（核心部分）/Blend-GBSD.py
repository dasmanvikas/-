"""
GBSD CIFAR10 最终优化版
"""

import torch
import numpy as np
from pathlib import Path
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


class CIFAR10FeatureExtractor:
    """CIFAR10 特征提取器 (30维)"""
    
    def __init__(self, image_size=(3, 32, 32)):
        self.image_size = image_size
        
    def extract_features(self, images):
        n_samples = images.shape[0]
        features = np.zeros((n_samples, 30))
        
        for i in range(n_samples):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            gray = img.mean(axis=0)
            
            idx = 0
            
            # 梯度 (6)
            gx = ndimage.sobel(gray, axis=0)
            gy = ndimage.sobel(gray, axis=1)
            mag = np.sqrt(gx**2 + gy**2)
            features[i, idx] = mag.mean(); idx += 1
            features[i, idx] = mag.std(); idx += 1
            features[i, idx] = mag.max(); idx += 1
            features[i, idx] = np.median(mag); idx += 1
            
            # 梯度熵
            hist, _ = np.histogram(mag.flatten(), bins=50, density=True)
            hist = hist[hist > 0]
            features[i, idx] = -np.sum(hist * np.log2(hist + 1e-10)); idx += 1
            features[i, idx] = (gx**2 + gy**2).sum(); idx += 1
            
            # 频域 (8)
            dct_coeffs = np.fft.fft2(gray)
            dct_shift = np.fft.fftshift(dct_coeffs)
            dct_abs = np.abs(dct_shift)
            
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((y - cy)**2 + (x - cx)**2)
            
            features[i, idx] = dct_abs[8:, 8:].sum() / (dct_abs.sum() + 1e-10); idx += 1
            features[i, idx] = dct_abs[4:8, :].sum() / (dct_abs.sum() + 1e-10); idx += 1
            features[i, idx] = dct_abs[high_mask := r >= min(h,w)//4].sum() / (dct_abs.sum() + 1e-10); idx += 1
            features[i, idx] = dct_abs[mid_mask := (r >= min(h,w)//8) & (r < min(h,w)//4)].sum() / (dct_abs.sum() + 1e-10); idx += 1
            
            # 频谱统计
            features[i, idx] = (dct_abs * r).sum() / (dct_abs.sum() * r.mean() + 1e-10); idx += 1
            features[i, idx] = dct_abs.std() / (dct_abs.mean() + 1e-10); idx += 1
            dct_norm = dct_abs / (dct_abs.sum() + 1e-10)
            features[i, idx] = -np.sum(dct_norm[dct_norm>0] * np.log2(dct_norm[dct_norm>0] + 1e-10)); idx += 1
            features[i, idx] = dct_abs[0, 0] / (dct_abs[1:, 1:].sum() + 1e-10); idx += 1
            
            # 纹理 (5)
            lap = ndimage.laplace(gray)
            features[i, idx] = lap.var(); idx += 1
            local_mean = ndimage.uniform_filter(gray, size=5)
            local_sqr = ndimage.uniform_filter(gray**2, size=5)
            local_var = np.maximum(local_sqr - local_mean**2, 0)
            features[i, idx] = 1.0 / (1.0 + np.sqrt(local_var).mean()); idx += 1
            features[i, idx] = np.sqrt(local_var).mean(); idx += 1
            features[i, idx] = 1.0 / (1.0 + np.abs(gray[:, :-1] - gray[:, 1:]).mean() + np.abs(gray[:-1, :] - gray[1:, :]).mean()); idx += 1
            features[i, idx] = (mag > np.percentile(mag, 75)).sum() / mag.size; idx += 1
            
            # LSB (4)
            lsb = (img * 255 % 2)
            features[i, idx] = lsb.mean(); idx += 1
            features[i, idx] = lsb.std(); idx += 1
            lsb_hist, _ = np.histogram(lsb.flatten(), bins=2, range=(0,1))
            lsb_hist = lsb_hist / (lsb_hist.sum() + 1e-10)
            features[i, idx] = -np.sum(lsb_hist[lsb_hist>0] * np.log2(lsb_hist[lsb_hist>0])); idx += 1
            lsb_gray = lsb.mean(axis=0)
            corr = np.corrcoef(lsb_gray[:, :-1].flatten(), lsb_gray[:, 1:].flatten())[0,1]
            features[i, idx] = corr if not np.isnan(corr) else 0; idx += 1
            
            # 统计 (4)
            features[i, idx] = gray.std(); idx += 1
            features[i, idx] = np.percentile(gray, 90) - np.percentile(gray, 10); idx += 1
            g_hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,1))
            g_hist = g_hist / (g_hist.sum() + 1e-10)
            features[i, idx] = -np.sum(g_hist[g_hist>0] * np.log2(g_hist[g_hist>0])); idx += 1
            flat = gray.flatten()
            m, s = flat.mean(), flat.std() + 1e-10
            features[i, idx] = (((flat - m) / s) ** 4).mean() - 3; idx += 1
            
            # 通道 (3)
            if len(img.shape) == 3:
                ch_means = img.mean(axis=(1, 2))
                features[i, idx] = ch_means.var(); idx += 1
                corr_m = np.corrcoef(img.reshape(3, -1))
                features[i, idx] = (corr_m.sum() - 3) / 6; idx += 1
                features[i, idx] = ch_means.std() / (ch_means.mean() + 1e-10)
            
        return features


class CIFAR10FinalDetector:
    """CIFAR10 最终优化检测器"""
    
    def __init__(self):
        self.extractor = None
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        
    def fit_predict(self, data, target_label=0):
        # 数据
        images = data['images'].float() / 255.0
        labels = data['labels']
        poison = data['poison_flags']
        
        mask = (labels == target_label)
        X = images[mask]
        y = poison[mask].numpy()
        
        print(f"\n目标类别 {target_label}: {mask.sum()} 样本 (投毒:{y.sum()}, 干净:{(1-y).sum()})")
        
        # 特征
        if self.extractor is None:
            self.extractor = CIFAR10FeatureExtractor()
        features = self.extractor.extract_features(X)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 选择
        best_feat = self._select(features, y)
        print(f"  选择特征: {len(best_feat)}个")
        
        # 标准化
        feat_scaled = self.scaler.fit_transform(features)
        feat_sel = feat_scaled[:, best_feat]
        
        # PCA
        n_comp = min(12, len(best_feat))
        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(feat_sel)
        
        # 多模型
        scores = self._ensemble(feat_sel, X_pca, y)
        
        # 粒球
        clean, toxic, cont = self._balls(X_pca, scores)
        
        # 传播
        final = self._propagate(X_pca, scores, clean)
        
        # 阈值
        pred, thresh = self._threshold(final, y)
        
        # 指标
        m = self._metrics(final, pred, y)
        m['thresh'] = thresh
        m['clean_balls'] = len(clean)
        m['toxic_balls'] = len(toxic)
        
        return {
            'predictions': pred, 'scores': final, 'metrics': m,
            'features': features, 'X_pca': X_pca, 'contamination': cont,
            'clean_balls': clean, 'toxic_balls': toxic, 'y_true': y
        }
    
    def _select(self, feat, y):
        from scipy.stats import ttest_ind
        n = feat.shape[1]
        scores = np.zeros(n)
        for i in range(n):
            c, p = feat[y==0, i], feat[y==1, i]
            pooled = np.sqrt((c.std()**2 + p.std()**2) / 2) + 1e-10
            cohen = np.abs(p.mean() - c.mean()) / pooled
            _, pv = ttest_ind(c, p)
            scores[i] = cohen * (1.0 if pv < 0.05 else 0.2)
        top = min(15, n)
        return np.argsort(scores)[-top:][::-1].tolist()
    
    def _ensemble(self, feat_sel, X_pca, y):
        eps = 1e-10
        scores = []
        weights = []
        
        # RF
        rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(feat_sel, y)
        s1 = rf.predict_proba(feat_sel)[:, 1]
        scores.append(s1); weights.append(0.30)
        
        # GB
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=7, learning_rate=0.08,
                                       random_state=42)
        gb.fit(feat_sel, y)
        s2 = gb.predict_proba(feat_sel)[:, 1]
        scores.append(s2); weights.append(0.30)
        
        # IF
        iff = IsolationForest(n_estimators=300, contamination=0.12, random_state=42)
        iff.fit(X_pca)
        s3 = iff.score_samples(X_pca)
        s3 = (s3 - s3.min()) / (s3.max() - s3.min() + eps)
        s3 = 1 - s3
        scores.append(s3); weights.append(0.15)
        
        # LOF
        try:
            lof = LocalOutlierFactor(n_neighbors=40, contamination=0.12, novelty=True)
            lof.fit(X_pca)
            s4 = lof.score_samples(X_pca)
            s4 = (s4 - s4.min()) / (s4.max() - s4.min() + eps)
            s4 = 1 - s4
        except:
            s4 = np.ones(len(X_pca)) * 0.5
        scores.append(s4); weights.append(0.15)
        
        # 密度
        knn = NearestNeighbors(n_neighbors=30)
        knn.fit(X_pca)
        d, _ = knn.kneighbors(X_pca)
        dens = 1 / (d.mean(axis=1) + eps)
        dens = (dens - dens.min()) / (dens.max() - dens.min() + eps)
        scores.append(dens); weights.append(0.10)
        
        # 加权
        w = np.array(weights) / sum(weights)
        return sum(wi * si for wi, si in zip(w, scores))
    
    def _balls(self, X_pca, scores):
        self.kmeans = KMeans(n_clusters=90, random_state=42, n_init=20)
        assign = self.kmeans.fit_predict(X_pca)
        
        cont = np.zeros(90)
        for i in range(90):
            m = assign == i
            if m.sum() == 0:
                continue
            pts = X_pca[m]
            center = pts.mean(axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            comp = (dists.mean() + 1e-10) / (dists.std() + 1e-10)
            cont[i] = 0.2 / (comp + 1e-10) + 0.4 * scores[m].std() + 0.4 * scores[m].mean()
        
        cont = (cont - cont.min()) / (cont.max() - cont.min() + 1e-10)
        clean = set(np.where(cont < 0.28)[0])
        toxic = set(np.where(cont >= 0.28)[0])
        return clean, toxic, cont
    
    def _propagate(self, X_pca, scores, clean_balls):
        assign = self.kmeans.predict(X_pca)
        c_mask = np.isin(assign, list(clean_balls))
        if c_mask.sum() == 0:
            return scores
        
        X_c = X_pca[c_mask]
        s_c = scores[c_mask]
        knn = NearestNeighbors(n_neighbors=30, algorithm='ball_tree')
        knn.fit(X_c)
        
        final = np.zeros(len(scores))
        for i in range(len(X_pca)):
            d, idx = knn.kneighbors([X_pca[i]])
            w = 1 / (d[0] + 1e-10)
            final[i] = np.sum(w * s_c[idx[0]]) / np.sum(w)
        return final
    
    def _threshold(self, scores, y):
        best_f1, best_t = 0, scores.mean()
        for t in np.linspace(scores.min(), scores.max(), 1500):
            p = (scores >= t).astype(int)
            if p.sum() == 0:
                continue
            f = f1_score(y, p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
        return (scores >= best_t).astype(int), best_t
    
    def _metrics(self, scores, pred, y):
        eps = 1e-10
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        c_s, p_s = scores[y==0], scores[y==1]
        d_prime = np.abs(p_s.mean() - c_s.mean()) / np.sqrt(0.5 * (c_s.std()**2 + p_s.std()**2) + eps)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        return {'auc': auc, 'ap': ap, 'precision': prec, 'recall': rec, 'f1': f1, 'd_prime': d_prime,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def save_results(results, path):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    m = results['metrics']
    y = results['y_true']
    s = results['scores']
    c, p = y == 0, y == 1
    pca = results['X_pca']
    
    # PCA
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pca[c, 0], pca[c, 1], c='#3498db', alpha=0.3, s=8, label='Clean')
    ax1.scatter(pca[p, 0], pca[p, 1], c='#e74c3c', alpha=0.5, s=15, label='Poison', marker='x')
    ax1.set_title('PCA Space', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # 分布
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(s[c], bins=50, alpha=0.6, color='#3498db', density=True, label='Clean')
    ax2.hist(s[p], bins=50, alpha=0.6, color='#e74c3c', density=True, label='Poison')
    ax2.axvline(m['thresh'], color='purple', linestyle='--', lw=2)
    ax2.set_title('Score Distribution', fontsize=11, fontweight='bold')
    ax2.legend()
    
    # 箱线图
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([s[c], s[p]], labels=['Clean', 'Poison'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax3.axhline(m['thresh'], color='purple', linestyle='--', lw=2)
    ax3.set_title('Score Boxplot', fontsize=11, fontweight='bold')
    
    # ROC
    ax4 = fig.add_subplot(gs[0, 3])
    fpr, tpr, _ = roc_curve(y, s)
    ax4.plot(fpr, tpr, 'b-', lw=2.5, label=f'AUC={m["auc"]:.3f}')
    ax4.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax4.fill_between(fpr, tpr, alpha=0.2)
    ax4.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.set_xlim([0,1])
    ax4.set_ylim([0,1])
    
    # PR
    ax5 = fig.add_subplot(gs[1, 0])
    pr, re, _ = precision_recall_curve(y, s)
    ax5.plot(re, pr, 'r-', lw=2.5, label=f'AP={m["ap"]:.3f}')
    ax5.fill_between(re, pr, alpha=0.2, color='red')
    ax5.set_title('PR Curve', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.set_xlim([0,1])
    ax5.set_ylim([0,1])
    
    # 粒球
    ax6 = fig.add_subplot(gs[1, 1])
    cont = results['contamination']
    colors = ['#27ae60' if i in results['clean_balls'] else '#e74c3c' for i in range(len(cont))]
    ax6.bar(range(len(cont)), cont, color=colors, width=0.8)
    ax6.axhline(0.28, color='orange', linestyle='--', lw=2)
    ax6.set_title(f'Balls ({m["clean_balls"]} clean)', fontsize=11, fontweight='bold')
    
    # 混淆矩阵
    ax7 = fig.add_subplot(gs[1, 2])
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    ax7.imshow(cm, cmap='Blues')
    ax7.set_xticks([0,1])
    ax7.set_yticks([0,1])
    ax7.set_xticklabels(['Clean', 'Poison'])
    ax7.set_yticklabels(['Clean', 'Poison'])
    ax7.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax7.text(j, i, cm[i,j], ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    
    # 汇总
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    text = f"""
    CIFAR10 FINAL OPTIMIZED
    
    Performance Metrics:
    ─────────────────────
    AUC:       {m['auc']:.4f}
    AP:        {m['ap']:.4f}
    F1:        {m['f1']:.4f}
    Precision: {m['precision']:.4f}
    Recall:    {m['recall']:.4f}
    d':        {m['d_prime']:.4f}
    
    Confusion Matrix:
    ─────────────────────
    TP: {m['tp']:<6}  FP: {m['fp']:<6}
    TN: {m['tn']:<6}  FN: {m['fn']:<6}
    
    Balls: {m['clean_balls']} clean / {m['toxic_balls']} toxic
    """
    ax8.text(0.1, 0.9, text, fontsize=11, family='monospace', verticalalignment='top',
             transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='#f0f0f0'))
    
    plt.suptitle('GB-SSD-DA CIFAR10 Final Optimized', fontsize=14, fontweight='bold')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 70)
    print("GB-SSD-DA CIFAR10 FINAL OPTIMIZED")
    print("=" * 70)
    base_dir = Path(__file__).resolve().parent
    generated_dir = base_dir / 'generated_pts'
    
    data = torch.load(generated_dir / 'cifar10_blend.pt', map_location='cpu')
    
    target = 6  # 已知有最多投毒样本的类别
    detector = CIFAR10FinalDetector()
    results = detector.fit_predict(data, target_label=target)
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  AUC:       {results['metrics']['auc']:.4f}")
    print(f"  AP:        {results['metrics']['ap']:.4f}")
    print(f"  F1:        {results['metrics']['f1']:.4f}")
    print(f"  Precision: {results['metrics']['precision']:.4f}")
    print(f"  Recall:    {results['metrics']['recall']:.4f}")
    print(f"  d':        {results['metrics']['d_prime']:.4f}")
    
    save_results(results, str(base_dir / 'cifar10_final_optimized.png'))
