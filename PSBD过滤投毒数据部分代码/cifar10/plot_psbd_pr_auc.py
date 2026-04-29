import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "psbd_resnet50_repro" / "results"


def precision_recall_from_scores(y_true, scores):
    y_true = y_true.bool().cpu()
    scores = scores.float().cpu()
    order = torch.argsort(scores, descending=True)
    y = y_true[order]
    s = scores[order]

    positives = int(y.sum().item())
    total = int(y.numel())
    if positives == 0:
        raise ValueError("AUC-PR requires at least one positive sample.")

    precision = [positives / total]
    recall = [0.0]
    thresholds = [float("inf")]
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0

    i = 0
    n = len(y)
    while i < n:
        threshold = s[i].item()
        while i < n and s[i].item() == threshold:
            if y[i]:
                tp += 1
            else:
                fp += 1
            i += 1
        rec = tp / positives
        prec = tp / max(tp + fp, 1)
        precision.append(prec)
        recall.append(rec)
        thresholds.append(float(threshold))
        ap += (rec - prev_recall) * prec
        prev_recall = rec

    # Trapezoidal area under the PR curve. AP above is the step-wise information retrieval metric.
    auc_pr = 0.0
    for j in range(1, len(recall)):
        auc_pr += (recall[j] - recall[j - 1]) * (precision[j] + precision[j - 1]) / 2.0
    return recall, precision, thresholds, auc_pr, ap


def main():
    train_psu = torch.load(OUT_DIR / "train_psu.pt", map_location="cpu")
    y_true = torch.load(OUT_DIR / "ground_truth_poison_flags.pt", map_location="cpu").bool()

    # PSBD predicts poison when PSU is smaller, so anomaly score is -PSU.
    scores = -train_psu
    recall, precision, thresholds, auc_pr, ap = precision_recall_from_scores(y_true, scores)

    baseline = float(y_true.float().mean().item())
    result = {
        "task": "PSBD poisoned-sample detection",
        "positive_class": "poisoned training sample",
        "score_definition": "score = -PSU; higher score means more suspicious/poison-like",
        "num_samples": int(y_true.numel()),
        "num_positive_poison": int(y_true.sum().item()),
        "num_negative_clean": int((~y_true).sum().item()),
        "positive_rate_baseline": baseline,
        "auc_pr_trapezoidal": auc_pr,
        "average_precision_ap": ap,
        "note": "This AUC-PR/AP is for poisoned-sample detection, not ResNet CIFAR-10 classification.",
    }
    with (OUT_DIR / "psbd_detection_pr_auc.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    rows = [{"recall": r, "precision": p, "threshold": t} for r, p, t in zip(recall, precision, thresholds)]
    with (OUT_DIR / "psbd_detection_pr_curve.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(recall, precision, label=f"PSBD PR (AP = {ap:.4f})", color="#d62728", linewidth=2)
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Positive rate = {baseline:.2f}")
    plt.xlabel("Recall / TPR")
    plt.ylabel("Precision")
    plt.title("PSBD Poisoned-Sample Detection Precision-Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psbd_detection_pr_auc.png", dpi=220)
    plt.close()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
