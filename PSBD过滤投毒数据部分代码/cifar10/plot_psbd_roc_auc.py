import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "psbd_resnet50_repro" / "results"


def roc_curve_from_scores(y_true, scores):
    y_true = y_true.bool().cpu()
    scores = scores.float().cpu()
    order = torch.argsort(scores, descending=True)
    y = y_true[order]
    s = scores[order]

    positives = int(y.sum().item())
    negatives = int((~y).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples.")

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [float("inf")]
    tp = 0
    fp = 0
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
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        thresholds.append(float(threshold))

    auc = 0.0
    for j in range(1, len(fpr)):
        auc += (fpr[j] - fpr[j - 1]) * (tpr[j] + tpr[j - 1]) / 2.0
    return fpr, tpr, thresholds, auc


def main():
    train_psu = torch.load(OUT_DIR / "train_psu.pt", map_location="cpu")
    y_true = torch.load(OUT_DIR / "ground_truth_poison_flags.pt", map_location="cpu").bool()

    # PSBD predicts poison when PSU is smaller, so anomaly score is -PSU.
    scores = -train_psu
    fpr, tpr, thresholds, auc = roc_curve_from_scores(y_true, scores)

    result = {
        "task": "PSBD poisoned-sample detection",
        "positive_class": "poisoned training sample",
        "score_definition": "score = -PSU; higher score means more suspicious/poison-like",
        "num_samples": int(y_true.numel()),
        "num_positive_poison": int(y_true.sum().item()),
        "num_negative_clean": int((~y_true).sum().item()),
        "auc_roc": auc,
        "note": "This ROC-AUC is for backdoor sample detection, not ResNet CIFAR-10 classification.",
    }
    with (OUT_DIR / "psbd_detection_roc_auc.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    roc_rows = [{"fpr": x, "tpr": y, "threshold": z} for x, y, z in zip(fpr, tpr, thresholds)]
    with (OUT_DIR / "psbd_detection_roc_curve.json").open("w", encoding="utf-8") as f:
        json.dump(roc_rows, f)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, label=f"PSBD detection ROC (AUC = {auc:.4f})", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PSBD Poisoned-Sample Detection ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psbd_detection_roc_auc.png", dpi=220)
    plt.close()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
