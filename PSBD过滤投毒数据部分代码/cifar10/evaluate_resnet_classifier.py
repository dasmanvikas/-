import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from run_experiment import (
    CKPT_DIR,
    OUT_DIR,
    TEST_PT,
    IndexedSubset,
    PtCifarDataset,
    preact_resnet50_drop,
    set_dropout,
)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = PtCifarDataset(TEST_PT, normalize=True)
    test_idx_path = OUT_DIR / "clean_test_indices.pt"
    if test_idx_path.exists():
        indices = torch.load(test_idx_path, map_location="cpu")
    else:
        indices = torch.arange(len(test_set))
    loader = DataLoader(IndexedSubset(test_set, indices), batch_size=256, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    ckpt = torch.load(CKPT_DIR / "final_epoch_050.pt", map_location=device)
    model = preact_resnet50_drop(num_classes=10, dropout_p=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    set_dropout(model, 0.0, False)

    cm = torch.zeros(10, 10, dtype=torch.long)
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Evaluate ResNet classifier"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            total_loss += criterion(logits, y).item()
            correct += (pred == y).sum().item()
            total += y.numel()
            for true, predicted in zip(y.cpu(), pred.cpu()):
                cm[int(true), int(predicted)] += 1

    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i].item()
        support = cm[i, :].sum().item()
        pred_count = cm[:, i].sum().item()
        recall = tp / support if support else 0.0
        precision = tp / pred_count if pred_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "class_id": i,
            "class_name": name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_precision = sum(x["precision"] for x in per_class) / 10
    macro_recall = sum(x["recall"] for x in per_class) / 10
    macro_f1 = sum(x["f1"] for x in per_class) / 10
    metrics = {
        "split": "clean_test_remaining_9500" if len(indices) != len(test_set) else "clean_test_all_10000",
        "num_samples": int(total),
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    with (OUT_DIR / "resnet_classification_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (OUT_DIR / "resnet_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([CLASS_NAMES[i]] + row)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"ResNet-50 CIFAR-10 Confusion Matrix\nAccuracy={metrics['accuracy']:.4f}")
    for i in range(10):
        for j in range(10):
            val = int(cm[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "resnet_confusion_matrix.png", dpi=220)
    plt.close(fig)

    print(json.dumps({
        "num_samples": metrics["num_samples"],
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
