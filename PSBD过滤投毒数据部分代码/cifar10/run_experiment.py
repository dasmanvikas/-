import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PT = ROOT / "cifar10_patch.pt"
TEST_PT = ROOT / "cifar10_test.pt"
OUT_DIR = ROOT / "psbd_resnet50_repro" / "results"
CKPT_DIR = ROOT / "psbd_resnet50_repro" / "checkpoints"

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2430, 0.2610]).view(3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class PtCifarDataset(Dataset):
    def __init__(self, pt_path, normalize=True):
        obj = torch.load(pt_path, map_location="cpu")
        self.images = obj["images"].contiguous()
        self.labels = obj["labels"].long().contiguous()
        self.poison_flags = obj["poison_flags"].bool().contiguous()
        self.normalize = normalize

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, idx):
        x = self.images[idx].float().div(255.0)
        if self.normalize:
            x = (x - CIFAR10_MEAN) / CIFAR10_STD
        return x, self.labels[idx], self.poison_flags[idx]


class IndexedSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices.long().cpu()

    def __len__(self):
        return int(self.indices.numel())

    def __getitem__(self, idx):
        return self.base[int(self.indices[idx])]


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout_p=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = out + shortcut
        out = self.dropout(out)
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, dropout_p=0.0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_p=dropout_p)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_p=dropout_p)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_p=dropout_p)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_p=dropout_p)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_p):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(block(self.in_planes, planes, s, dropout_p=dropout_p))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


def preact_resnet50_drop(num_classes=10, dropout_p=0.0):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes, dropout_p=dropout_p)


def set_dropout(model, p, enabled):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            m.train(enabled)


def evaluate(model, loader, device, criterion=None, desc="Eval"):
    model.eval()
    set_dropout(model, 0.0, False)
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return {
        "loss": total_loss / max(total, 1) if criterion is not None else None,
        "accuracy": total_correct / max(total, 1),
    }


def load_previous_metrics():
    metrics = []
    for ckpt_path in sorted(CKPT_DIR.glob("epoch_*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue
        row = ckpt.get("metrics")
        if row is not None:
            metrics.append(row)
    seen = set()
    unique = []
    for row in metrics:
        epoch = int(row["epoch"])
        if epoch not in seen:
            unique.append(row)
            seen.add(epoch)
    return sorted(unique, key=lambda r: int(r["epoch"]))


def train_model(args, model, train_loader, clean_test_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 38], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    metrics = []
    start_epoch = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        metrics = load_previous_metrics()
        print(f"Resume from {args.resume_from}, next epoch = {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        set_dropout(model, 0.0, False)
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{args.epochs:03d}] Train")
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch = y.size(0)
            running_loss += loss.item() * batch
            correct += (logits.argmax(1) == y).sum().item()
            total += batch
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{running_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.4g}",
            })

        train_loss = running_loss / total
        train_acc = correct / total
        clean_eval = evaluate(model, clean_test_loader, device, criterion, desc=f"Epoch [{epoch:03d}] Clean Test")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "clean_test_loss": clean_eval["loss"],
            "clean_test_accuracy": clean_eval["accuracy"],
            "lr": lr,
            "epoch_time_sec": elapsed,
        }
        metrics.append(row)
        print(
            f"Epoch {epoch:03d} Summary | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} clean_loss={clean_eval['loss']:.4f} "
            f"clean_acc={clean_eval['accuracy']:.4f} time={elapsed:.1f}s"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": row,
            "args": vars(args),
        }
        torch.save(ckpt, CKPT_DIR / f"epoch_{epoch:03d}.pt")
        save_training_metrics(metrics)
        plot_training(metrics)

    torch.save({"epoch": args.epochs, "model": model.state_dict(), "metrics": metrics}, CKPT_DIR / "final_epoch_050.pt")
    return metrics


def compute_shift_and_psu(model, loader, device, random_seed, forward_passes=3, drop_p=0.8, return_shift_labels=False, desc="PSU"):
    model.eval()
    psu_all = []
    shift_labels = []
    shift_count = 0
    total_count = 0
    with torch.no_grad():
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        for x, _, _ in tqdm(loader, desc=desc):
            x = x.to(device, non_blocking=True)
            set_dropout(model, 0.0, False)
            primitive_prob = F.softmax(model(x), dim=1)
            primitive_label = primitive_prob.argmax(dim=1)

            set_dropout(model, drop_p, True)
            probs = []
            for _pass in range(forward_passes):
                prob = F.softmax(model(x), dim=1)
                pred = prob.argmax(dim=1)
                shifted = pred != primitive_label
                shift_count += shifted.sum().item()
                total_count += pred.numel()
                if return_shift_labels:
                    shift_labels.append(pred[shifted].detach().cpu())
                probs.append(prob)

            mean_prob = torch.stack(probs, dim=0).mean(dim=0)
            diff = primitive_prob - mean_prob
            psu = diff.gather(1, primitive_label.view(-1, 1)).squeeze(1)
            psu_all.append(psu.detach().cpu())
    set_dropout(model, 0.0, False)
    psu_all = torch.cat(psu_all)
    shift_ratio = shift_count / max(total_count, 1)
    if return_shift_labels:
        if shift_labels:
            shift_labels = torch.cat(shift_labels)
        else:
            shift_labels = torch.empty(0, dtype=torch.long)
        return psu_all, shift_ratio, shift_labels
    return psu_all, shift_ratio


def select_dropout_rate(args, model, total_loader, clean_train_loader, poison_train_loader, val_loader, device):
    rows = []
    best_p = None
    best_score = -math.inf
    fallback_p = None
    fallback_score = -math.inf
    for p in args.dropout_candidates:
        _, total_ratio = compute_shift_and_psu(model, total_loader, device, args.seed, args.forward_passes, p, desc=f"Shift total p={p:.1f}")
        _, clean_ratio = compute_shift_and_psu(model, clean_train_loader, device, args.seed, args.forward_passes, p, desc=f"Shift clean p={p:.1f}")
        _, poison_ratio = compute_shift_and_psu(model, poison_train_loader, device, args.seed, args.forward_passes, p, desc=f"Shift poison p={p:.1f}")
        _, val_ratio = compute_shift_and_psu(model, val_loader, device, args.seed, args.forward_passes, p, desc=f"Shift val p={p:.1f}")
        score = val_ratio - total_ratio
        rows.append({
            "dropout_p": p,
            "total_shift_ratio": total_ratio,
            "clean_train_shift_ratio": clean_ratio,
            "poison_train_shift_ratio": poison_ratio,
            "clean_val_shift_ratio": val_ratio,
            "score": score,
        })
        print(
            f"p={p:.1f} total={total_ratio:.4f} clean={clean_ratio:.4f} "
            f"poison={poison_ratio:.4f} val={val_ratio:.4f} score={score:.4f}"
        )
        if score > fallback_score:
            fallback_score = score
            fallback_p = p
        if val_ratio >= 0.8 and score > best_score:
            best_score = score
            best_p = p
    return (best_p if best_p is not None else fallback_p), rows


def detection_metrics(y_true, y_pred):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = int((y_true & y_pred).sum().item())
    fp = int((~y_true & y_pred).sum().item())
    tn = int((~y_true & ~y_pred).sum().item())
    fn = int((y_true & ~y_pred).sum().item())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "main_metrics": {"tpr": tpr, "fpr": fpr},
        "supplementary_metrics": {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc},
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def save_training_metrics(metrics):
    path = OUT_DIR / "train_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT_DIR / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_training(metrics):
    epochs = [m["epoch"] for m in metrics]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [m["train_loss"] for m in metrics], label="train loss")
    plt.plot(epochs, [m["clean_test_loss"] for m in metrics], label="clean test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "train_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [m["train_accuracy"] for m in metrics], label="train accuracy")
    plt.plot(epochs, [m["clean_test_accuracy"] for m in metrics], label="clean test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_shift(rows, selected_p):
    xs = [r["dropout_p"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["total_shift_ratio"] for r in rows], marker="o", label="clean+poison train")
    plt.plot(xs, [r["clean_train_shift_ratio"] for r in rows], marker="o", label="clean train")
    plt.plot(xs, [r["poison_train_shift_ratio"] for r in rows], marker="o", label="poison train")
    plt.plot(xs, [r["clean_val_shift_ratio"] for r in rows], marker="o", label="clean validation")
    plt.axvline(selected_p, linestyle="--", color="black", label=f"selected p={selected_p:.1f}")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Shift Ratio")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shift_ratio_curve.png", dpi=200)
    plt.close()


def plot_psu(clean_train_psu, poison_train_psu, val_psu):
    plt.figure(figsize=(7, 5))
    plt.boxplot([clean_train_psu.numpy(), poison_train_psu.numpy(), val_psu.numpy()], tick_labels=["clean", "poison", "val"])
    plt.ylabel("PSU")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psu_boxplot.png", dpi=200)
    plt.close()


def plot_confusion(metrics):
    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["pred clean", "pred poison"])
    plt.yticks([0, 1], ["true clean", "true poison"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def build_loaders(args):
    train_set = PtCifarDataset(TRAIN_PT, normalize=True)
    test_set = PtCifarDataset(TEST_PT, normalize=True)
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(test_set), generator=gen)
    val_idx = perm[: args.clean_val_size]
    test_idx = perm[args.clean_val_size :]
    torch.save(val_idx, OUT_DIR / "clean_val_indices.pt")
    torch.save(test_idx, OUT_DIR / "clean_test_indices.pt")

    clean_mask = ~train_set.poison_flags
    poison_mask = train_set.poison_flags
    clean_indices = clean_mask.nonzero(as_tuple=False).flatten()
    poison_indices = poison_mask.nonzero(as_tuple=False).flatten()

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    total_eval_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)
    clean_train_loader = DataLoader(IndexedSubset(train_set, clean_indices), shuffle=False, **loader_kwargs)
    poison_train_loader = DataLoader(IndexedSubset(train_set, poison_indices), shuffle=False, **loader_kwargs)
    clean_val_loader = DataLoader(IndexedSubset(test_set, val_idx), shuffle=False, **loader_kwargs)
    clean_test_loader = DataLoader(IndexedSubset(test_set, test_idx), shuffle=False, **loader_kwargs)
    return train_set, train_loader, total_eval_loader, clean_train_loader, poison_train_loader, clean_val_loader, clean_test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clean-val-size", type=int, default=500)
    parser.add_argument("--forward-passes", type=int, default=3)
    parser.add_argument("--dropout-candidates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--threshold-quantile", type=float, default=0.25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_set, train_loader, total_eval_loader, clean_train_loader, poison_train_loader, clean_val_loader, clean_test_loader = build_loaders(args)
    print(f"Train samples: {len(train_set)} | poison samples: {int(train_set.poison_flags.sum())} | clean val: {args.clean_val_size}")

    model = preact_resnet50_drop(num_classes=10, dropout_p=0.0).to(device)
    if args.skip_train:
        ckpt = torch.load(CKPT_DIR / "final_epoch_050.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        metrics = ckpt.get("metrics", [])
    else:
        metrics = train_model(args, model, train_loader, clean_test_loader, device)
        save_training_metrics(metrics)
        plot_training(metrics)

    selected_p, shift_rows = select_dropout_rate(args, model, total_eval_loader, clean_train_loader, poison_train_loader, clean_val_loader, device)
    plot_shift(shift_rows, selected_p)
    with (OUT_DIR / "shift_ratio_log.json").open("w", encoding="utf-8") as f:
        json.dump({"selected_dropout_p": selected_p, "rows": shift_rows}, f, indent=2)

    train_psu, _ = compute_shift_and_psu(model, total_eval_loader, device, args.seed, args.forward_passes, selected_p, desc="Compute train PSU")
    val_psu, _ = compute_shift_and_psu(model, clean_val_loader, device, args.seed, args.forward_passes, selected_p, desc="Compute clean val PSU")
    clean_train_psu, _, clean_shift_labels = compute_shift_and_psu(model, clean_train_loader, device, args.seed, args.forward_passes, selected_p, True, desc="Compute clean train PSU")
    poison_train_psu, _, poison_shift_labels = compute_shift_and_psu(model, poison_train_loader, device, args.seed, args.forward_passes, selected_p, True, desc="Compute poison train PSU")

    threshold = torch.quantile(val_psu, args.threshold_quantile)
    pred_poison = train_psu < threshold
    y_true = train_set.poison_flags.cpu()
    metrics_det = detection_metrics(y_true, pred_poison)
    metrics_det["psbd_config"] = {
        "selected_dropout_p": selected_p,
        "threshold": float(threshold.item()),
        "forward_passes": args.forward_passes,
        "threshold_quantile": args.threshold_quantile,
    }
    print("\nDetection Performance:")
    print(f"TPR: {metrics_det['main_metrics']['tpr']:.3f}")
    print(f"FPR: {metrics_det['main_metrics']['fpr']:.3f}")
    print(json.dumps(metrics_det, indent=2))

    torch.save(train_psu, OUT_DIR / "train_psu.pt")
    torch.save(val_psu, OUT_DIR / "clean_val_psu.pt")
    torch.save(pred_poison, OUT_DIR / "pred_poison_flags.pt")
    torch.save(y_true, OUT_DIR / "ground_truth_poison_flags.pt")
    with (OUT_DIR / "psbd_detection_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_det, f, indent=2)
    plot_psu(clean_train_psu, poison_train_psu, val_psu)
    plot_confusion(metrics_det)


if __name__ == "__main__":
    main()
