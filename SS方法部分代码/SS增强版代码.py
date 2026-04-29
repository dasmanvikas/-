import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 42
PT_PATH = r"C:\Users\DELL\PycharmProjects\PythonProject25\SS方法部分代码\cifar10_patch.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
NUM_CLASSES = 10
FEAT_DIM = 128
TOP_K = 6
RECON_WEIGHT = 0.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SimpleDataset(Dataset):
    def __init__(self, x, y, poison=None):
        self.x = x
        self.y = y
        self.poison = poison

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.poison is None:
            return self.x[idx], self.y[idx]
        return self.x[idx], self.y[idx], self.poison[idx], idx


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, feat_dim)
        self.fc2 = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feat=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        if return_feat:
            return logits, feat
        return logits


def load_input(pt_path: str = PT_PATH) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    images = obj["images"].float()
    if images.max() > 1:
        images = images / 255.0
    labels = obj["labels"].long()
    poison_flags = obj["poison_flags"].long()
    return {
        "images": images,
        "labels": labels,
        "poison_flags": poison_flags,
        "dataset": obj.get("dataset", "unknown"),
        "method": obj.get("method", "unknown"),
    }


def train_feature_model(images: torch.Tensor, labels: torch.Tensor, poison_flags: torch.Tensor):
    dataset = SimpleDataset(images, labels, poison_flags)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = SmallCNN(num_classes=NUM_CLASSES, feat_dim=FEAT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for x, y, _, _ in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return model, full_loader


def extract_representations(model: nn.Module, full_loader: DataLoader):
    model.eval()
    all_feats = []
    all_labels = []
    all_poison = []
    all_indices = []
    with torch.no_grad():
        for x, y, p, idx in full_loader:
            x = x.to(DEVICE)
            _, feat = model(x, return_feat=True)
            all_feats.append(feat.cpu())
            all_labels.append(y.cpu())
            all_poison.append(p.cpu())
            all_indices.append(idx.cpu())
    return (
        torch.cat(all_feats, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_poison, dim=0),
        torch.cat(all_indices, dim=0),
    )


def compute_spectral_scores(all_feats: torch.Tensor, all_labels: torch.Tensor):
    scores = torch.zeros(all_feats.shape[0], dtype=torch.float32)
    class_strength = []
    for c in range(NUM_CLASSES):
        idx_c = torch.where(all_labels == c)[0]
        F_c = all_feats[idx_c]
        mean_c = F_c.mean(dim=0, keepdim=True)
        C_c = F_c - mean_c
        q = min(TOP_K, C_c.shape[0] - 1, C_c.shape[1])
        if q < 1:
            scores[idx_c] = 0.0
            class_strength.append(0.0)
            continue

        _, svals, v = torch.pca_lowrank(C_c, q=q, center=False)
        proj = C_c @ v[:, :q]
        proj_energy = (proj ** 2).sum(dim=1)
        recon = proj @ v[:, :q].T
        residual = ((C_c - recon) ** 2).mean(dim=1)
        scores[idx_c] = (proj_energy + RECON_WEIGHT * residual).cpu()

        lam1 = float((svals[0] ** 2).item()) if svals.numel() >= 1 else 0.0
        lam2 = float((svals[1] ** 2).item()) if svals.numel() >= 2 else 0.0
        class_strength.append(lam1 / (lam2 + 1e-12))
    return scores, np.array(class_strength)


def mad_normalize_per_class(scores: torch.Tensor, all_labels: torch.Tensor) -> np.ndarray:
    global_score = torch.zeros_like(scores)
    for c in range(NUM_CLASSES):
        idx_c = torch.where(all_labels == c)[0]
        s = scores[idx_c].numpy()
        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-12
        z = (s - med) / (1.4826 * mad + 1e-12)
        z = np.maximum(z, 0.0)
        global_score[idx_c] = torch.from_numpy(z).float()
    return global_score.numpy()


def run() -> dict:
    payload = load_input()
    model, full_loader = train_feature_model(payload["images"], payload["labels"], payload["poison_flags"])
    all_feats, all_labels, all_poison, all_indices = extract_representations(model, full_loader)
    raw_scores, class_strength = compute_spectral_scores(all_feats, all_labels)
    normalized_scores = mad_normalize_per_class(raw_scores, all_labels)
    poison_classes = sorted(np.unique(all_labels.numpy()[all_poison.numpy() == 1]).tolist())
    target_mask = np.isin(all_labels.numpy(), poison_classes)
    return {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "model": model,
        "features": all_feats,
        "labels": all_labels,
        "poison_flags": all_poison,
        "indices": all_indices,
        "raw_scores": raw_scores.numpy(),
        "normalized_scores": normalized_scores,
        "class_strength": class_strength,
        "poison_classes": poison_classes,
        "target_mask": target_mask,
    }
