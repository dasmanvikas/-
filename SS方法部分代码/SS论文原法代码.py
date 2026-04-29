import numpy as np
import torch


PT_PATH = r"C:\Users\DELL\PycharmProjects\PythonProject25\SS方法部分代码\pathmnist_patch.pt"


def load_input(pt_path: str = PT_PATH) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    images = obj["images"].float() / 255.0
    labels = obj["labels"].numpy().astype(np.int32)
    poison_flags = obj["poison_flags"].numpy().astype(np.int32)
    return {
        "images": images,
        "labels": labels,
        "poison_flags": poison_flags,
        "dataset": obj.get("dataset", "unknown"),
        "method": obj.get("method", "unknown"),
    }


def find_suspect_class(labels: np.ndarray) -> tuple[int, np.ndarray]:
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    suspect_class = int(np.argmax(counts))
    return suspect_class, counts


def compute_spectral_scores(images: torch.Tensor, labels: np.ndarray, suspect_class: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(labels == suspect_class)[0]
    X = images[idx].reshape(len(idx), -1)
    X = X - X.mean(dim=0, keepdim=True)
    X = X / (X.std(dim=0, keepdim=True) + 1e-6)

    v = torch.randn(X.shape[1], 1)
    v = v / (torch.norm(v) + 1e-12)
    for _ in range(10):
        v = X.T @ (X @ v)
        v = v / (torch.norm(v) + 1e-12)

    score_suspect = (X @ v).squeeze(1).pow(2).cpu().numpy()
    scores = np.zeros(len(labels), dtype=np.float64)
    scores[idx] = score_suspect
    return idx, scores


def otsu_threshold(score_suspect: np.ndarray) -> float:
    z = np.log1p(score_suspect)
    hist, edges = np.histogram(z, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1))[::-1]

    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    best_idx = int(np.argmax(between))
    thr_log = float(centers[best_idx])
    return float(np.expm1(thr_log))


def run() -> dict:
    payload = load_input()
    suspect_class, counts = find_suspect_class(payload["labels"])
    idx, scores = compute_spectral_scores(payload["images"], payload["labels"], suspect_class)
    threshold = otsu_threshold(scores[idx])
    y_pred = (scores >= threshold).astype(np.int32)
    return {
        "dataset": payload["dataset"],
        "method": payload["method"],
        "suspect_class": suspect_class,
        "class_counts": counts,
        "suspect_indices": idx,
        "scores": scores,
        "threshold": threshold,
        "y_pred": y_pred,
    }
