"""
Generate a full, reproducible metrics report for the trained models.

Evaluates every .pth in models/ on data/Testing and writes:
  reports/metrics.json      machine-readable numbers
  reports/metrics.md        human-readable summary (resume / README ready)
  reports/confusion_matrix_<model>.png
"""
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TEST_DIR = "./data/Testing"
TRAIN_DIR = "./data/Training"
BATCH_SIZE = 32
OUT_DIR = "reports"


def build_loader():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(TEST_DIR, transform=tf)
    return ds, DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def evaluate_model(path, ds, loader, device):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(ds.classes))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels, total_loss = [], [], 0.0

    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    elapsed = time.time() - t0

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = probs.argmax(axis=1)
    n = len(y_true)

    report = classification_report(
        y_true, y_pred, target_names=ds.classes, output_dict=True, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # Binary clinical view: any tumor (glioma/meningioma/pituitary) vs no tumor
    notumor_idx = ds.classes.index("notumor")
    y_true_bin = (y_true != notumor_idx).astype(int)   # 1 = tumor present
    y_pred_bin = (y_pred != notumor_idx).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    correct_mask = y_pred == y_true
    conf = probs.max(axis=1)

    params = sum(p.numel() for p in model.parameters())

    result = {
        "weights_file": path,
        "weights_size_mb": round(os.path.getsize(path) / 1e6, 2),
        "total_parameters": params,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_images": n,
        "classes": ds.classes,
        "test_accuracy_pct": round(100 * accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy_pct": round(100 * balanced_accuracy_score(y_true, y_pred), 4),
        "top2_accuracy_pct": round(100 * top_k_accuracy_score(y_true, probs, k=2, labels=range(len(ds.classes))), 4),
        "test_loss": round(total_loss / n, 4),
        "misclassified_count": int((~correct_mask).sum()),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "roc_auc_ovr_macro": round(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"), 6),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "matthews_corrcoef": round(matthews_corrcoef(y_true, y_pred), 4),
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in ds.classes
        },
        "confusion_matrix": cm.tolist(),
        "binary_tumor_vs_notumor": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity_recall_pct": round(100 * tp / (tp + fn), 4) if tp + fn else None,
            "specificity_pct": round(100 * tn / (tn + fp), 4) if tn + fp else None,
            "precision_pct": round(100 * tp / (tp + fp), 4) if tp + fp else None,
            "missed_tumors_false_negatives": fn,
        },
        "mean_confidence_pct": round(100 * float(conf.mean()), 2),
        "mean_confidence_when_correct_pct": round(100 * float(conf[correct_mask].mean()), 2),
        "mean_confidence_when_wrong_pct": round(100 * float(conf[~correct_mask].mean()), 2) if (~correct_mask).any() else None,
        "total_inference_seconds": round(elapsed, 2),
        "throughput_images_per_sec": round(n / elapsed, 1),
        "latency_ms_per_image": round(1000 * elapsed / n, 2),
        "device": str(device),
    }

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds.classes)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(f"{os.path.basename(path)}\nTest Accuracy: {result['test_accuracy_pct']:.2f}%")
    plt.tight_layout()
    tag = os.path.basename(path).replace(".pth", "")
    fig.savefig(f"{OUT_DIR}/confusion_matrix_{tag}.png", dpi=150)
    plt.close(fig)

    return result


def dataset_stats():
    stats = {}
    for split, d in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        if not os.path.isdir(d):
            continue
        counts = {
            c: len([f for f in os.listdir(os.path.join(d, c)) if not f.startswith(".")])
            for c in sorted(os.listdir(d))
            if os.path.isdir(os.path.join(d, c))
        }
        stats[split] = {"per_class": counts, "total": sum(counts.values())}
    if "train" in stats:
        t = stats["train"]["total"]
        stats["train"]["split_80_20"] = {"train_subset": int(0.8 * t), "val_subset": t - int(0.8 * t)}
    stats["grand_total"] = sum(v["total"] for k, v in stats.items() if isinstance(v, dict) and "total" in v)
    return stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds, loader = build_loader()

    weights = sorted(f for f in os.listdir("models") if f.endswith(".pth"))
    results = {}
    for w in weights:
        path = os.path.join("models", w)
        print(f"Evaluating {path} ...")
        results[w] = evaluate_model(path, ds, loader, device)
        print(f"  -> {results[w]['test_accuracy_pct']:.2f}% accuracy")

    payload = {"dataset": dataset_stats(), "models": results}
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
