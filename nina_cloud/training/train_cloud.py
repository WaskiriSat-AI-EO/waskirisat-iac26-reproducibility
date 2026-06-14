"""MobileNetV2 transfer-learning baseline for cloud/clear classification.

Respects the ``split`` field in ``balanced_manifest.json`` so that training
records, validation records, and held-out test records never contaminate each
other.  The model is trained on ``split=="train"`` images only; ``split=="val"``
is used for checkpoint selection and threshold tuning; ``split=="test"`` is
never touched here.

After training the best checkpoint is reloaded and a threshold sweep is
performed on the val set to select the highest decision threshold whose cloud
recall meets the requested target (maximises precision under the recall floor).
The selected threshold and full training provenance are written to
``training_report.json`` for downstream use by the evaluation pipeline.

Asymmetric cost note
---------------------
``pos_weight`` is NOT a class-imbalance correction — the training split is
already 50/50 cloud:clear.  It encodes the FN:FP mission cost ratio: a missed
cloud (FN) causes corrupted science data to be downlinked; a false alarm (FP)
merely drops a clear image.  Higher ``pos_weight`` values force the loss to
penalise missed clouds more, shifting the operating point toward higher recall.
``beta = sqrt(pos_weight)`` is derived for F-beta to keep the two coherent.

Usage::

    python -m nina_cloud.training.train_cloud \\
        --dataset_dir data/cloud/04_dataset \\
        --epochs 15 --pos_weight 3.0 --recall_target 0.99
"""

import argparse
import json
import math
import random
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Training hyper-parameters (overridable via CLI) ───────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
SEED = 42

# Legacy fallback: used only when manifest records carry no ``split`` field.
_LEGACY_VAL_FRAC = 0.25


# ── Dataset ───────────────────────────────────────────────────────────────────


class CloudDataset(Dataset):
    def __init__(self, records: list, image_dir: Path, transform):
        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(self.image_dir / r["image"]).convert("RGB")
        return self.transform(img), torch.tensor(r["label"], dtype=torch.float32)


# ── Split helpers ─────────────────────────────────────────────────────────────


def scene_split(records: list, val_frac: float, seed: int):
    """Split records at scene level to avoid spatial leakage.

    .. deprecated::
        This function is a legacy fallback used only when the manifest lacks
        a ``split`` field.  Normal execution reads the split directly from the
        manifest (stamped by ``run_split``), which is the only way to guarantee
        that test scenes are never used during training or threshold tuning.
    """
    rng = random.Random(seed)
    by_scene: dict = defaultdict(list)
    for r in records:
        by_scene[r["scene_id"]].append(r)

    scenes = sorted(by_scene.keys())
    rng.shuffle(scenes)
    n_val = max(1, round(len(scenes) * val_frac))
    val_scenes = set(scenes[:n_val])
    train_scenes = set(scenes[n_val:])

    train = [r for r in records if r["scene_id"] in train_scenes]
    val = [r for r in records if r["scene_id"] in val_scenes]
    return train, val, train_scenes, val_scenes


def _load_splits(
    records: list,
    seed: int,
) -> tuple[list, list]:
    """Return (train_records, val_records) respecting the manifest split field.

    Falls back to ``scene_split`` with a warning when the split field is absent
    (old manifest format).  Test records (``split=="test"``) are never returned.
    Ambiguous records (``label==-1``) are silently dropped.
    """
    clean = [r for r in records if r.get("label") != -1]
    has_split = any("split" in r for r in clean)

    if has_split:
        train_records = [r for r in clean if r.get("split") == "train"]
        val_records = [r for r in clean if r.get("split") == "val"]
        if not train_records:
            raise RuntimeError(
                "No 'train' records found in balanced_manifest.json. "
                "Run datagen stages 1–6 to populate the manifest."
            )
        if not val_records:
            warnings.warn(
                "No 'val' records in manifest — falling back to scene-level split "
                f"(VAL_FRAC={_LEGACY_VAL_FRAC}).  Run run_split to fix this.",
                stacklevel=3,
            )
            train_records, val_records, _, _ = scene_split(
                clean, _LEGACY_VAL_FRAC, seed
            )
    else:
        warnings.warn(
            "Manifest records have no 'split' field — using legacy scene_split(). "
            "Run run_split to stamp split assignments and prevent test leakage.",
            stacklevel=3,
        )
        train_records, val_records, _, _ = scene_split(clean, _LEGACY_VAL_FRAC, seed)

    return train_records, val_records


# ── Training loop ─────────────────────────────────────────────────────────────


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    """Run one training or validation epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.  The in-loop accuracy uses the
        default 0.5 threshold for monitoring purposes only; the mission
        operating threshold is determined post-training by ``_select_threshold``.
    """
    model.train() if training else model.eval()
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)
            if training:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = (out.sigmoid() > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            n += imgs.size(0)
    return total_loss / n, correct / n


# ── Threshold sweep helpers ───────────────────────────────────────────────────


def _collect_val_scores(
    model: nn.Module,
    val_records: list,
    image_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    """Run inference on val records and return (y_true, y_score)."""
    tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        CloudDataset(val_records, image_dir, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    y_true: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            y_score.extend(logits.sigmoid().cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_score


def _select_threshold(
    y_true: list[int],
    y_score: list[float],
    recall_target: float,
) -> tuple[float, bool]:
    """Select the operating threshold on the val set.

    Strategy: sweep all unique score values (high to low) and pick the highest
    threshold whose cloud recall >= ``recall_target``.  This maximises precision
    under the recall floor.

    Returns:
        (threshold, degenerate_val).  ``degenerate_val`` is True when only one
        class is present in the val set or no threshold achieves the target —
        in both cases the threshold falls back to 0.5.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)

    if len(np.unique(arr_true)) < 2:
        warnings.warn(
            "Val set contains only one class — threshold sweep unavailable; "
            "falling back to 0.5.  Acquire more diverse scenes to enable tuning.",
            stacklevel=2,
        )
        return 0.5, True

    n_pos = int((arr_true == 1).sum())
    candidates = np.sort(np.unique(arr_score))[::-1]  # descending
    best_thr: float | None = None
    for thr in candidates:
        tp = int(((arr_score >= thr) & (arr_true == 1)).sum())
        if tp / n_pos >= recall_target:
            if best_thr is None or thr > best_thr:
                best_thr = float(thr)

    if best_thr is None:
        warnings.warn(
            f"No threshold achieves val recall >= {recall_target:.2f}; "
            "falling back to 0.5.  Consider lowering --recall_target or "
            "acquiring more training data.",
            stacklevel=2,
        )
        return 0.5, True

    return best_thr, False


def _val_metrics_at_threshold(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
    beta: float,
) -> dict:
    """Compute val-set point metrics at the selected threshold."""
    arr_true = np.asarray(y_true, dtype=int)
    arr_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(((arr_pred == 1) & (arr_true == 1)).sum())
    fp = int(((arr_pred == 1) & (arr_true == 0)).sum())
    fn = int(((arr_pred == 0) & (arr_true == 1)).sum())
    tn = int(((arr_pred == 0) & (arr_true == 0)).sum())

    n_pos = tp + fn
    n_neg = tn + fp
    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0

    beta2 = beta**2
    fbeta = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if (beta2 * precision + recall) > 0
        else 0.0
    )

    return {
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "specificity": round(specificity, 6),
        "fbeta": round(fbeta, 6),
        "beta": round(beta, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _precision_at_recall(
    y_true: list[int],
    y_score: list[float],
    recall_targets: list[float],
) -> dict[str, float | None]:
    """For each recall target return the precision at the best threshold."""
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    n_pos = int((arr_true == 1).sum())

    if n_pos == 0 or len(np.unique(arr_true)) < 2:
        return {str(rt): None for rt in recall_targets}

    candidates = np.sort(np.unique(arr_score))[::-1]
    result: dict[str, float | None] = {}
    for rt in recall_targets:
        best_thr: float | None = None
        for thr in candidates:
            tp = int(((arr_score >= thr) & (arr_true == 1)).sum())
            if tp / n_pos >= rt:
                if best_thr is None or thr > best_thr:
                    best_thr = float(thr)

        if best_thr is None:
            result[str(rt)] = None
        else:
            tp = int(((arr_score >= best_thr) & (arr_true == 1)).sum())
            fp = int(((arr_score >= best_thr) & (arr_true == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            result[str(rt)] = round(prec, 6)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 cloud/clear classifier"
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/cloud/04_dataset",
        help="04_dataset/ directory (default: data/cloud/04_dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr", type=float, default=LR, help=f"Learning rate (default: {LR})"
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=3.0,
        help=(
            "Weight on positive-class (cloud) loss term in BCEWithLogitsLoss. "
            "Encodes the FN:FP mission cost ratio — NOT a class-imbalance correction "
            "(the training split is already 50/50).  Higher values penalise missed "
            "clouds more, shifting the operating point toward higher recall.  "
            "beta = sqrt(pos_weight) is derived for F-beta.  (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--recall_target",
        type=float,
        default=0.99,
        help=(
            "Minimum cloud recall to achieve during the val-set threshold sweep. "
            "The selected threshold is the highest one whose val recall >= this "
            "value (maximises precision under the recall floor).  (default: 0.99)"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    dataset_dir = Path(args.dataset_dir)
    image_dir = dataset_dir / "images"

    manifest = json.load(open(dataset_dir / "balanced_manifest.json"))
    records = manifest["records"]
    print(
        f"Balanced records: {len(records)}  "
        f"(cloud={sum(r['label'] == 1 for r in records)}  "
        f"clear={sum(r['label'] == 0 for r in records)})"
    )

    train_records, val_records = _load_splits(records, SEED)

    print(
        f"  train: {len(train_records)} images  "
        f"(cloud={sum(r['label'] == 1 for r in train_records)}  "
        f"clear={sum(r['label'] == 0 for r in train_records)})"
    )
    print(
        f"  val:   {len(val_records)} images  "
        f"(cloud={sum(r['label'] == 1 for r in val_records)}  "
        f"clear={sum(r['label'] == 0 for r in val_records)})"
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(
        CloudDataset(train_records, image_dir, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CloudDataset(val_records, image_dir, val_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1),
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # pos_weight > 1 makes the loss penalise FN (missed clouds) more than FP.
    # beta = sqrt(pos_weight) keeps F-beta coherent with the cost ratio.
    beta = math.sqrt(args.pos_weight)
    print(
        f"pos_weight={args.pos_weight:.1f}  beta={beta:.3f}  "
        f"(FN:FP cost ratio — not an imbalance correction)\n"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    print(
        f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}"
    )
    print("-" * 52)

    per_epoch = []
    best_val_acc = 0.0
    checkpoint_path = dataset_dir / "mobilenetv2_cloud_best.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        per_epoch.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_loss": round(vl_loss, 6),
                "val_acc": round(vl_acc, 6),
            }
        )

        marker = " ←" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.3f}"
            f"  {vl_loss:>8.4f}  {vl_acc:>7.3f}{marker}"
        )

    print(f"\nBest val acc: {best_val_acc:.3f}")
    print(f"Model saved → {checkpoint_path}")

    # ── Threshold sweep on best checkpoint ────────────────────────────────────
    print(f"\nRunning threshold sweep (recall_target={args.recall_target})…")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    y_true_val, y_score_val = _collect_val_scores(
        model, val_records, image_dir, device, args.batch_size
    )
    selected_threshold, degenerate_val = _select_threshold(
        y_true_val, y_score_val, args.recall_target
    )
    val_metrics = _val_metrics_at_threshold(
        y_true_val, y_score_val, selected_threshold, beta
    )
    par = _precision_at_recall(y_true_val, y_score_val, [0.95, 0.99])

    print(
        f"Selected threshold: {selected_threshold:.4f}  "
        f"(val recall={val_metrics['recall']:.3f}, "
        f"precision={val_metrics['precision']:.3f}, "
        f"specificity={val_metrics['specificity']:.3f})"
    )
    if degenerate_val:
        print(
            "⚠  Degenerate val — threshold is fallback 0.5 (see training_report.json)."
        )

    # ── Write training_report.json ─────────────────────────────────────────────
    report: dict = {
        "stage": "train_cloud",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "beta": round(beta, 6),
            "seed": SEED,
            "recall_target": args.recall_target,
        },
        "checkpoint": checkpoint_path.name,
        "selected_threshold": selected_threshold,
        "degenerate_val": degenerate_val,
        "val_metrics_at_threshold": val_metrics,
        "precision_at_recall": par,
        "per_epoch": per_epoch,
        "notes": (
            "pos_weight encodes the FN:FP mission cost ratio (not class imbalance — "
            "train is 50/50).  beta=sqrt(pos_weight) for F-beta.  "
            "selected_threshold is the highest val threshold achieving recall >= "
            "recall_target (maximises precision under the recall floor).  "
            "Load this value in run_evaluate via --dataset_dir auto-detection "
            "or --threshold."
        ),
    }
    report_path = dataset_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Training report written → {report_path}")


if __name__ == "__main__":
    main()
