"""Model loading, inference, report building, and figure generation.

This module is the core "builder" for the evaluation pipeline, analogous to
``datagen/dataset/report.py`` for the dataset-report stage.

NOTE: The ``build_model`` function must stay in sync with
``nina_cloud/training/train_cloud.py`` (architecture definition).  If the
classifier head or backbone configuration changes during training, update
``build_model`` to match.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from nina_cloud.evaluation.metrics import (
    compute_classification_metrics,
    group_metrics_by_profile,
)
from nina_cloud.training.train_cloud import CloudDataset

logger = logging.getLogger(__name__)

# Inference threshold — must match training/train_cloud.py line 81
THRESHOLD = 0.5


def build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Rebuild the MobileNetV2 classifier and load trained weights.

    Mirrors the architecture defined in ``training/train_cloud.py:160-167``.
    The backbone (``features``) is frozen and the classifier head is a
    single-logit binary head.

    Args:
        checkpoint_path: Path to ``mobilenetv2_cloud_best.pt`` (bare
            ``state_dict`` saved by ``torch.save(model.state_dict(), ...)``.
        device: Target device.

    Returns:
        Model in ``eval()`` mode with weights loaded from the checkpoint.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 1),
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    return model


def eval_transform() -> transforms.Compose:
    """Return the deterministic evaluation transform (no augmentation).

    Identical to the val transform in ``training/train_cloud.py:134-140``.
    Uses ImageNet normalisation statistics.

    Returns:
        Composed torchvision transform suitable for evaluation.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def run_inference(
    model: nn.Module,
    records: list,
    image_dir: Path,
    device: torch.device,
    batch_size: int = 16,
) -> list:
    """Run batch inference and return per-record predictions.

    The dataset is iterated in index order (``shuffle=False``) so that the
    returned list aligns one-to-one with *records*.

    Args:
        model: Trained model in ``eval()`` mode.
        records: List of manifest record dicts (must contain ``image`` and
            ``label`` keys as in ``balanced_manifest.json``).
        image_dir: Directory containing the PNG images.
        device: Device to run inference on.
        batch_size: DataLoader batch size.

    Returns:
        List of dicts — each input record with two extra keys added:
        ``score`` (sigmoid probability for class cloud=1) and ``pred``
        (hard prediction: 0 or 1 at threshold 0.5).
    """
    tf = eval_transform()
    dataset = CloudDataset(records, image_dir, tf)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    all_scores: list = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)  # (B,)
            scores = logits.sigmoid().cpu().tolist()
            all_scores.extend(scores)

    augmented = []
    for rec, score in zip(records, all_scores):
        pred = 1 if score >= THRESHOLD else 0
        augmented.append({**rec, "score": round(score, 6), "pred": pred})

    return augmented


def build_evaluation_report(
    records: list,
    image_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 16,
    split: str = "test",
) -> dict:
    """Run end-to-end evaluation and write ``evaluation_summary.json``.

    Filters *records* to the requested *split*, runs inference with the
    checkpoint, computes overall and per-profile metrics, and writes two
    files:

    * ``output_path`` — ``evaluation_summary.json`` with the full report.
    * ``<output_path.parent>/predictions.json`` — per-image predictions for
      downstream inspection.

    Args:
        records: All manifest records (will be filtered to *split*).
        image_dir: Directory containing the PNG images.
        checkpoint_path: Path to the trained checkpoint file.
        output_path: Destination path for ``evaluation_summary.json``.
        device: Inference device.
        batch_size: DataLoader batch size.
        split: Which manifest split to evaluate (``"test"`` by default;
            pass ``"val"`` when the dataset has too few scenes to produce a
            test partition).

    Returns:
        The report dict that was written to *output_path*.
    """
    test_records = [r for r in records if r.get("split") == split]
    logger.info(
        "Evaluating on %d '%s' records (of %d total)",
        len(test_records),
        split,
        len(records),
    )
    if not test_records:
        raise ValueError(f"No records with split=='{split}' found in manifest.")

    model = build_model(checkpoint_path, device)
    predictions = run_inference(model, test_records, image_dir, device, batch_size)

    y_true = [r["label"] for r in predictions]
    y_pred = [r["pred"] for r in predictions]
    y_score = [r["score"] for r in predictions]
    overall = compute_classification_metrics(y_true, y_pred, y_score)
    by_profile = group_metrics_by_profile(predictions)

    report = {
        "stage": "run_evaluate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path.name,
        "threshold": THRESHOLD,
        "split": split,
        "device": str(device),
        "overall": overall,
        "by_profile": by_profile,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written: %s", output_path)

    # Per-record predictions for downstream inspection
    pred_path = output_path.parent / "predictions.json"
    pred_records = [
        {
            "image": r["image"],
            "label": r["label"],
            "profile": r.get("profile"),
            "scene_id": r.get("scene_id"),
            "cloud_frac": r.get("cloud_frac"),
            "score": r["score"],
            "pred": r["pred"],
        }
        for r in predictions
    ]
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(pred_records), "records": pred_records}, f, indent=2)
    logger.info("Per-record predictions written: %s", pred_path)

    return report


def save_figures(report: dict, predictions: list, output_dir: Path) -> None:
    """Save confusion-matrix heatmaps and ROC curves as PNG files.

    Renders an overall figure plus one figure per degradation profile
    (A/B/C) that appears in *predictions*.  All figures are written to
    ``<output_dir>/figures/``.

    Uses the ``Agg`` matplotlib backend so this works in headless environments.

    Args:
        report: Report dict as returned by :func:`build_evaluation_report`.
        predictions: Per-record prediction dicts (with ``label``, ``score``,
            ``pred``, ``profile`` keys).
        output_dir: Dataset directory; figures go into ``output_dir/figures/``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _plot_group(
        group_name: str,
        metrics: dict,
        recs: list,
    ) -> None:
        """Render and save a 2-panel figure: confusion matrix + ROC curve."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Evaluation — {group_name}", fontsize=13)

        # --- Confusion matrix (rendered as a table to avoid a matplotlib
        # 3.10 + Python 3.14 infinite deepcopy bug in tick creation) ---
        cm_dict = metrics["confusion_matrix"]
        ax = axes[0]
        ax.axis("off")
        ax.set_title("Confusion Matrix", pad=14)
        cell_text = [
            [str(cm_dict["tn"]), str(cm_dict["fp"])],
            [str(cm_dict["fn"]), str(cm_dict["tp"])],
        ]
        col_labels = ["Pred Clear", "Pred Cloud"]
        row_labels = ["True Clear", "True Cloud"]
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.4, 2.0)
        # Shade the diagonal (TN=top-left, TP=bottom-right) in light blue
        for (row, col), cell in tbl.get_celld().items():
            if row > 0 and (row - 1) == col:  # diagonal cells (offset by header row)
                cell.set_facecolor("#c6dbef")

        # --- ROC curve ---
        ax2 = axes[1]
        y_true_g = np.array([r["label"] for r in recs])
        y_score_g = np.array([r["score"] for r in recs])
        unique = np.unique(y_true_g)
        if len(unique) >= 2:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true_g, y_score_g)
            auc_val = metrics.get("roc_auc") or 0.0
            ax2.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", lw=1)
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend(loc="lower right")
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1.02])
        else:
            ax2.text(
                0.5,
                0.5,
                "Single class — ROC N/A",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("ROC Curve")

        # tight_layout triggers a matplotlib 3.10 + Python 3.14 deepcopy bug
        # (infinite recursion in matplotlib/path.py).  Use subplots_adjust instead.
        fig.subplots_adjust(top=0.88, wspace=0.4, left=0.15, right=0.95)
        safe_name = group_name.lower().replace(" ", "_")
        out_path = figures_dir / f"eval_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Figure saved: %s", out_path)

    # Overall
    _plot_group("overall", report["overall"], predictions)

    # Per profile
    by_profile: dict = {}
    for rec in predictions:
        p = rec.get("profile", "unknown")
        by_profile.setdefault(p, []).append(rec)

    for profile, recs in sorted(by_profile.items()):
        profile_metrics = report["by_profile"].get(profile)
        if profile_metrics:
            _plot_group(f"profile_{profile}", profile_metrics, recs)
