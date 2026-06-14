"""Model loading, inference, report building, and figure generation.

This module is the core "builder" for the evaluation pipeline, analogous to
``datagen/dataset/report.py`` for the dataset-report stage.

NOTE: The ``build_model`` function must stay in sync with
``nina_cloud/training/train_cloud.py`` (architecture definition).  If the
classifier head or backbone configuration changes during training, update
``build_model`` to match.

Evaluation design — asymmetric cost and profile policy
-------------------------------------------------------
The test set is kept at its natural class distribution (not balanced to 50/50).
Balancing would discard the clear samples needed for stable specificity /
PR-AUC estimates, and would embed a non-operational 50% prevalence assumption
into every reported precision number.  Instead, precision is projected for
stated operational prevalences via Bayes' theorem (``precision_at_prevalence``).

The model is trained on profile B only (nominal on-orbit condition).  The test
set carries profiles A/B/C, turning A and C into out-of-distribution robustness
bounds (graceful degradation from best-case to SAA-transit worst-case).

Per-profile is the primary unit of analysis: A/B/C tiles from the same scene
are near-duplicates whose scores are correlated.  Pooling them into one metric
would inflate effective sample size and smear score distributions that belong to
distinct operating conditions.  The ``nominal`` block (profile B only) is the
headline operating-point summary.
"""

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from nina_cloud.datagen import config as datagen_config
from nina_cloud.evaluation.metrics import (
    compute_classification_metrics,
    compute_pr_curve,
    group_metrics_by_profile,
)
from nina_cloud.training.train_cloud import CloudDataset

logger = logging.getLogger(__name__)

# Nominal training profile — the deployment condition and threshold-tuning basis.
_NOMINAL_PROFILE = "B"


def build_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Rebuild the MobileNetV2 classifier and load trained weights.

    Mirrors the architecture defined in ``training/train_cloud.py``.
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

    Identical to the val transform in ``training/train_cloud.py``.
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
    threshold: float = 0.5,
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
        threshold: Decision threshold for hard predictions (sigmoid score >=
            threshold → cloud).  Should be loaded from ``training_report.json``
            rather than hardcoded to 0.5.

    Returns:
        List of dicts — each input record with two extra keys added:
        ``score`` (sigmoid probability for class cloud=1) and ``pred``
        (hard prediction: 0 or 1 at *threshold*).
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

    all_scores: list[float] = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)  # (B,)
            scores = logits.sigmoid().cpu().tolist()
            all_scores.extend(scores)

    augmented = []
    for rec, score in zip(records, all_scores):
        pred = 1 if score >= threshold else 0
        augmented.append({**rec, "score": round(score, 6), "pred": pred})

    return augmented


def _resolve_threshold(
    dataset_dir: Path,
    explicit_threshold: Optional[float],
) -> tuple[float, Optional[float]]:
    """Return (threshold, beta) from training_report.json or fallback.

    Resolution order:
    1. *explicit_threshold* if provided (e.g. via ``--threshold`` CLI).
    2. ``selected_threshold`` in ``<dataset_dir>/training_report.json``.
    3. 0.5 with a warning (legacy / pre-threshold-sweep runs).

    Returns:
        (threshold, beta) where beta may be None if the report is absent.
    """
    if explicit_threshold is not None:
        logger.info("Using explicit threshold: %.4f", explicit_threshold)
        return explicit_threshold, None

    report_path = dataset_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            tr = json.load(f)
        thr = float(tr["selected_threshold"])
        beta = tr.get("hyperparameters", {}).get("beta")
        beta_val = float(beta) if beta is not None else None
        logger.info(
            "Threshold loaded from training_report.json: %.4f (beta=%s)",
            thr,
            f"{beta_val:.3f}" if beta_val is not None else "N/A",
        )
        return thr, beta_val

    warnings.warn(
        "training_report.json not found — falling back to threshold=0.5. "
        "Train the model with the updated train_cloud.py to generate it.",
        stacklevel=3,
    )
    return 0.5, None


def build_evaluation_report(
    records: list,
    image_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int = 16,
    split: str = "test",
    threshold: Optional[float] = None,
    beta: float = 2.0,
    eval_prevalences: Optional[list[float]] = None,
) -> dict:
    """Run end-to-end evaluation and write ``evaluation_summary.json``.

    Filters *records* to the requested *split*, runs inference with the
    checkpoint, and writes two files:

    * ``output_path`` — ``evaluation_summary.json`` with the full report.
    * ``<output_path.parent>/predictions.json`` — per-image predictions.

    Report structure
    ~~~~~~~~~~~~~~~~
    - ``nominal``    — profile-B-only metrics (the deployment headline).
    - ``by_profile`` — per-profile A/B/C metrics (robustness table).
    - No pooled-across-profiles ``overall``: A/B/C tiles are correlated
      near-duplicates; pooling smears distinct operating conditions.

    Args:
        records: All manifest records (will be filtered to *split*).
        image_dir: Directory containing the PNG images.
        checkpoint_path: Path to the trained checkpoint file.
        output_path: Destination path for ``evaluation_summary.json``.
        device: Inference device.
        batch_size: DataLoader batch size.
        split: Which manifest split to evaluate.
        threshold: Decision threshold.  When ``None``, resolved automatically
            from ``training_report.json`` in the same dataset directory.
        beta: F-beta weighting (beta²=C_FN/C_FP).  Overridden by
            ``training_report.json`` when *threshold* is not explicit.
        eval_prevalences: Cloud prevalences for Bayes-projected precision.

    Returns:
        The report dict written to *output_path*.
    """
    if eval_prevalences is None:
        eval_prevalences = datagen_config.EVAL_PREVALENCES

    # Resolve threshold (and potentially beta) from training_report.json
    resolved_threshold, reported_beta = _resolve_threshold(
        output_path.parent, threshold
    )
    if reported_beta is not None:
        beta = reported_beta

    split_records = [r for r in records if r.get("split") == split]
    logger.info(
        "Evaluating on %d '%s' records (of %d total)",
        len(split_records),
        split,
        len(records),
    )
    if not split_records:
        raise ValueError(f"No records with split=='{split}' found in manifest.")

    # ── Degenerate-eval guard ─────────────────────────────────────────────────
    n_clear = sum(1 for r in split_records if r.get("label") == 0)
    degenerate_eval = n_clear < datagen_config.MIN_CLEAR_FOR_EVAL
    if degenerate_eval:
        warnings.warn(
            f"Split '{split}' has only {n_clear} clear samples "
            f"(minimum is {datagen_config.MIN_CLEAR_FOR_EVAL}).  "
            "Specificity, precision, and PR-AUC estimates are unreliable.  "
            "Acquire more diverse scenes to enable meaningful evaluation.",
            stacklevel=2,
        )
        logger.warning(
            "Degenerate eval: %d clear samples in '%s' split (MIN_CLEAR_FOR_EVAL=%d).",
            n_clear,
            split,
            datagen_config.MIN_CLEAR_FOR_EVAL,
        )

    model = build_model(checkpoint_path, device)
    predictions = run_inference(
        model, split_records, image_dir, device, batch_size, resolved_threshold
    )

    # ── Nominal metrics (profile B — the deployment condition) ────────────────
    nominal_recs = [r for r in predictions if r.get("profile") == _NOMINAL_PROFILE]
    if not nominal_recs:
        # Fallback: use all records when profile B is absent (e.g. val-only runs
        # before test scenes have been acquired and simulated).
        logger.warning(
            "No profile-B records in '%s' split — using all records for nominal.",
            split,
        )
        nominal_recs = predictions

    y_true_nom = [r["label"] for r in nominal_recs]
    y_pred_nom = [r["pred"] for r in nominal_recs]
    y_score_nom = [r["score"] for r in nominal_recs]
    nominal = compute_classification_metrics(
        y_true_nom,
        y_pred_nom,
        y_score_nom,
        beta=beta,
        prevalences=eval_prevalences,
    )

    # ── Per-profile metrics (robustness table) ────────────────────────────────
    by_profile = group_metrics_by_profile(
        predictions, beta=beta, prevalences=eval_prevalences
    )

    report = {
        "stage": "run_evaluate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path.name,
        "threshold_used": resolved_threshold,
        "split": split,
        "device": str(device),
        "degenerate_eval": degenerate_eval,
        "nominal": nominal,
        "by_profile": by_profile,
        "notes": (
            f"'nominal' = profile {_NOMINAL_PROFILE} only (the on-orbit deployment "
            "condition; the threshold was tuned on this profile's val distribution).  "
            "Profiles A/C are out-of-distribution robustness bounds (model trained on "
            "B only).  Test set is at natural class distribution; 'precision' is "
            "measured-as-sampled while 'precision_at_prevalence' is the Bayes "
            "projection for the stated operational cloud prevalences."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written: %s", output_path)

    # Per-record predictions for downstream inspection and figure generation
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
    """Save evaluation figures as PNG files.

    Renders one figure per degradation profile (A/B/C) plus one for the nominal
    (profile-B) operating point.  Each figure has three panels:
    confusion matrix | ROC curve | Precision-Recall curve.

    Figures are written to ``<output_dir>/figures/``.
    Uses the ``Agg`` matplotlib backend for headless environments.

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
        """Render and save a 3-panel figure: confusion matrix | ROC | PR."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Evaluation — {group_name}", fontsize=13)

        # ── Panel 0: Confusion matrix (table, not imshow) ─────────────────────
        # Rendered as ax.table to avoid a matplotlib 3.10 + Python 3.14
        # infinite deepcopy bug in tick creation.
        cm_dict = metrics["confusion_matrix"]
        ax = axes[0]
        ax.axis("off")
        ax.set_title("Confusion Matrix", pad=14)
        cell_text = [
            [str(cm_dict["tn"]), str(cm_dict["fp"])],
            [str(cm_dict["fn"]), str(cm_dict["tp"])],
        ]
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=["True Clear", "True Cloud"],
            colLabels=["Pred Clear", "Pred Cloud"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.4, 2.0)
        # Shade the diagonal (TN=top-left, TP=bottom-right) in light blue
        for (row, col), cell in tbl.get_celld().items():
            if row > 0 and (row - 1) == col:
                cell.set_facecolor("#c6dbef")

        # ── Panel 1: ROC curve ────────────────────────────────────────────────
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

        # ── Panel 2: Precision-Recall curve ──────────────────────────────────
        ax3 = axes[2]
        pr_result = compute_pr_curve(
            [r["label"] for r in recs],
            [r["score"] for r in recs],
        )
        if pr_result is not None:
            p_vals, r_vals, _ = pr_result
            ap_val = metrics.get("average_precision") or 0.0
            ax3.plot(r_vals, p_vals, lw=2, label=f"AP = {ap_val:.3f}")
            # Baseline: random classifier precision = prevalence
            n_total = metrics.get("n", 1)
            n_cloud = metrics.get("n_cloud", 0)
            baseline = n_cloud / n_total if n_total > 0 else 0.5
            ax3.axhline(
                baseline,
                color="k",
                linestyle="--",
                lw=1,
                label=f"baseline (π={baseline:.2f})",
            )
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve")
            ax3.legend(loc="upper right")
            ax3.set_xlim([0, 1])
            ax3.set_ylim([0, 1.02])
        else:
            ax3.text(
                0.5,
                0.5,
                "Single class — PR N/A",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Precision-Recall Curve")

        # tight_layout triggers a matplotlib 3.10 + Python 3.14 deepcopy bug
        # (infinite recursion in matplotlib/path.py).  Use subplots_adjust instead.
        fig.subplots_adjust(top=0.88, wspace=0.4, left=0.08, right=0.97)
        safe_name = group_name.lower().replace(" ", "_")
        out_path = figures_dir / f"eval_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Figure saved: %s", out_path)

    # Nominal (profile B) figure
    b_recs = [r for r in predictions if r.get("profile") == _NOMINAL_PROFILE]
    if not b_recs:
        b_recs = predictions  # fallback when only one profile is present
    _plot_group(f"nominal_profile_{_NOMINAL_PROFILE}", report["nominal"], b_recs)

    # Per-profile robustness figures
    by_profile_recs: dict = {}
    for rec in predictions:
        p = rec.get("profile", "unknown")
        by_profile_recs.setdefault(p, []).append(rec)

    for profile, recs in sorted(by_profile_recs.items()):
        profile_metrics = report["by_profile"].get(profile)
        if profile_metrics:
            _plot_group(f"profile_{profile}", profile_metrics, recs)
