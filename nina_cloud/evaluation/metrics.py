"""Pure classification metrics for cloud/clear binary evaluation.

All functions operate on plain Python lists or numpy arrays and depend only on
numpy and scikit-learn — no torch required.  This keeps the module importable
and unit-testable in CI environments that do not have GPU/PyTorch installed.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    y_score: list,
) -> dict:
    """Compute binary classification metrics for the cloud/clear task.

    Args:
        y_true: Ground-truth labels (1=cloud, 0=clear).
        y_pred: Hard predictions (1=cloud, 0=clear) at threshold 0.5.
        y_score: Predicted probabilities for class 1 (sigmoid output).

    Returns:
        Dict with keys: ``n``, ``n_cloud``, ``n_clear``, ``accuracy``,
        ``precision``, ``recall``, ``f1``, ``specificity``, ``roc_auc``
        (``None`` when only one class is present), and ``confusion_matrix``
        (dict with ``tn``, ``fp``, ``fn``, ``tp``).
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_pred = np.asarray(y_pred, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)

    n = len(arr_true)
    n_cloud = int((arr_true == 1).sum())
    n_clear = int((arr_true == 0).sum())

    accuracy = float(accuracy_score(arr_true, arr_pred))
    precision = float(precision_score(arr_true, arr_pred, zero_division=0))
    recall = float(recall_score(arr_true, arr_pred, zero_division=0))
    f1 = float(f1_score(arr_true, arr_pred, zero_division=0))

    # confusion_matrix with fixed label order [0, 1] so indexing is stable
    cm = confusion_matrix(arr_true, arr_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP) — true-negative rate for the "clear" class
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC requires both classes to be present
    roc_auc: Optional[float]
    unique_classes = np.unique(arr_true)
    if len(unique_classes) < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(arr_true, arr_score))

    return {
        "n": n,
        "n_cloud": n_cloud,
        "n_clear": n_clear,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "specificity": round(specificity, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def group_metrics_by_profile(records_with_preds: list) -> dict:
    """Compute per-degradation-profile classification metrics.

    Args:
        records_with_preds: List of record dicts, each containing at least
            ``profile`` (str), ``label`` (int), ``pred`` (int), and
            ``score`` (float) keys.

    Returns:
        Dict mapping profile name (e.g. ``"A"``, ``"B"``, ``"C"``) to a
        metrics dict as returned by :func:`compute_classification_metrics`.
        Profiles with no records are omitted.
    """
    by_profile: dict = {}
    for rec in records_with_preds:
        p = rec["profile"]
        by_profile.setdefault(p, []).append(rec)

    result: dict = {}
    for profile, recs in sorted(by_profile.items()):
        y_true = [r["label"] for r in recs]
        y_pred = [r["pred"] for r in recs]
        y_score = [r["score"] for r in recs]
        result[profile] = compute_classification_metrics(y_true, y_pred, y_score)

    return result
