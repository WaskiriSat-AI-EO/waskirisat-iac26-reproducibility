"""Pure classification metrics for cloud/clear binary evaluation.

All functions operate on plain Python lists or numpy arrays and depend only on
numpy and scikit-learn — no torch required.  This keeps the module importable
and unit-testable in CI environments that do not have GPU/PyTorch installed.

Design note — asymmetric error costs
-------------------------------------
The cloud detector is a safety filter: a False Negative (cloud predicted clear)
causes corrupted science data to be downlinked, wasting irreplaceable bandwidth.
A False Positive (clear predicted cloud) merely drops a valid image.

Primary metric: cloud **recall** (sensitivity / TPR).
Secondary metrics: specificity, ROC-AUC, Average Precision (PR-AUC).
Operational precision: reported as ``precision_at_prevalence`` for stated π
values via Bayes' theorem rather than from the test-set sample proportion
(which depends on the data collection protocol, not on the orbit).

F-beta uses beta=sqrt(pos_weight) to stay coherent with the weighted loss.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_prevalence(
    recall: float,
    specificity: float,
    prevalence: float,
) -> float:
    """Project precision at a stated cloud prevalence via Bayes' theorem.

    Because the test set is kept at its natural class distribution (not
    artificially balanced), raw precision depends on the data collection
    protocol rather than on the classifier.  This function separates that
    dependence: given the prevalence-free intrinsic quantities (recall = TPR
    and specificity = TNR) it projects precision for any assumed operational
    cloud prevalence π.

    Formula::

        FPR = 1 - specificity
        precision(π) = π·TPR / (π·TPR + (1-π)·FPR)

    Args:
        recall: Cloud recall (TPR = TP / P).  Prevalence-free.
        specificity: True-negative rate (TNR = TN / N).  Prevalence-free.
        prevalence: Assumed fraction of cloud images on-orbit (0 < π < 1).

    Returns:
        Projected precision at the given prevalence, rounded to 6 dp.
        Returns 0.0 when both TPR and FPR are zero (degenerate classifier).
    """
    tpr = recall
    fpr = 1.0 - specificity
    denom = prevalence * tpr + (1.0 - prevalence) * fpr
    if denom == 0.0:
        return 0.0
    return round(prevalence * tpr / denom, 6)


def compute_pr_curve(
    y_true: list,
    y_score: list,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the precision-recall curve arrays.

    Args:
        y_true: Ground-truth labels (1=cloud, 0=clear).
        y_score: Predicted probabilities for class 1 (sigmoid output).

    Returns:
        ``(precision_vals, recall_vals, thresholds)`` numpy arrays as returned
        by ``sklearn.metrics.precision_recall_curve``, or ``None`` when only
        one class is present in *y_true*.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    if len(np.unique(arr_true)) < 2:
        return None
    p_vals, r_vals, thresholds = precision_recall_curve(arr_true, arr_score)
    return p_vals, r_vals, thresholds


def compute_classification_metrics(
    y_true: list,
    y_pred: list,
    y_score: list,
    *,
    beta: float = 2.0,
    prevalences: Optional[list[float]] = None,
) -> dict:
    """Compute binary classification metrics for the cloud/clear task.

    Field ordering is intentional: recall (the mission-primary metric) appears
    first, followed by other prevalence-free metrics, then prevalence-projected
    precision, and finally secondary metrics that assume the test distribution.

    Args:
        y_true: Ground-truth labels (1=cloud, 0=clear).
        y_pred: Hard predictions (1=cloud, 0=clear) at the operating threshold.
        y_score: Predicted probabilities for class 1 (sigmoid output).
        beta: F-beta weighting.  Set to ``sqrt(pos_weight)`` so it is coherent
            with the weighted training loss (beta²=C_FN/C_FP).  Default: 2.0.
        prevalences: Optional list of cloud prevalences π for Bayes-projected
            precision.  When provided, a ``precision_at_prevalence`` sub-dict
            is included in the output (keyed by str(π)).

    Returns:
        Dict with fields ordered recall-first.  Key ``roc_auc`` and
        ``average_precision`` are ``None`` when only one class is present.
        ``precision_at_prevalence`` is only present when *prevalences* is given.
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
    fbeta = float(fbeta_score(arr_true, arr_pred, beta=beta, zero_division=0))

    # confusion_matrix with fixed label order [0, 1] so indexing is stable
    cm = confusion_matrix(arr_true, arr_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Specificity = TN / (TN + FP) — true-negative rate for the "clear" class
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Score-based metrics require both classes to be present
    unique_classes = np.unique(arr_true)
    both_classes = len(unique_classes) >= 2

    roc_auc: Optional[float] = (
        float(roc_auc_score(arr_true, arr_score)) if both_classes else None
    )
    avg_prec: Optional[float] = (
        float(average_precision_score(arr_true, arr_score)) if both_classes else None
    )

    result: dict = {
        "n": n,
        "n_cloud": n_cloud,
        "n_clear": n_clear,
        # ── Prevalence-free headline metrics (mission-primary) ──────────────
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
        "average_precision": round(avg_prec, 6) if avg_prec is not None else None,
        "fbeta": round(fbeta, 6),
        "beta": round(beta, 6),
        # ── Secondary metrics (distribution-dependent) ──────────────────────
        "precision": round(precision, 6),
        "accuracy": round(accuracy, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if prevalences is not None:
        result["precision_at_prevalence"] = {
            str(pi): precision_at_prevalence(recall, specificity, pi)
            for pi in prevalences
        }

    return result


def group_metrics_by_profile(
    records_with_preds: list,
    *,
    beta: float = 2.0,
    prevalences: Optional[list[float]] = None,
) -> dict:
    """Compute per-degradation-profile classification metrics.

    Per-profile is the primary unit of analysis for the cross-degradation
    robustness evaluation (test set carries profiles A/B/C; each tile's A/B/C
    variants are near-duplicates correlated by scene content, so pooling them
    would understate variance and smear the score distributions of distinct
    operating conditions).

    Args:
        records_with_preds: List of record dicts, each containing at least
            ``profile`` (str), ``label`` (int), ``pred`` (int), and
            ``score`` (float) keys.
        beta: F-beta weighting passed to :func:`compute_classification_metrics`.
        prevalences: Prevalences for Bayes-projected precision (passed through).

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
        result[profile] = compute_classification_metrics(
            y_true, y_pred, y_score, beta=beta, prevalences=prevalences
        )

    return result
