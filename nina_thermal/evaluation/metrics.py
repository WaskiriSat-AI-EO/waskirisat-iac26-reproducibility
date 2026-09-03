"""Classification metrics for the fire / no-fire evaluation report.

Torch-free (numpy + scikit-learn only) so it imports and unit-tests in CI
without a GPU build. The recall-floor threshold sweep is **not** reimplemented
here — :mod:`nina_thermal.training.threshold` is the single source of truth for
the operating point, and this module reuses its ``roc_auc`` for the bootstrap.

Design notes
------------
* **Asymmetric cost.** A missed wildfire (FN) forfeits the early-warning
  window; a false alarm (FP) costs one downlink. Fire *recall* is the primary
  metric; F-beta uses ``beta = sqrt(pos_weight)`` to stay coherent with the
  weighted training loss.
* **Seasonal prevalence.** A synthesized patch covers 640 x 480 km
  (~307,000 km^2; the flight frame is 570 x 428 km, ~a quarter of Bolivia) and
  fire prevalence over it is strongly seasonal, so the raw test-set precision
  is not an operational number. :func:`precision_at_prevalence` projects it for
  stated prevalences via Bayes' theorem.
* **Granule-level bootstrap.** 50% tile overlap makes adjacent patches share up
  to 75% of their pixels, so :func:`cluster_bootstrap_ci` resamples *granules*,
  never patches.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any

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

from nina_thermal.datagen import config
from nina_thermal.training.threshold import roc_auc as _roc_auc_np

logger = logging.getLogger(__name__)

# ── Report parameters ────────────────────────────────────────────────────────
# Operational fire prevalences for the Bayes-projected precision block. Near 1.0
# during the Aug-Sep acquisition peak, near 0 off-season; both bounds reported
# because the true mix depends on when NINA observes.
EVAL_PREVALENCES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Below this many fire patches the recall-target sweep and precision estimates
# are too noisy to trust; the report is flagged ``degenerate_eval`` instead.
MIN_FIRE_FOR_EVAL: int = 20

# Granule-level cluster bootstrap.
BOOTSTRAP_N_RESAMPLES: int = 1000
BOOTSTRAP_CI: float = 0.90

# The profile the model trains on; per-profile deltas are reported against it.
NOMINAL_PROFILE: str = "B"


# ── Prevalence-projected precision ───────────────────────────────────────────


def precision_at_prevalence(
    recall: float, specificity: float, prevalence: float
) -> float:
    """Project precision at a stated fire prevalence via Bayes' theorem.

    ``precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR)`` with ``FPR = 1 - specificity``.

    Args:
        recall: Fire recall (TPR). Prevalence-free.
        specificity: True-negative rate. Prevalence-free.
        prevalence: Assumed fraction of fire patches on orbit.

    Returns:
        Projected precision, rounded to 6 dp; 0.0 for a degenerate classifier.
    """
    tpr = recall
    fpr = 1.0 - specificity
    denom = prevalence * tpr + (1.0 - prevalence) * fpr
    if denom == 0.0:
        return 0.0
    return round(prevalence * tpr / denom, 6)


def compute_pr_curve(
    y_true: list[int], y_score: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``sklearn.precision_recall_curve`` arrays, or ``None`` if single-class."""
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    if len(np.unique(arr_true)) < 2:
        return None
    return precision_recall_curve(arr_true, arr_score)


# ── Core classification metrics ─────────────────────────────────────────────


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float],
    *,
    beta: float,
    prevalences: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Binary fire/no-fire metrics, recall first.

    Args:
        y_true: Ground-truth labels (1 = fire).
        y_pred: Hard predictions at the operating threshold.
        y_score: Sigmoid scores for class fire.
        beta: F-beta weighting (``sqrt(pos_weight)``).
        prevalences: When given, add a ``precision_at_prevalence`` sub-dict
            keyed by ``str(pi)``.

    Returns:
        Metrics dict. ``roc_auc`` / ``average_precision`` are ``None`` when
        only one class is present.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_pred = np.asarray(y_pred, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)

    n_fire = int((arr_true == 1).sum())
    n_nofire = int((arr_true == 0).sum())

    cm = confusion_matrix(arr_true, arr_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    both = len(np.unique(arr_true)) >= 2
    recall = float(recall_score(arr_true, arr_pred, zero_division=0))

    result: dict[str, Any] = {
        "n": int(arr_true.size),
        "n_fire": n_fire,
        "n_nofire": n_nofire,
        # headline, prevalence-free
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "roc_auc": (
            round(float(roc_auc_score(arr_true, arr_score)), 6) if both else None
        ),
        "average_precision": (
            round(float(average_precision_score(arr_true, arr_score)), 6)
            if both
            else None
        ),
        "fbeta": round(
            float(fbeta_score(arr_true, arr_pred, beta=beta, zero_division=0)), 6
        ),
        "beta": round(beta, 6),
        # secondary, distribution-dependent
        "precision": round(
            float(precision_score(arr_true, arr_pred, zero_division=0)), 6
        ),
        "accuracy": round(float(accuracy_score(arr_true, arr_pred)), 6),
        "f1": round(float(f1_score(arr_true, arr_pred, zero_division=0)), 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if prevalences is not None:
        result["precision_at_prevalence"] = {
            str(pi): precision_at_prevalence(recall, specificity, pi)
            for pi in prevalences
        }
    return result


# ── Per-slice grouping ──────────────────────────────────────────────────────


def group_metrics(
    records: list[dict[str, Any]],
    key: str,
    *,
    beta: float,
    prevalences: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Per-value classification metrics for a categorical field.

    Args:
        records: Prediction dicts carrying ``key``, ``label``, ``pred``, ``score``.
        key: Field to group by (``"profile"`` or ``"surface_type"``).
        beta: F-beta weighting, passed through.
        prevalences: Passed through.

    Returns:
        ``{value: metrics_dict}``, sorted by value; empty groups omitted.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        buckets.setdefault(str(rec[key]), []).append(rec)

    return {
        value: compute_classification_metrics(
            [r["label"] for r in rows],
            [r["pred"] for r in rows],
            [r["score"] for r in rows],
            beta=beta,
            prevalences=prevalences,
        )
        for value, rows in sorted(buckets.items())
    }


def profile_deltas(
    by_profile: dict[str, dict[str, Any]], baseline: str = NOMINAL_PROFILE
) -> dict[str, dict[str, float]]:
    """``metric(profile) - metric(baseline)`` for the robustness table.

    Profiles A (post-FFC best case) and C (SAA worst case) are out-of-distribution
    for a model trained on *baseline* only; the signed drop is the reportable
    robustness bound (mission plan risk PH-05).

    Args:
        by_profile: Output of :func:`group_metrics` with ``key="profile"``.
        baseline: Profile the model trained on.

    Returns:
        ``{profile: {metric: delta}}`` for every profile except *baseline*.
        Empty when the baseline profile is absent.
    """
    base = by_profile.get(baseline)
    if base is None:
        return {}
    fields = (
        "recall",
        "specificity",
        "roc_auc",
        "average_precision",
        "fbeta",
        "precision",
    )
    out: dict[str, dict[str, float]] = {}
    for profile, metrics in by_profile.items():
        if profile == baseline:
            continue
        out[profile] = {
            f: round(metrics[f] - base[f], 6)
            for f in fields
            if metrics.get(f) is not None and base.get(f) is not None
        }
    return out


# ── Granule-level cluster bootstrap ─────────────────────────────────────────


def cluster_bootstrap_ci(
    y_true: list[int],
    y_score: list[float],
    groups: list[str],
    metric_fn: Callable[[np.ndarray, np.ndarray], float | None],
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, Any]:
    """Granule-level cluster bootstrap CI for an arbitrary metric.

    Resamples *granules* with replacement. ``metric_fn`` returns ``None`` /
    ``nan`` for undefined resamples (e.g. one class only); those are dropped.

    Returns:
        ``{point_estimate, ci_low, ci_high, n_resamples_used,
        n_resamples_requested}``.
    """
    rng = np.random.default_rng(seed)
    arr_true = np.asarray(y_true)
    arr_score = np.asarray(y_score)
    groups_arr = np.asarray(groups)
    unique_groups = np.array(sorted(set(groups)))

    result: dict[str, Any] = {
        "point_estimate": None,
        "ci_low": None,
        "ci_high": None,
        "n_resamples_used": 0,
        "n_resamples_requested": n_resamples,
    }
    if unique_groups.size == 0:
        return result

    try:
        point = metric_fn(arr_true, arr_score)
        result["point_estimate"] = (
            float(point) if point is not None and not np.isnan(point) else None
        )
    except Exception:  # noqa: BLE001 - a broken metric_fn must not abort the report
        result["point_estimate"] = None

    group_to_idx = {g: np.where(groups_arr == g)[0] for g in unique_groups}
    values: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        try:
            v = metric_fn(arr_true[idx], arr_score[idx])
        except (
            Exception
        ):  # noqa: BLE001, S112 - a degenerate resample is skipped by design
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        values.append(float(v))

    result["n_resamples_used"] = len(values)
    if values:
        alpha = 1.0 - ci
        result["ci_low"] = float(np.percentile(values, 100.0 * alpha / 2.0))
        result["ci_high"] = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    else:
        logger.warning("cluster_bootstrap_ci: no resample produced a defined value.")
    return result


# ── Bootstrap metric_fn factories ──────────────────────────────────────────


def recall_fn(threshold: float) -> Callable[[np.ndarray, np.ndarray], float | None]:
    """Bootstrap ``metric_fn`` for fire recall at a fixed threshold."""

    def _fn(yt: np.ndarray, ys: np.ndarray) -> float | None:
        n_pos = int((yt == 1).sum())
        if n_pos == 0:
            return None
        preds = (ys >= threshold).astype(int)
        return float(((preds == 1) & (yt == 1)).sum() / n_pos)

    return _fn


def precision_fn(threshold: float) -> Callable[[np.ndarray, np.ndarray], float | None]:
    """Bootstrap ``metric_fn`` for precision at a fixed threshold."""

    def _fn(yt: np.ndarray, ys: np.ndarray) -> float | None:
        preds = (ys >= threshold).astype(int)
        n_pred_pos = int((preds == 1).sum())
        if n_pred_pos == 0:
            return None
        return float(((preds == 1) & (yt == 1)).sum() / n_pred_pos)

    return _fn


def fbeta_fn(
    threshold: float, beta: float
) -> Callable[[np.ndarray, np.ndarray], float | None]:
    """Bootstrap ``metric_fn`` for F-beta at a fixed threshold."""

    def _fn(yt: np.ndarray, ys: np.ndarray) -> float | None:
        preds = (ys >= threshold).astype(int)
        return float(fbeta_score(yt, preds, beta=beta, zero_division=0))

    return _fn


def roc_auc_fn(yt: np.ndarray, ys: np.ndarray) -> float | None:
    """Bootstrap ``metric_fn`` for ROC-AUC (threshold-free)."""
    return _roc_auc_np(yt.tolist(), ys.tolist())
