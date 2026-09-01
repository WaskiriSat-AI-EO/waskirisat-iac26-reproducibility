"""Operating-point selection on the validation set.

Fire detection carries a strongly asymmetric cost: a missed fire (FN) forfeits
the early-warning window the payload exists to provide, while a false alarm
(FP) costs one downlink slot.  The decision threshold is therefore not fixed at
0.5 — it is swept post-training to enforce a *recall floor*::

    t* = max { t : recall(t) >= recall_target }

Taking the maximum such threshold maximises precision subject to the floor.
Because recall is monotonically non-increasing in ``t``, the sweep reduces to a
single sort plus a cumulative sum rather than the quadratic scan of candidate
thresholds.

These helpers are deliberately local to ``nina_thermal`` rather than imported
from ``nina_cloud``: the repository is organised as one bounded context per
payload, and a cross-payload import would couple the fire model's operating
point to the cloud model's release cycle.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Threshold used when the sweep cannot produce a meaningful answer.
_FALLBACK_THRESHOLD = 0.5


def _tp_fp_by_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, tp, fp) for every distinct score, descending.

    ``tp[i]`` / ``fp[i]`` are the true and false positive counts obtained by
    predicting positive when ``score >= thresholds[i]``.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        y_score: Scores in [0, 1], shape ``(n,)``.

    Returns:
        Tuple of parallel arrays ordered by decreasing threshold.
    """
    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    labels = y_true[order]

    # Last index of each run of equal scores — a threshold must include every
    # tied sample, otherwise the reported recall is not achievable in practice.
    distinct = np.where(np.diff(scores))[0]
    last = np.r_[distinct, labels.size - 1]

    tp = np.cumsum(labels)[last]
    fp = (last + 1) - tp
    return scores[last], tp, fp


def select_threshold(
    y_true: list[int],
    y_score: list[float],
    recall_target: float,
) -> tuple[float, bool]:
    """Select the highest threshold meeting the fire-recall floor.

    Args:
        y_true: Binary fire labels.
        y_score: Sigmoid scores in [0, 1].
        recall_target: Minimum fire recall the operating point must achieve.

    Returns:
        ``(threshold, degenerate)``.  ``degenerate`` is True when the
        validation set holds a single class or no threshold reaches the
        target; the threshold then falls back to 0.5 and must not be trusted
        as an operating point.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)

    if np.unique(arr_true).size < 2:
        logger.warning(
            "Val set contains a single class — threshold sweep is undefined; "
            "falling back to %.1f.  Acquire granules covering both classes.",
            _FALLBACK_THRESHOLD,
        )
        return _FALLBACK_THRESHOLD, True

    thresholds, tp, _ = _tp_fp_by_threshold(arr_true, arr_score)
    recall = tp / int((arr_true == 1).sum())

    feasible = np.flatnonzero(recall >= recall_target)
    if feasible.size == 0:
        logger.warning(
            "No threshold reaches val fire recall >= %.2f (best %.3f); "
            "falling back to %.1f.  Lower --recall_target or acquire more "
            "fire granules.",
            recall_target,
            float(recall.max()),
            _FALLBACK_THRESHOLD,
        )
        return _FALLBACK_THRESHOLD, True

    # Thresholds descend, so the first feasible index is the highest threshold.
    return float(thresholds[feasible[0]]), False


def metrics_at_threshold(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
    beta: float,
) -> dict:
    """Compute point metrics at a fixed decision threshold.

    Args:
        y_true: Binary fire labels.
        y_score: Sigmoid scores in [0, 1].
        threshold: Decision threshold; predict fire when ``score >= threshold``.
        beta: F-beta weighting; ``beta > 1`` favours recall.

    Returns:
        Recall, precision, specificity, F-beta and the confusion matrix.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(((arr_pred == 1) & (arr_true == 1)).sum())
    fp = int(((arr_pred == 1) & (arr_true == 0)).sum())
    fn = int(((arr_pred == 0) & (arr_true == 1)).sum())
    tn = int(((arr_pred == 0) & (arr_true == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    beta2 = beta**2
    denom = beta2 * precision + recall
    fbeta = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0

    return {
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "specificity": round(specificity, 6),
        "fbeta": round(fbeta, 6),
        "beta": round(beta, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def precision_at_recall(
    y_true: list[int],
    y_score: list[float],
    recall_targets: list[float],
) -> dict[str, float | None]:
    """Report achievable precision at each recall floor.

    Args:
        y_true: Binary fire labels.
        y_score: Sigmoid scores in [0, 1].
        recall_targets: Recall floors to evaluate.

    Returns:
        Maps each target (as a string key) to the precision at the highest
        threshold meeting it, or ``None`` when unreachable.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    n_pos = int((arr_true == 1).sum())

    if n_pos == 0 or np.unique(arr_true).size < 2:
        return {str(target): None for target in recall_targets}

    _, tp, fp = _tp_fp_by_threshold(arr_true, arr_score)
    recall = tp / n_pos

    result: dict[str, float | None] = {}
    for target in recall_targets:
        feasible = np.flatnonzero(recall >= target)
        if feasible.size == 0:
            result[str(target)] = None
            continue
        idx = feasible[0]
        predicted_positive = tp[idx] + fp[idx]
        result[str(target)] = (
            round(float(tp[idx] / predicted_positive), 6) if predicted_positive else 0.0
        )
    return result


def average_precision(y_true: list[int], y_score: list[float]) -> float | None:
    """Area under the precision-recall curve (PR-AUC).

    Reported alongside the operating point because a single threshold hides
    whether the model ranks fire above no-fire across the whole score range —
    the property that matters when the threshold is later re-swept on the
    quantized model.

    Args:
        y_true: Binary fire labels.
        y_score: Sigmoid scores in [0, 1].

    Returns:
        Average precision, or ``None`` when only one class is present.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    if np.unique(arr_true).size < 2:
        return None

    _, tp, fp = _tp_fp_by_threshold(arr_true, arr_score)
    n_pos = int((arr_true == 1).sum())
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)

    # Step-wise integral: sum precision_i * (recall_i - recall_{i-1}).
    return round(float(np.sum(precision * np.diff(np.r_[0.0, recall]))), 6)


def roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """Area under the ROC curve.

    Reported alongside PR-AUC because it is invariant to class prevalence,
    which differs between the val and test splits here.  A value near 0.5 means
    the model cannot rank fire above no-fire at any threshold — a diagnosis the
    operating-point metrics can mask, since a recall-floor sweep will always
    report high recall by simply predicting fire everywhere.

    Args:
        y_true: Binary fire labels.
        y_score: Sigmoid scores in [0, 1].

    Returns:
        ROC-AUC, or ``None`` when only one class is present.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)
    n_pos = int((arr_true == 1).sum())
    n_neg = arr_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Mann-Whitney U via mid-ranks, which handles tied scores correctly.
    order = np.argsort(arr_score, kind="mergesort")
    ranks = np.empty(arr_score.size, dtype=float)
    sorted_scores = arr_score[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_sum_pos = float(ranks[arr_true == 1].sum())
    return round((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 6)
