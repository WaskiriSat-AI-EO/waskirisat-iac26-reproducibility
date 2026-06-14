"""Tests for evaluation/metrics.py.

All tests are pure numpy/sklearn — no torch required, so they run cleanly in CI
environments without GPU or PyTorch installed.
"""

import math

import numpy as np

from nina_cloud.evaluation.metrics import (
    compute_classification_metrics,
    compute_pr_curve,
    group_metrics_by_profile,
    precision_at_prevalence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(
    n_cloud: int,
    n_clear: int,
    profile: str = "B",
    correct: bool = True,
) -> list:
    """Build synthetic prediction records.

    Args:
        n_cloud: Number of cloud records (label=1).
        n_clear: Number of clear records (label=0).
        profile: Degradation profile assigned to all records.
        correct: If True all predictions are correct, otherwise all are wrong.

    Returns:
        List of dicts with ``label``, ``pred``, ``score``, ``profile`` keys.
    """
    records = []
    for _ in range(n_cloud):
        label = 1
        pred = label if correct else 1 - label
        score = 0.9 if pred == 1 else 0.1
        records.append(
            {"label": label, "pred": pred, "score": score, "profile": profile}
        )
    for _ in range(n_clear):
        label = 0
        pred = label if correct else 1 - label
        score = 0.1 if pred == 0 else 0.9
        records.append(
            {"label": label, "pred": pred, "score": score, "profile": profile}
        )
    return records


# ---------------------------------------------------------------------------
# Tests for precision_at_prevalence
# ---------------------------------------------------------------------------


class TestPrecisionAtPrevalence:
    """Tests for precision_at_prevalence."""

    def test_perfect_classifier_any_prevalence(self):
        """TPR=1, FPR=0 → precision=1 regardless of prevalence."""
        for pi in [0.1, 0.3, 0.5, 0.9]:
            result = precision_at_prevalence(recall=1.0, specificity=1.0, prevalence=pi)
            assert result == 1.0, f"Failed at prevalence={pi}"

    def test_zero_recall_gives_zero_precision(self):
        """TPR=0 → precision=0 (numerator is zero)."""
        result = precision_at_prevalence(recall=0.0, specificity=0.5, prevalence=0.5)
        assert result == 0.0

    def test_balanced_prevalence_reduces_to_sample_precision(self):
        """At π=0.5 the formula reduces to TP/(TP+FP) = TPR/(TPR+FPR)."""
        recall = 0.8
        specificity = 0.7
        fpr = 1.0 - specificity
        expected = recall / (recall + fpr)
        result = precision_at_prevalence(
            recall=recall, specificity=specificity, prevalence=0.5
        )
        assert abs(result - expected) < 1e-5, f"result={result}, expected={expected}"

    def test_lower_prevalence_decreases_precision(self):
        """For a fixed classifier, lower cloud prevalence → lower precision."""
        recall, specificity = 0.9, 0.8
        prec_high = precision_at_prevalence(recall, specificity, prevalence=0.7)
        prec_low = precision_at_prevalence(recall, specificity, prevalence=0.2)
        assert prec_low < prec_high, "Lower prevalence should yield lower precision"

    def test_degenerate_both_zero(self):
        """TPR=0 and FPR=0 → precision=0 (denominator guard)."""
        result = precision_at_prevalence(recall=0.0, specificity=1.0, prevalence=0.5)
        assert result == 0.0

    def test_known_values(self):
        """Hand-crafted example: TPR=0.9, FPR=0.1, π=0.5 → precision=0.9."""
        # 0.5*0.9 / (0.5*0.9 + 0.5*0.1) = 0.45/0.5 = 0.9
        result = precision_at_prevalence(recall=0.9, specificity=0.9, prevalence=0.5)
        assert abs(result - 0.9) < 1e-5, f"result={result}"


# ---------------------------------------------------------------------------
# Tests for compute_pr_curve
# ---------------------------------------------------------------------------


class TestComputePrCurve:
    """Tests for compute_pr_curve."""

    def test_returns_none_for_single_class(self):
        """Single-class input (all cloud) → None, no crash."""
        y_true = [1, 1, 1, 1]
        y_score = [0.9, 0.8, 0.7, 0.6]
        result = compute_pr_curve(y_true, y_score)
        assert result is None

    def test_returns_tuple_for_both_classes(self):
        """Both classes present → returns (precision, recall, thresholds) arrays."""
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        result = compute_pr_curve(y_true, y_score)
        assert result is not None
        p_vals, r_vals, thresholds = result
        assert isinstance(p_vals, np.ndarray)
        assert isinstance(r_vals, np.ndarray)
        assert isinstance(thresholds, np.ndarray)

    def test_precision_recall_in_valid_range(self):
        """All precision and recall values are in [0, 1]."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_score = [r["score"] for r in records]
        result = compute_pr_curve(y_true, y_score)
        assert result is not None
        p_vals, r_vals, _ = result
        assert np.all(p_vals >= 0) and np.all(p_vals <= 1)
        assert np.all(r_vals >= 0) and np.all(r_vals <= 1)

    def test_perfect_classifier_ap_one(self):
        """Perfect predictions → average precision should be 1.0."""
        from sklearn.metrics import average_precision_score

        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        ap = average_precision_score(y_true, y_score)
        assert abs(ap - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests for compute_classification_metrics
# ---------------------------------------------------------------------------


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    def test_perfect_predictions(self):
        """Perfect predictions → recall, specificity, precision, f1 all 1.0."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["recall"] == 1.0
        assert m["specificity"] == 1.0
        assert m["precision"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["average_precision"] == 1.0

    def test_counts(self):
        """n, n_cloud, n_clear counts are correct."""
        records = _make_records(7, 3, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["n"] == 10
        assert m["n_cloud"] == 7
        assert m["n_clear"] == 3

    def test_known_confusion_matrix(self):
        """Hand-crafted preds → verify precision, recall, F1, specificity."""
        # TP=3, FN=1, FP=2, TN=4
        # precision = 3/5 = 0.6
        # recall = 3/4 = 0.75
        # specificity = 4/6 = 0.666...
        y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        y_score = [0.9, 0.8, 0.7, 0.4, 0.6, 0.55, 0.3, 0.2, 0.15, 0.1]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["confusion_matrix"] == {"tn": 4, "fp": 2, "fn": 1, "tp": 3}
        assert abs(m["precision"] - 0.6) < 1e-5
        assert abs(m["recall"] - 0.75) < 1e-5
        expected_f1 = 2 * 0.6 * 0.75 / (0.6 + 0.75)
        assert abs(m["f1"] - expected_f1) < 1e-5
        assert abs(m["specificity"] - (4 / 6)) < 1e-5

    def test_single_class_no_score_metrics(self):
        """When only one class is present roc_auc and average_precision are None."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_score = [0.1, 0.2, 0.15, 0.05]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["roc_auc"] is None
        assert m["average_precision"] is None
        assert m["n"] == 4
        assert m["n_cloud"] == 0
        assert m["n_clear"] == 4

    def test_all_wrong_zero_recall_precision(self):
        """All predictions inverted → recall=0, precision=0."""
        records = _make_records(5, 5, correct=False)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["accuracy"] == 0.0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_confusion_matrix_all_correct(self):
        """Perfect predictions → TP and TN equal the class sizes, FP=FN=0."""
        records = _make_records(4, 6, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["confusion_matrix"]["tp"] == 4
        assert m["confusion_matrix"]["tn"] == 6
        assert m["confusion_matrix"]["fp"] == 0
        assert m["confusion_matrix"]["fn"] == 0

    def test_fbeta_with_default_beta(self):
        """fbeta with beta=2 weights recall more than precision."""
        # Perfect predictions → fbeta=1.0
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score, beta=2.0)
        assert m["fbeta"] == 1.0
        assert abs(m["beta"] - 2.0) < 1e-5

    def test_fbeta_from_pos_weight(self):
        """beta=sqrt(pos_weight) is accepted and stored in the result."""
        pos_weight = 3.0
        beta = math.sqrt(pos_weight)
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score, beta=beta)
        assert abs(m["beta"] - beta) < 1e-5
        assert m["fbeta"] == 1.0  # still perfect preds

    def test_fbeta_weights_recall_over_precision(self):
        """With beta>1, a high-recall / low-precision classifier scores
        better on fbeta than a low-recall / high-precision one."""
        # Classifier A: high recall, lower precision
        y_true = [1, 1, 1, 1, 0, 0, 0, 0]
        y_pred_a = [1, 1, 1, 1, 1, 1, 0, 0]  # TP=4,FP=2,FN=0,TN=2
        y_score = [0.9, 0.85, 0.8, 0.75, 0.6, 0.55, 0.3, 0.2]
        m_a = compute_classification_metrics(y_true, y_pred_a, y_score, beta=2.0)

        # Classifier B: lower recall, higher precision
        y_pred_b = [1, 1, 0, 0, 0, 0, 0, 0]  # TP=2,FP=0,FN=2,TN=4
        m_b = compute_classification_metrics(y_true, y_pred_b, y_score, beta=2.0)

        assert m_a["recall"] > m_b["recall"], "A should have higher recall"
        assert m_a["fbeta"] > m_b["fbeta"], (
            f"High-recall classifier A (fbeta={m_a['fbeta']:.3f}) should score "
            f"higher than low-recall B (fbeta={m_b['fbeta']:.3f}) at beta=2"
        )

    def test_prevalence_projected_precision_present(self):
        """When prevalences are given, precision_at_prevalence is in the result."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(
            y_true, y_pred, y_score, prevalences=[0.30, 0.50]
        )

        assert "precision_at_prevalence" in m
        assert "0.3" in m["precision_at_prevalence"]
        assert "0.5" in m["precision_at_prevalence"]
        # Perfect classifier → projected precision = 1.0 at any prevalence
        assert m["precision_at_prevalence"]["0.3"] == 1.0
        assert m["precision_at_prevalence"]["0.5"] == 1.0

    def test_prevalence_projected_precision_absent_without_arg(self):
        """Without prevalences arg, precision_at_prevalence is NOT in result."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)
        assert "precision_at_prevalence" not in m

    def test_metric_keys_present(self):
        """All expected base keys are present in the returned dict."""
        records = _make_records(3, 3, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        expected_keys = {
            "n",
            "n_cloud",
            "n_clear",
            "recall",
            "specificity",
            "roc_auc",
            "average_precision",
            "fbeta",
            "beta",
            "precision",
            "accuracy",
            "f1",
            "confusion_matrix",
        }
        assert expected_keys == set(m.keys())
        cm_keys = {"tn", "fp", "fn", "tp"}
        assert cm_keys == set(m["confusion_matrix"].keys())

    def test_roc_auc_present_when_both_classes(self):
        """roc_auc is a float in [0,1] when both classes are present."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["roc_auc"] is not None
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_average_precision_in_valid_range(self):
        """average_precision is a float in [0,1] when both classes present."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["average_precision"] is not None
        assert 0.0 <= m["average_precision"] <= 1.0


# ---------------------------------------------------------------------------
# Tests for group_metrics_by_profile
# ---------------------------------------------------------------------------


class TestGroupMetricsByProfile:
    """Tests for group_metrics_by_profile."""

    def test_groups_abc_profiles(self):
        """Records spanning A/B/C are grouped into three separate metrics dicts."""
        records = (
            _make_records(4, 4, profile="A")
            + _make_records(3, 3, profile="B")
            + _make_records(2, 2, profile="C")
        )
        result = group_metrics_by_profile(records)

        assert set(result.keys()) == {"A", "B", "C"}
        assert result["A"]["n"] == 8
        assert result["B"]["n"] == 6
        assert result["C"]["n"] == 4

    def test_missing_profile_omitted(self):
        """Profiles not present in the records are not included in the output."""
        records = _make_records(5, 5, profile="B")
        result = group_metrics_by_profile(records)

        assert "A" not in result
        assert "C" not in result
        assert "B" in result

    def test_single_profile_perfect_metrics(self):
        """Single-profile group of correct predictions → recall=1.0."""
        records = _make_records(6, 6, profile="A", correct=True)
        result = group_metrics_by_profile(records)

        assert result["A"]["recall"] == 1.0
        assert result["A"]["n"] == 12

    def test_profile_counts_sum_to_total(self):
        """Total records across all profiles equals len(input)."""
        records = (
            _make_records(3, 3, profile="A")
            + _make_records(4, 4, profile="B")
            + _make_records(5, 5, profile="C")
        )
        result = group_metrics_by_profile(records)

        total = sum(m["n"] for m in result.values())
        assert total == len(records)

    def test_empty_input_returns_empty_dict(self):
        """Empty records list returns an empty dict."""
        result = group_metrics_by_profile([])
        assert result == {}

    def test_beta_threaded_through(self):
        """beta parameter propagates to per-profile metrics dicts."""
        records = _make_records(5, 5, profile="B", correct=True)
        result = group_metrics_by_profile(records, beta=math.sqrt(3.0))
        assert abs(result["B"]["beta"] - math.sqrt(3.0)) < 1e-5

    def test_prevalences_threaded_through(self):
        """prevalences parameter populates precision_at_prevalence per profile."""
        records = _make_records(5, 5, profile="B", correct=True)
        result = group_metrics_by_profile(records, prevalences=[0.30, 0.50])
        assert "precision_at_prevalence" in result["B"]
        assert "0.3" in result["B"]["precision_at_prevalence"]
        assert "0.5" in result["B"]["precision_at_prevalence"]

    def test_no_prevalences_key_when_omitted(self):
        """Without prevalences, precision_at_prevalence is absent per profile."""
        records = _make_records(5, 5, profile="B", correct=True)
        result = group_metrics_by_profile(records)
        assert "precision_at_prevalence" not in result["B"]
