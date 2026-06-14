"""Tests for evaluation/metrics.py.

All tests are pure numpy/sklearn — no torch required, so they run cleanly in CI
environments without GPU or PyTorch installed.
"""

from nina_cloud.evaluation.metrics import (
    compute_classification_metrics,
    group_metrics_by_profile,
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
# Tests for compute_classification_metrics
# ---------------------------------------------------------------------------


class TestComputeClassificationMetrics:
    """Tests for compute_classification_metrics."""

    def test_perfect_predictions_all_ones(self):
        """Perfect cloud-only predictions → all scalar metrics are 1.0."""
        # Need both classes for AUC; use 5 cloud + 5 clear, all correct
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["accuracy"] == 1.0, f"accuracy={m['accuracy']}"
        assert m["precision"] == 1.0, f"precision={m['precision']}"
        assert m["recall"] == 1.0, f"recall={m['recall']}"
        assert m["f1"] == 1.0, f"f1={m['f1']}"
        assert m["specificity"] == 1.0, f"specificity={m['specificity']}"
        assert m["roc_auc"] == 1.0, f"roc_auc={m['roc_auc']}"

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
        # TP=3, FN=1, FP=2, TN=4  (cloud=1, clear=0)
        # precision = 3/(3+2) = 0.6
        # recall = 3/(3+1) = 0.75
        # f1 = 2*0.6*0.75/(0.6+0.75) = 0.666...
        # specificity = TN/(TN+FP) = 4/(4+2) = 0.666...
        y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        y_score = [0.9, 0.8, 0.7, 0.4, 0.6, 0.55, 0.3, 0.2, 0.15, 0.1]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["confusion_matrix"] == {"tn": 4, "fp": 2, "fn": 1, "tp": 3}
        assert abs(m["precision"] - 0.6) < 1e-5, m["precision"]
        assert abs(m["recall"] - 0.75) < 1e-5, m["recall"]
        expected_f1 = 2 * 0.6 * 0.75 / (0.6 + 0.75)
        assert abs(m["f1"] - expected_f1) < 1e-5, m["f1"]
        assert abs(m["specificity"] - (4 / 6)) < 1e-5, m["specificity"]

    def test_single_class_no_roc_auc(self):
        """When only one class is present roc_auc is None (no crash)."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_score = [0.1, 0.2, 0.15, 0.05]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["roc_auc"] is None
        assert m["n"] == 4
        assert m["n_cloud"] == 0
        assert m["n_clear"] == 4

    def test_all_wrong_zero_precision_recall(self):
        """All predictions inverted → precision=0, recall=0 (zero_division guard)."""
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

    def test_roc_auc_present_when_both_classes(self):
        """roc_auc is a float in [0,1] when both classes are present."""
        records = _make_records(5, 5, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        assert m["roc_auc"] is not None
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_metric_keys_present(self):
        """All expected keys are present in the returned dict."""
        records = _make_records(3, 3, correct=True)
        y_true = [r["label"] for r in records]
        y_pred = [r["pred"] for r in records]
        y_score = [r["score"] for r in records]
        m = compute_classification_metrics(y_true, y_pred, y_score)

        expected_keys = {
            "n",
            "n_cloud",
            "n_clear",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "roc_auc",
            "confusion_matrix",
        }
        assert expected_keys == set(m.keys())
        cm_keys = {"tn", "fp", "fn", "tp"}
        assert cm_keys == set(m["confusion_matrix"].keys())


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
        """Single-profile group of correct predictions → accuracy=1.0."""
        records = _make_records(6, 6, profile="A", correct=True)
        result = group_metrics_by_profile(records)

        assert result["A"]["accuracy"] == 1.0
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
