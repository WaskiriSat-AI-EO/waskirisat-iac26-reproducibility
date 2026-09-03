"""Tests for nina_thermal.evaluation.metrics — torch-free metric helpers."""

import numpy as np
import pytest
from sklearn.metrics import fbeta_score, roc_auc_score

from nina_thermal.evaluation import metrics as M


class TestPrecisionAtPrevalence:
    def test_bayes_projection(self):
        # perfect recall, 90% specificity -> FPR 0.1
        # pi=0.5: 0.5*1 / (0.5*1 + 0.5*0.1) = 1/1.1
        assert M.precision_at_prevalence(1.0, 0.9, 0.5) == pytest.approx(1 / 1.1, abs=1e-6)

    def test_rare_prevalence_drives_precision_down(self):
        hi = M.precision_at_prevalence(0.9, 0.9, 0.9)
        lo = M.precision_at_prevalence(0.9, 0.9, 0.1)
        assert lo < hi

    def test_degenerate_classifier_returns_zero(self):
        assert M.precision_at_prevalence(0.0, 1.0, 0.5) == 0.0


class TestComputeClassificationMetrics:
    def test_confusion_and_headline(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        y_score = [0.9, 0.4, 0.2, 0.7]
        m = M.compute_classification_metrics(y_true, y_pred, y_score, beta=2.0)
        assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
        assert m["recall"] == pytest.approx(0.5)
        assert m["specificity"] == pytest.approx(0.5)
        assert m["n_fire"] == 2 and m["n_nofire"] == 2

    def test_roc_auc_matches_sklearn(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100).tolist()
        s = [float(np.clip(0.5 + (0.2 if v else -0.2) + rng.normal(0, 0.3), 0, 1)) for v in y]
        m = M.compute_classification_metrics(y, [int(x >= 0.5) for x in s], s, beta=1.7)
        assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)

    def test_single_class_nulls_ranking_metrics(self):
        m = M.compute_classification_metrics([1, 1, 1], [1, 1, 0], [0.9, 0.8, 0.3], beta=2.0)
        assert m["roc_auc"] is None and m["average_precision"] is None

    def test_prevalence_block_only_when_requested(self):
        base = M.compute_classification_metrics([1, 0], [1, 0], [0.9, 0.1], beta=2.0)
        assert "precision_at_prevalence" not in base
        withp = M.compute_classification_metrics(
            [1, 0], [1, 0], [0.9, 0.1], beta=2.0, prevalences=(0.1, 0.9)
        )
        assert set(withp["precision_at_prevalence"]) == {"0.1", "0.9"}

    def test_beta_is_threaded_into_fbeta(self):
        y_true, y_pred = [1, 1, 1, 0], [1, 1, 1, 1]
        m = M.compute_classification_metrics(y_true, y_pred, [0.9] * 4, beta=3.0)
        assert m["fbeta"] == pytest.approx(fbeta_score(y_true, y_pred, beta=3.0))


class TestGroupMetrics:
    @pytest.fixture
    def records(self):
        return [
            {"profile": "B", "surface_type": "fire", "label": 1, "pred": 1, "score": 0.9},
            {"profile": "B", "surface_type": "nominal", "label": 0, "pred": 0, "score": 0.1},
            {"profile": "A", "surface_type": "nominal", "label": 0, "pred": 1, "score": 0.8},
            {"profile": "C", "surface_type": "warm_surface", "label": 0, "pred": 0, "score": 0.3},
        ]

    def test_groups_by_profile(self, records):
        out = M.group_metrics(records, "profile", beta=2.0)
        assert set(out) == {"A", "B", "C"}
        assert out["B"]["n"] == 2

    def test_groups_by_surface_type(self, records):
        out = M.group_metrics(records, "surface_type", beta=2.0)
        assert set(out) == {"fire", "nominal", "warm_surface"}


class TestProfileDeltas:
    def test_signed_drop_vs_baseline(self):
        by_profile = {
            "A": {"recall": 0.90, "specificity": 0.80, "roc_auc": 0.85,
                  "average_precision": 0.7, "fbeta": 0.88, "precision": 0.6},
            "B": {"recall": 0.95, "specificity": 0.85, "roc_auc": 0.90,
                  "average_precision": 0.8, "fbeta": 0.92, "precision": 0.7},
            "C": {"recall": 0.80, "specificity": 0.70, "roc_auc": 0.75,
                  "average_precision": 0.6, "fbeta": 0.78, "precision": 0.5},
        }
        deltas = M.profile_deltas(by_profile, baseline="B")
        assert set(deltas) == {"A", "C"}
        assert deltas["A"]["recall"] == pytest.approx(-0.05)
        assert deltas["C"]["roc_auc"] == pytest.approx(-0.15)

    def test_empty_when_baseline_absent(self):
        assert M.profile_deltas({"A": {"recall": 0.9}}, baseline="B") == {}


class TestClusterBootstrapCI:
    def _data(self, n_groups=20, per_group=6, sep=0.35, seed=0):
        rng = np.random.default_rng(seed)
        y_true, y_score, groups = [], [], []
        for g in range(n_groups):
            for _ in range(per_group):
                label = int(rng.random() < 0.4)
                y_true.append(label)
                y_score.append(
                    float(np.clip(0.5 + (sep if label else -sep) + rng.normal(0, 0.2), 0, 1))
                )
                groups.append(f"g{g}")
        return y_true, y_score, groups

    def test_interval_contains_point_estimate(self):
        y_true, y_score, groups = self._data()
        out = M.cluster_bootstrap_ci(
            y_true, y_score, groups, M.roc_auc_fn, n_resamples=300, seed=1
        )
        assert out["n_resamples_used"] > 250
        assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]

    def test_resamples_by_group_not_row(self):
        # A single duplicated group should still be resampled as one unit:
        # with one group the CI collapses to a point.
        y_true, y_score, _ = self._data(n_groups=1, per_group=30)
        out = M.cluster_bootstrap_ci(
            y_true, y_score, ["g0"] * 30, M.roc_auc_fn, n_resamples=100, seed=1
        )
        assert out["ci_low"] == out["ci_high"] == out["point_estimate"]

    def test_no_groups_returns_nulls(self):
        out = M.cluster_bootstrap_ci([], [], [], M.roc_auc_fn)
        assert out["point_estimate"] is None and out["n_resamples_used"] == 0

    def test_recall_fn_at_threshold(self):
        fn = M.recall_fn(0.5)
        yt = np.array([1, 1, 0, 0])
        ys = np.array([0.9, 0.4, 0.6, 0.1])
        assert fn(yt, ys) == pytest.approx(0.5)
        assert fn(np.array([0, 0]), np.array([0.1, 0.2])) is None  # no positives
