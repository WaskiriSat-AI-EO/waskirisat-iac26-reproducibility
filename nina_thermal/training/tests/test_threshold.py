"""Tests for nina_thermal.training.threshold — recall-floor operating point.

Fire detection has an asymmetric cost, so the decision threshold is swept to
the highest value still clearing a fire-recall floor, and ranking metrics
(ROC-AUC / PR-AUC) are reported alongside because a recall-floor sweep alone
cannot tell a real classifier from one that predicts fire everywhere.
"""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from nina_thermal.training import threshold


class TestSelectThreshold:
    def test_picks_highest_threshold_meeting_recall(self):
        # scores: fires at .9/.8/.4, non-fires at .6/.3/.1
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.9, 0.8, 0.4, 0.6, 0.3, 0.1]
        t, degenerate = threshold.select_threshold(y_true, y_score, recall_target=1.0)
        assert not degenerate
        assert t == pytest.approx(0.4)  # must reach down to the lowest fire
        # at a looser floor the sweep can stop higher
        t2, _ = threshold.select_threshold(y_true, y_score, recall_target=0.66)
        assert t2 == pytest.approx(0.8)

    def test_tied_scores_included_together(self):
        y_true = [1, 1, 0]
        y_score = [0.5, 0.5, 0.5]
        t, degenerate = threshold.select_threshold(y_true, y_score, recall_target=1.0)
        assert not degenerate and t == pytest.approx(0.5)

    def test_single_class_is_degenerate(self):
        t, degenerate = threshold.select_threshold([0, 0, 0], [0.1, 0.2, 0.3], 0.9)
        assert degenerate and t == 0.5

    def test_lowest_threshold_always_reaches_full_recall(self):
        # predicting fire for score >= min(score) recalls every fire, so
        # recall_target <= 1.0 is always feasible on a two-class set.
        _, degenerate = threshold.select_threshold(
            [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], recall_target=1.0
        )
        assert not degenerate

    def test_unreachable_recall_is_degenerate(self):
        # recall_target > 1.0 cannot be met by any threshold.
        t, degenerate = threshold.select_threshold(
            [1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], recall_target=1.01
        )
        assert degenerate and t == 0.5


class TestMetricsAtThreshold:
    def test_confusion_and_rates(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.4, 0.6, 0.1]
        m = threshold.metrics_at_threshold(y_true, y_score, threshold=0.5, beta=2.0)
        assert m["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
        assert m["recall"] == pytest.approx(0.5)
        assert m["precision"] == pytest.approx(0.5)
        assert m["specificity"] == pytest.approx(0.5)

    def test_fbeta_favours_recall_when_beta_gt_1(self):
        y_true = [1, 1, 1, 1, 0]
        y_score = [0.9, 0.9, 0.9, 0.9, 0.9]  # predict fire everywhere
        m1 = threshold.metrics_at_threshold(y_true, y_score, 0.5, beta=1.0)
        m2 = threshold.metrics_at_threshold(y_true, y_score, 0.5, beta=3.0)
        assert m2["fbeta"] > m1["fbeta"]  # high recall, modest precision

    def test_empty_predictions_do_not_divide_by_zero(self):
        m = threshold.metrics_at_threshold([1, 0], [0.1, 0.2], threshold=0.9, beta=2.0)
        assert m["precision"] == 0.0 and m["fbeta"] == 0.0


class TestRankingMetrics:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200).tolist()
        s = [
            float(np.clip(0.5 + (0.3 if yi else -0.3) + rng.normal(0, 0.2), 0, 1))
            for yi in y
        ]
        return y, s

    def test_roc_auc_matches_sklearn(self, data):
        y, s = data
        assert threshold.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-6)

    def test_roc_auc_handles_ties(self):
        y = [1, 0, 1, 0]
        s = [0.5, 0.5, 0.5, 0.5]
        assert threshold.roc_auc(y, s) == pytest.approx(0.5)

    def test_average_precision_close_to_sklearn(self, data):
        y, s = data
        # threshold.average_precision is a step-wise integral, not sklearn's
        # interpolation-free estimator, so allow a small tolerance.
        assert threshold.average_precision(y, s) == pytest.approx(
            average_precision_score(y, s), abs=0.03
        )

    def test_single_class_returns_none(self):
        assert threshold.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
        assert threshold.average_precision([0, 0], [0.1, 0.2]) is None


class TestPrecisionAtRecall:
    def test_reports_precision_per_floor(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.9, 0.8, 0.4, 0.6, 0.3, 0.1]
        out = threshold.precision_at_recall(y_true, y_score, [0.66, 1.0])
        assert out["0.66"] == pytest.approx(1.0)  # top-2 are both fire
        assert 0.0 < out["1.0"] <= 1.0

    def test_single_class_all_none(self):
        out = threshold.precision_at_recall([0, 0], [0.1, 0.2], [0.9])
        assert out == {"0.9": None}


class TestEvalSizeCheck:
    def test_17_fires_is_degenerate_although_sweep_is_not(self):
        # The v1 quantization val set: 17 fire / 19 no-fire.  The sweep is
        # well defined (select_threshold says not degenerate), the size check
        # must still flag it.
        y_true = [1] * 17 + [0] * 19
        y_score = [0.9] * 17 + [0.1] * 19
        _, sweep_degenerate = threshold.select_threshold(y_true, y_score, 0.95)
        out = threshold.eval_size_check(y_true, "val")
        assert sweep_degenerate is False
        assert out["degenerate_eval"] is True
        assert (out["n"], out["n_fire"], out["n_nofire"]) == (36, 17, 19)
        assert "17 fire samples" in out["warning"]
        assert "MIN_FIRE_FOR_EVAL=20" in out["warning"]

    def test_at_minimum_is_not_degenerate(self):
        out = threshold.eval_size_check([1] * 20 + [0] * 5, "val")
        assert out["degenerate_eval"] is False
        assert out["warning"] is None

    def test_warning_is_logged(self, caplog):
        with caplog.at_level("WARNING"):
            threshold.eval_size_check([1] * 3 + [0] * 3, "val")
        assert "DEGENERATE EVALUATION" in caplog.text

    def test_evaluation_metrics_share_the_constant(self):
        from nina_thermal.evaluation import metrics

        assert metrics.MIN_FIRE_FOR_EVAL == threshold.MIN_FIRE_FOR_EVAL == 20
