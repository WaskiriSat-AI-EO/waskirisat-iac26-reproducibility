"""Tests for evaluate.int8_metrics_block — int8 test uncertainty (task A2).

The int8 block must carry the same uncertainty as the fp32 block: a 90%
granule-level cluster bootstrap with 1000 resamples, and the Bayes-projected
precision at prevalences 0.10 / 0.50 / 0.90.  No TFLite is needed: the block
takes the int8 scores as input.
"""

import numpy as np
import pytest

from nina_thermal.evaluation import evaluate
from nina_thermal.evaluation import metrics as M

BETA = float(np.sqrt(3.0))


def _predictions(n_granules=12, per_granule=6, seed=0):
    """Synthetic profile-B records: 6 patches per granule, mixed labels."""
    rng = np.random.default_rng(seed)
    recs = []
    for g in range(n_granules):
        for i in range(per_granule):
            label = int((g + i) % 2 == 0)
            recs.append(
                {
                    "label": label,
                    "profile": "B",
                    "granule_id": f"g{g:02d}",
                    "patch": f"g{g:02d}_p{i}",
                    "score": float(np.clip(0.6 * label + rng.normal(0.2, 0.2), 0, 1)),
                }
            )
    return recs


def _fp32_nominal(recs, threshold=0.5):
    return M.compute_classification_metrics(
        [r["label"] for r in recs],
        [int(r["score"] >= threshold) for r in recs],
        [r["score"] for r in recs],
        beta=BETA,
        prevalences=M.EVAL_PREVALENCES,
    )


@pytest.fixture
def block():
    recs = _predictions()
    rng = np.random.default_rng(1)
    int8 = [float(np.clip(r["score"] + rng.normal(0, 0.05), 0, 1)) for r in recs]
    return recs, int8, evaluate.int8_metrics_block(
        recs, int8, 0.45, _fp32_nominal(recs), BETA, seed=7
    )


class TestBootstrap:
    def test_same_four_ci_metrics_as_fp32(self, block):
        _, _, out = block
        assert set(out["nominal"]["bootstrap_ci"]) == {
            "recall",
            "precision",
            "fbeta",
            "roc_auc",
        }

    def test_1000_resamples_at_90_percent(self, block):
        recs, int8, out = block
        ci = out["nominal"]["bootstrap_ci"]["roc_auc"]
        assert ci["n_resamples_requested"] == M.BOOTSTRAP_N_RESAMPLES == 1000
        assert M.BOOTSTRAP_CI == 0.90
        # Reproduce the interval: 5th / 95th percentile of granule resamples.
        ref = M.cluster_bootstrap_ci(
            [r["label"] for r in recs], int8, [r["granule_id"] for r in recs],
            M.roc_auc_fn, seed=7,
        )
        assert (ci["ci_low"], ci["ci_high"]) == (ref["ci_low"], ref["ci_high"])

    def test_interval_brackets_point_estimate(self, block):
        _, _, out = block
        for name, ci in out["nominal"]["bootstrap_ci"].items():
            assert ci["ci_low"] <= ci["point_estimate"] <= ci["ci_high"], name
        assert out["nominal"]["bootstrap_ci"]["roc_auc"]["point_estimate"] == (
            pytest.approx(out["nominal"]["roc_auc"], abs=1e-6)
        )

    def test_resamples_granules_not_patches(self):
        # All patches of a granule share one label: a granule bootstrap of
        # recall sees whole granules appear or vanish, so with 2 fire granules
        # out of 4 recall can only be 0, 0.5 or 1 at this threshold.
        recs = [
            {"label": int(g < 2), "profile": "B", "granule_id": f"g{g}",
             "patch": f"g{g}_{i}", "score": 0.0}
            for g in range(4) for i in range(5)
        ]
        int8 = [0.9 if r["granule_id"] == "g0" else 0.1 for r in recs]
        out = evaluate.int8_metrics_block(
            recs, int8, 0.5, _fp32_nominal(recs), BETA, seed=0
        )
        ci = out["nominal"]["bootstrap_ci"]["recall"]
        assert ci["point_estimate"] == 0.5
        assert {ci["ci_low"], ci["ci_high"]} <= {0.0, 0.5, 1.0}


class TestPrecisionAtPrevalence:
    def test_three_prevalences(self, block):
        _, _, out = block
        assert set(out["nominal"]["precision_at_prevalence"]) == {"0.1", "0.5", "0.9"}

    def test_matches_bayes_formula(self, block):
        _, _, out = block
        nom = out["nominal"]
        tpr, fpr = nom["recall"], 1.0 - nom["specificity"]
        for pi in (0.1, 0.5, 0.9):
            expected = pi * tpr / (pi * tpr + (1 - pi) * fpr)
            assert nom["precision_at_prevalence"][str(pi)] == pytest.approx(
                expected, abs=1e-6
            )

    def test_per_profile_blocks_carry_it_too(self, block):
        _, _, out = block
        assert "precision_at_prevalence" in out["by_profile"]["B"]


class TestRecordsAndDeltas:
    def test_predictions_get_int8_columns(self, block):
        recs, int8, out = block
        assert [r["int8_score"] for r in recs] == [round(s, 6) for s in int8]
        assert all(r["int8_pred"] == int(s >= 0.45) for r, s in zip(recs, int8))
        # fp32 columns untouched
        assert "score" in recs[0] and recs[0]["score"] != recs[0]["int8_score"]

    def test_deltas_are_int8_minus_fp32(self, block):
        recs, _, out = block
        fp32 = _fp32_nominal(recs)
        assert out["deltas_vs_fp32_nominal"]["recall"] == pytest.approx(
            out["nominal"]["recall"] - fp32["recall"], abs=1e-6
        )

    def test_threshold_recorded(self, block):
        assert block[2]["operating_threshold"] == 0.45

    def test_misaligned_scores_raise(self):
        recs = _predictions(n_granules=2)
        with pytest.raises(ValueError):
            evaluate.int8_metrics_block(recs, [0.5], 0.5, _fp32_nominal(recs), BETA, 0)
