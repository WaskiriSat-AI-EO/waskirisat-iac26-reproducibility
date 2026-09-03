"""Tests for nina_thermal.training.baseline — the feasibility gate.

The fixture's synthetic "fire" patches carry literal 350-400 K pixels, so the
baseline trivially separates them — the *numeric* AUC here is a fixture
artefact, not the real-data finding. These tests assert the mechanism: BT
recovery, feature extraction, grouped-by-granule CV, the degeneracy guard, and
the datagen-vs-FRP label switch.
"""

import numpy as np
import pytest

from nina_thermal.training import baseline
from nina_thermal.training.dataset import load_merged_records


class TestBtFromUint8:
    def test_inverts_the_agc_ramp(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        bt = baseline.bt_from_uint8(img, t_min_k=258.15, t_max_k=358.15)
        assert bt[0, 0] == pytest.approx(258.15)
        assert bt[0, 2] == pytest.approx(358.15)
        assert bt[0, 1] == pytest.approx(258.15 + 128 / 255 * 100, abs=0.01)


class TestPatchStatistics:
    def test_returns_every_feature(self):
        img = np.full((120, 160), 128, dtype=np.uint8)
        stats = baseline.patch_statistics(img, t_min_k=258.15, t_max_k=358.15)
        assert set(stats) == set(baseline.FEATURE_NAMES)

    def test_hotspot_lifts_top_pixel_excess(self):
        flat = np.full((120, 160), 100, dtype=np.uint8)
        hot = flat.copy()
        hot[0, 0] = 255
        s_flat = baseline.patch_statistics(flat, t_min_k=258.15, t_max_k=358.15)
        s_hot = baseline.patch_statistics(hot, t_min_k=258.15, t_max_k=358.15)
        assert s_hot["top_pixel_excess"] > s_flat["top_pixel_excess"]
        assert s_hot["bt_max"] > s_flat["bt_max"]


class TestBuildFeatureMatrix:
    def test_shapes_and_grouping(self, multiroot):
        records, _ = load_merged_records(multiroot.roots)
        X, y, groups = baseline.build_feature_matrix(
            records, agc_mode="fixed", profile="B"
        )
        assert X.shape == (len(records), len(baseline.FEATURE_NAMES))
        assert y.shape == (len(records),)
        assert len(groups) == len(records)
        assert set(groups) <= {r["granule_id"] for r in records}

    def test_absent_variant_yields_empty(self, multiroot):
        records, _ = load_merged_records(multiroot.roots)
        X, y, groups = baseline.build_feature_matrix(
            records, agc_mode="global", profile="B"  # fixture renders fixed only here
        )
        assert X.size == 0 and y.size == 0 and groups == []

    def test_reads_agc_bounds_from_metadata(self, multiroot):
        records, _ = load_merged_records(multiroot.roots)
        X, _, _ = baseline.build_feature_matrix(records, agc_mode="fixed", profile="B")
        # bt_mean should land in the ambient band, not clamp to the AGC edges
        bt_mean_col = X[:, baseline.FEATURE_NAMES.index("bt_mean")]
        assert 270.0 < float(np.median(bt_mean_col)) < 340.0


class TestGroupedCvAuc:
    def test_clamps_folds_to_granule_count(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, len(baseline.FEATURE_NAMES)))
        y = np.array([0, 1] * 15)
        groups = [f"g{i // 10}" for i in range(30)]  # 3 granules
        out = baseline.grouped_cv_auc(X, y, groups, n_splits=5)
        assert out["n_splits"] == 3

    def test_single_class_is_reported_not_raised(self):
        X = np.zeros((10, len(baseline.FEATURE_NAMES)))
        out = baseline.grouped_cv_auc(X, np.zeros(10, dtype=int), [f"g{i}" for i in range(10)])
        assert out["auc_mean"] is None and "one class" in out["error"].lower()

    def test_separable_data_scores_high(self):
        rng = np.random.default_rng(0)
        n = 60
        y = np.array([i % 2 for i in range(n)])
        X = rng.normal(size=(n, len(baseline.FEATURE_NAMES)))
        X[:, 0] += y * 6.0  # a strongly separating feature
        groups = [f"g{i // 4}" for i in range(n)]
        out = baseline.grouped_cv_auc(X, y, groups, n_splits=5)
        assert out["auc_mean"] > 0.9


class TestDetectDegenerateLabels:
    def test_all_positive_flagged(self):
        out = baseline.detect_degenerate_labels(
            np.ones(20, dtype=int), [f"g{i // 4}" for i in range(20)]
        )
        assert out["degenerate"] is True
        assert out["all_positive_granules"] == out["n_granules"]
        assert any("near-constant positive" in w for w in out["warnings"])

    def test_balanced_not_flagged(self):
        y = np.array([0, 1] * 20)
        out = baseline.detect_degenerate_labels(y, [f"g{i // 4}" for i in range(40)])
        assert out["degenerate"] is False

    def test_empty_is_degenerate(self):
        out = baseline.detect_degenerate_labels(np.array([], dtype=int), [])
        assert out["degenerate"] is True and out["n_granules"] == 0


class TestRecalibrateMinFireFraction:
    def test_recommends_the_target_quantile(self, tmp_path):
        import json

        manifest = {
            "patches": [{"fire_fraction": f / 100} for f in range(100)]
        }
        path = tmp_path / "patches_manifest.json"
        path.write_text(json.dumps(manifest))
        out = baseline.recalibrate_min_fire_fraction(path, target_positive_rate=0.20)
        # 20% positive rate -> the 80th percentile of [0.00 .. 0.99]
        assert out["recommended_min_fire_fraction"] == pytest.approx(0.792, abs=0.01)

    def test_empty_manifest_raises(self, tmp_path):
        import json

        path = tmp_path / "m.json"
        path.write_text(json.dumps({"patches": []}))
        with pytest.raises(ValueError, match="fire_fraction"):
            baseline.recalibrate_min_fire_fraction(path)


class TestBuildBaselineReport:
    def test_datagen_scheme_keeps_all_trainval_records(self, multiroot):
        rep = baseline.build_baseline_report(
            [str(r) for r in multiroot.roots], label_scheme="datagen"
        )
        assert rep["label_scheme"] == "datagen"
        assert rep["frp_relabel"] is None
        assert rep["frp_threshold_mw"] is None
        assert set(rep["grouped_cv"]) == {
            "n_splits", "auc_per_fold", "auc_mean", "auc_std", "error"
        }
        assert set(rep["per_feature_auc"]) == set(baseline.FEATURE_NAMES)

    def test_frp_scheme_drops_the_ambiguous_band(self, multiroot):
        datagen = baseline.build_baseline_report(
            [str(r) for r in multiroot.roots], label_scheme="datagen"
        )
        frp = baseline.build_baseline_report(
            [str(r) for r in multiroot.roots], label_scheme="frp"
        )
        assert frp["frp_relabel"]["counts"]["ambiguous"] > 0
        assert frp["n_records_used"] < datagen["n_records_used"]

    def test_never_reads_the_test_split(self, multiroot):
        records, _ = load_merged_records(multiroot.roots)
        test_granules = {r["granule_id"] for r in records if r["split"] == "test"}
        rep = baseline.build_baseline_report(
            [str(r) for r in multiroot.roots], label_scheme="datagen"
        )
        # the report exposes granule count; assert it excludes the test granules
        trainval = {r["granule_id"] for r in records if r["split"] in ("train", "val")}
        assert rep["n_granules_used"] == len(trainval - test_granules) or (
            rep["n_granules_used"] <= len(trainval)
        )
