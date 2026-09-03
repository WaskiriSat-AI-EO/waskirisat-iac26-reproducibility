"""Tests for nina_thermal.training.frp_labels — the FRP relabel.

The datagen fire label is not learnable at the simulated GSD; frp_labels
reformulates the target as "large fire complex in view" (summed FRP over the
footprint) and drops the ambiguous middle band rather than mislabelling it.
"""


import numpy as np
import pytest

from nina_thermal.training import frp_labels


class TestGranuleDate:
    def test_parses_slstr_timestamp(self):
        d = frp_labels._granule_date("S3A_SL_1_RBT____20230801T023011_x_y_z")
        assert d == np.datetime64("2023-08-01")

    def test_unparseable_id_raises(self):
        with pytest.raises(ValueError, match="[Cc]annot parse"):
            frp_labels._granule_date("not-a-granule-id")


def _detections(rows):
    import pandas as pd

    return pd.DataFrame(
        rows, columns=["latitude", "longitude", "acq_date", "confidence", "frp"]
    ).astype({"acq_date": "datetime64[ns]"})


def _sample(patch, granule_id, meta_dir, centroid, date, *, label=0):
    """Write a metadata JSON and return the matching sample dict."""
    import json

    (meta_dir).mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{patch}_fixed_B.json").write_text(
        json.dumps({"centroid_lat": centroid[0], "centroid_lon": centroid[1]})
    )
    return {
        "patch": patch,
        "granule_id": granule_id,
        "label": label,
        "path": meta_dir.parent / "images" / f"{patch}_fixed_B.png",
    }


class TestComputePatchFrp:
    def test_sums_in_footprint_and_time_window(self, tmp_path):
        gid = "S3A_SL_1_RBT____20230815T120000_g001"
        s = _sample("g001_p0000", gid, tmp_path / "images_meta_root/metadata",
                    (-12.0, -63.0), None)
        # 2 rows in-window at the centroid, 1 row a week away (excluded).
        det = _detections(
            [
                (-12.0, -63.0, "2023-08-15", "nominal", 6000.0),
                (-12.1, -63.1, "2023-08-16", "nominal", 6000.0),
                (-12.0, -63.0, "2023-08-25", "nominal", 9000.0),
            ]
        )
        out = frp_labels.compute_patch_frp([s], det)
        assert out["g001_p0000"]["frp_sum"] == pytest.approx(12000.0)
        assert out["g001_p0000"]["n_detections"] == 2

    def test_detection_outside_box_excluded(self, tmp_path):
        gid = "S3A_SL_1_RBT____20230815T120000_g002"
        s = _sample("g002_p0000", gid, tmp_path / "r/metadata", (-12.0, -63.0), None)
        det = _detections([(-40.0, -63.0, "2023-08-15", "nominal", 50000.0)])
        out = frp_labels.compute_patch_frp([s], det)
        assert out["g002_p0000"]["frp_sum"] == 0.0
        assert out["g002_p0000"]["n_detections"] == 0

    def test_missing_metadata_is_skipped(self, tmp_path):
        s = {
            "patch": "ghost_p0",
            "granule_id": "S3A_SL_1_RBT____20230815T120000_g9",
            "label": 0,
            "path": tmp_path / "images" / "ghost_p0_fixed_B.png",
        }
        out = frp_labels.compute_patch_frp([s], _detections([]))
        assert "ghost_p0" not in out


class TestAssignFrpLabels:
    def _patch_frp(self):
        return {
            "p_fire": {"frp_sum": 25000.0, "frp_max": 20000.0, "n_detections": 4},
            "p_clean": {"frp_sum": 0.0, "frp_max": 0.0, "n_detections": 0},
            "p_ambig": {"frp_sum": 5000.0, "frp_max": 5000.0, "n_detections": 1},
        }

    def test_three_class_partition(self):
        samples = [
            {"patch": "p_fire", "label": 1},
            {"patch": "p_clean", "label": 0},
            {"patch": "p_ambig", "label": 1},
        ]
        relabelled, summary = frp_labels.assign_frp_labels(
            samples, self._patch_frp(), threshold_mw=20000.0
        )
        by_patch = {r["patch"]: r["label"] for r in relabelled}
        assert by_patch["p_fire"] == 1
        assert by_patch["p_clean"] == 0
        assert by_patch["p_ambig"] == frp_labels.LABEL_AMBIGUOUS
        assert summary["counts"] == {
            "fire": 1,
            "no_fire": 1,
            "ambiguous": 1,
            "no_frp_data": 0,
        }
        assert summary["retained"] == 2

    def test_threshold_is_inclusive(self):
        pf = {"p": {"frp_sum": 20000.0, "frp_max": 20000.0, "n_detections": 2}}
        relabelled, _ = frp_labels.assign_frp_labels(
            [{"patch": "p", "label": 0}], pf, threshold_mw=20000.0
        )
        assert relabelled[0]["label"] == 1

    def test_original_label_is_preserved(self):
        relabelled, _ = frp_labels.assign_frp_labels(
            [{"patch": "p_fire", "label": 0}], self._patch_frp(), threshold_mw=20000.0
        )
        assert relabelled[0]["original_label"] == 0

    def test_sample_without_frp_data_counted_not_kept(self):
        relabelled, summary = frp_labels.assign_frp_labels(
            [{"patch": "unknown", "label": 1}], self._patch_frp()
        )
        assert relabelled == []
        assert summary["counts"]["no_frp_data"] == 1


class TestApplyFrpLabelsEndToEnd:
    def test_runs_on_the_multiroot_fixture(self, multiroot):
        from nina_thermal.training.dataset import expand_to_samples, load_merged_records

        records, _ = load_merged_records(multiroot.roots)
        samples = expand_to_samples(records, ["B"], ["fixed"])
        kept, summary = frp_labels.apply_frp_labels(
            samples, multiroot.roots, threshold_mw=20000.0
        )

        assert summary["counts"]["fire"] > 0
        assert summary["counts"]["no_fire"] > 0
        assert summary["counts"]["ambiguous"] > 0
        assert all(s["label"] in (0, 1) for s in kept)
        assert len(kept) == summary["retained"]

    def test_missing_firms_csv_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="No FIRMS detections"):
            frp_labels.load_firms_detections([tmp_path])
