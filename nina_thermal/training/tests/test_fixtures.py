"""Tests for the synthetic dataset generator itself.

If the fixture is wrong, every downstream test is meaningless, so this module
checks the fixture produces the on-disk schema the real datagen pipeline does
and the specific pathologies :mod:`nina_thermal.training.dataset` /
:mod:`nina_thermal.training.frp_labels` exist to handle.
"""

import json

import pytest

from nina_thermal.training import fixtures


class TestSingleRoot:
    def test_manifests_written(self, single_root):
        dataset_dir, _ = single_root
        for name in (
            "label_manifest.json",
            "split_manifest.json",
            "balanced_manifest.json",
        ):
            assert (dataset_dir / name).is_file(), name

    def test_balanced_manifest_schema(self, single_root):
        dataset_dir, _ = single_root
        body = json.loads((dataset_dir / "balanced_manifest.json").read_text())
        assert set(body) >= {"records", "fire_ratio", "seed"}
        rec = body["records"][0]
        assert {"patch", "label", "granule_id", "split", "surface_type"} <= set(rec)

    def test_every_record_has_an_image_per_variant(self, single_root):
        dataset_dir, _ = single_root
        records = json.loads((dataset_dir / "label_manifest.json").read_text())
        for rec in records:
            stem = fixtures._image_stem(rec["patch"], "fixed", "B")
            assert (dataset_dir / "images" / f"{stem}.png").is_file()
            assert (dataset_dir / "metadata" / f"{stem}.json").is_file()

    def test_splits_are_granule_disjoint(self, single_root):
        dataset_dir, _ = single_root
        sm = json.loads((dataset_dir / "split_manifest.json").read_text())
        train, val, test = set(sm["train"]), set(sm["val"]), set(sm["test"])
        assert not (train & val) and not (train & test) and not (val & test)

    def test_summary_counts_consistent(self, single_root):
        _, summary = single_root
        assert summary["fire"] + summary["warm_surface"] + summary["nominal"] == (
            summary["patches"]
        )

    def test_explicit_granule_ids_are_honoured(self, tmp_path):
        ids = (
            "SIM_S3A_SL_1_RBT____20230802T120000_g000",
            "SIM_S3A_SL_1_RBT____20230903T120000_g001",
        )
        spec = fixtures.FixtureSpec(
            patches_per_granule=4,
            agc_modes=("fixed",),
            profiles=("B",),
            granule_ids=ids,
        )
        fixtures.generate_synthetic_dataset(tmp_path / "04_dataset", spec)
        records = json.loads(
            (tmp_path / "04_dataset" / "label_manifest.json").read_text()
        )
        assert {r["granule_id"] for r in records} == set(ids)

    def test_warm_surface_below_ambient_threshold_raises(self):
        with pytest.raises(ValueError, match="WARM_SURFACE_THRESH_K"):
            fixtures.FixtureSpec(ambient_mean_k=320.0, ambient_std_k=10.0)


class TestMultiRoot:
    def test_three_roots_each_with_dataset_and_firms(self, multiroot):
        assert len(multiroot.roots) == 3
        for root in multiroot.roots:
            assert (root / "04_dataset" / "balanced_manifest.json").is_file()
            assert (root / "firms" / "detections.csv").is_file()

    def test_roots_actually_overlap(self, multiroot):
        assert len(multiroot.shared_granules) > 0

    def test_some_shared_granules_have_conflicting_splits(self, multiroot):
        # The whole point of load_merged_records.reconcile_splits.
        assert len(multiroot.conflicting_granules) > 0
        assert set(multiroot.conflicting_granules) <= set(multiroot.shared_granules)

    def test_firms_csv_has_the_columns_frp_labels_reads(self, multiroot):
        import csv

        with open(multiroot.roots[0] / "firms" / "detections.csv") as fh:
            header = next(csv.reader(fh))
        assert {"latitude", "longitude", "acq_date", "confidence", "frp"} <= set(header)

    def test_frp_class_is_a_pure_function_of_granule_id(self, multiroot):
        for gid in multiroot.shared_granules:
            assert fixtures.frp_granule_class(gid) == fixtures.frp_granule_class(gid)

    def test_off_season_granules_are_always_clean(self, multiroot):
        for root in multiroot.roots:
            sm = json.loads((root / "04_dataset" / "split_manifest.json").read_text())
            for gid in sm["train"] + sm["val"] + sm["test"]:
                if fixtures._is_off_season(gid):
                    assert fixtures.frp_granule_class(gid) == "clean"
