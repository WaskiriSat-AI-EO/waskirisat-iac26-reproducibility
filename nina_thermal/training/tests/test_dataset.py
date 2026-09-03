"""Tests for nina_thermal.training.dataset — multi-root merge + leakage guard.

The two failure modes this module protects against (from the dataset.py
docstring): duplicate patches re-weighting the training set, and per-root
split disagreement leaking evaluation granules into training.
"""

import json

import pytest

from nina_thermal.training import dataset


class TestReconcileSplits:
    def test_precedence_test_over_val_over_train(self):
        resolved = dataset.reconcile_splits(
            {
                "g_tt": {"train", "test"},
                "g_tv": {"train", "val"},
                "g_vt": {"val", "test"},
                "g_t": {"train"},
            }
        )
        assert resolved == {
            "g_tt": "test",
            "g_tv": "val",
            "g_vt": "test",
            "g_t": "train",
        }

    def test_unknown_split_name_raises(self):
        with pytest.raises(ValueError, match="unknown split"):
            dataset.reconcile_splits({"g": {"train", "holdout"}})

    def test_single_split_is_identity(self):
        assert dataset.reconcile_splits({"g": {"val"}}) == {"g": "val"}


class TestResolveDatasetRoot:
    def test_accepts_root_and_dataset_dir(self, multiroot):
        root = multiroot.roots[0]
        assert dataset.resolve_dataset_root(root) == root / "04_dataset"
        assert dataset.resolve_dataset_root(root / "04_dataset") == root / "04_dataset"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="balanced_manifest"):
            dataset.resolve_dataset_root(tmp_path)


class TestLoadMergedRecords:
    def test_dedupes_and_reconciles(self, multiroot):
        records, stats = dataset.load_merged_records(multiroot.roots)
        patches = [r["patch"] for r in records]
        assert len(patches) == len(set(patches)), "duplicate patch survived the merge"
        assert stats["unique_patches"] == len(records)
        assert stats["duplicate_patches_dropped"] > 0
        assert stats["granules_with_conflicting_splits"] == len(
            multiroot.conflicting_granules
        )

    def test_reconciled_split_matches_precedence(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        by_granule_votes: dict[str, set[str]] = {}
        for root in multiroot.roots:
            sm = json.loads((root / "04_dataset" / "split_manifest.json").read_text())
            for split_name in ("train", "val", "test"):
                for gid in sm[split_name]:
                    by_granule_votes.setdefault(gid, set()).add(split_name)
        expected = dataset.reconcile_splits(by_granule_votes)
        for rec in records:
            assert rec["split"] == expected[rec["granule_id"]]

    def test_ambiguous_records_are_dropped(self, multiroot, tmp_path):
        # Inject a label == -1 record into a copied manifest and confirm it is filtered.
        src = multiroot.roots[0] / "04_dataset" / "balanced_manifest.json"
        body = json.loads(src.read_text())
        body["records"].append(
            {**body["records"][0], "patch": "ghost_p9999", "label": -1}
        )
        dst = tmp_path / "04_dataset"
        dst.mkdir()
        (dst / "balanced_manifest.json").write_text(json.dumps(body))
        records, _ = dataset.load_merged_records([dst])
        assert all(r["label"] != -1 for r in records)
        assert not any(r["patch"] == "ghost_p9999" for r in records)

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dataset.load_merged_records([tmp_path])


class TestExpandToSamples:
    def test_one_sample_per_existing_image(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        samples = dataset.expand_to_samples(records, ["B"], ["fixed"])
        assert len(samples) == len(records)  # 1 profile x 1 agc mode
        s = samples[0]
        assert {"path", "label", "granule_id", "patch", "split", "profile"} <= set(s)
        assert s["path"].is_file()

    def test_absent_variant_is_skipped_not_fatal(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        # Profile "A" is rendered by the fixture, "C" too; ask for a bogus agc mode.
        with pytest.raises(RuntimeError, match="No images found"):
            dataset.expand_to_samples(records, ["B"], ["nonexistent_mode"])

    def test_partial_variant_degrades_gracefully(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        img = next(
            (records[0]["_root"] / "images").glob(f"{records[0]['patch']}_fixed_B.png")
        )
        img.unlink()
        samples = dataset.expand_to_samples(records, ["B"], ["fixed"])
        assert len(samples) == len(records) - 1


class TestSplitSamples:
    def test_groups_by_split_without_leakage(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        samples = dataset.expand_to_samples(records, ["B"], ["fixed"])
        by_split = dataset.split_samples(samples)
        granules = {k: {s["granule_id"] for s in v} for k, v in by_split.items()}
        assert not (granules.get("train", set()) & granules.get("test", set()))
        assert not (granules.get("train", set()) & granules.get("val", set()))
        assert not (granules.get("val", set()) & granules.get("test", set()))

    def test_injected_leakage_is_caught(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        samples = dataset.expand_to_samples(records, ["B"], ["fixed"])
        # Move ONE sample of a train granule into test, so that granule now
        # straddles both splits — exactly the reconciliation-broke condition.
        train_gid = next(s["granule_id"] for s in samples if s["split"] == "train")
        poisoned, flipped = [], False
        for s in samples:
            if s["granule_id"] == train_gid and not flipped:
                poisoned.append({**s, "split": "test"})
                flipped = True
            else:
                poisoned.append(s)
        with pytest.raises(RuntimeError, match="[Ll]eakage"):
            dataset.split_samples(poisoned)


class TestDescribeSamples:
    def test_counts(self, multiroot):
        records, _ = dataset.load_merged_records(multiroot.roots)
        samples = dataset.expand_to_samples(records, ["B"], ["fixed"])
        desc = dataset.describe_samples(samples)
        assert desc["images"] == len(samples)
        assert desc["fire"] + desc["no_fire"] == len(samples)
        assert desc["patches"] <= desc["images"]
