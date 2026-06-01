"""Tests for registry-based split assignment.

Covers both the pure hash function (assign_split) and the manifest-stamping
helper (assign_splits_from_registry), including the stability guarantee that
is the primary reason for switching from the old stratified split.
"""

from shared.scene_registry import assign_split, register_scene, split_for
from nina_cloud.datagen.labeling.split import assign_splits_from_registry


# ── assign_split (pure hash) ──────────────────────────────────────────────────


class TestAssignSplit:
    def test_returns_valid_split(self):
        """Every scene_id maps to one of the three valid split names."""
        for i in range(200):
            result = assign_split(f"S2A_MSIL2A_20230601T120000_{i:04d}")
            assert result in ("train", "val", "test"), f"Unexpected split: {result!r}"

    def test_deterministic(self):
        """Same scene_id always maps to the same split — no RNG involved."""
        scene_id = "S2A_MSIL2A_20230601T120000_T19LGJ_R001_20230601T123456"
        results = {assign_split(scene_id) for _ in range(10)}
        assert len(results) == 1, "assign_split is not deterministic"

    def test_process_independent(self):
        """Result is independent of call order (no accumulated state)."""
        scenes = [f"SCENE_{i:04d}" for i in range(50)]
        forward = [assign_split(s) for s in scenes]
        backward = [assign_split(s) for s in reversed(scenes)]
        assert forward == list(reversed(backward))

    def test_approximate_70_15_15_distribution(self):
        """With 1000 scenes the empirical distribution should be near 70/15/15."""
        scenes = [f"S2X_MSIL2A_2023{i:06d}" for i in range(1000)]
        counts = {"train": 0, "val": 0, "test": 0}
        for s in scenes:
            counts[assign_split(s)] += 1
        assert 650 <= counts["train"] <= 750, f"train={counts['train']} not near 700"
        assert 100 <= counts["val"] <= 200, f"val={counts['val']} not near 150"
        assert 100 <= counts["test"] <= 200, f"test={counts['test']} not near 150"

    def test_custom_cut_points(self):
        """Custom train_cut / val_cut are respected."""
        # 50/25/25 split
        scenes = [f"SCENE_{i:05d}" for i in range(1000)]
        counts = {"train": 0, "val": 0, "test": 0}
        for s in scenes:
            counts[assign_split(s, train_cut=50, val_cut=75)] += 1
        assert 450 <= counts["train"] <= 550
        assert 200 <= counts["val"] <= 300
        assert 200 <= counts["test"] <= 300

    def test_stable_when_dataset_grows(self):
        """Adding new scenes never changes the split of existing scenes."""
        initial = [f"SCENE_{i:04d}" for i in range(20)]
        initial_splits = {s: assign_split(s) for s in initial}

        additional = [f"SCENE_{i:04d}" for i in range(20, 100)]
        for s in additional:
            assign_split(s)  # simulate growing the dataset

        # Original scenes must still map to the same split
        for s, expected in initial_splits.items():
            assert assign_split(s) == expected, (
                f"Split for {s} changed from {expected!r} to {assign_split(s)!r} "
                "after adding more scenes — hash is not stable"
            )


# ── split_for (registry read-through) ────────────────────────────────────────


class TestSplitFor:
    def test_returns_registry_value(self):
        registry: dict = {}
        register_scene(registry, "SCENE_001", cloud_mode="clear", operator="test")
        recorded = registry["SCENE_001"]["split"]
        assert split_for(registry, "SCENE_001") == recorded

    def test_fallback_when_not_registered(self):
        """Scenes absent from the registry fall back to assign_split — no crash."""
        registry: dict = {}
        result = split_for(registry, "UNREGISTERED_SCENE")
        assert result in ("train", "val", "test")
        # Result must equal what assign_split would return directly
        assert result == assign_split("UNREGISTERED_SCENE")

    def test_write_once_split(self):
        """Calling register_scene twice never changes the split field."""
        registry: dict = {}
        register_scene(registry, "SCENE_X", cloud_mode="clear", operator="alice")
        first_split = registry["SCENE_X"]["split"]

        # Second call (different operator, different cloud_mode) — split unchanged
        register_scene(registry, "SCENE_X", cloud_mode="cloudy", operator="bob")
        assert (
            registry["SCENE_X"]["split"] == first_split
        ), "register_scene must never overwrite an existing split"


# ── assign_splits_from_registry ───────────────────────────────────────────────


class TestAssignSplitsFromRegistry:
    def _make_records(self, scene_ids):
        records = []
        for sid in scene_ids:
            records.append({"scene_id": sid, "label": 0, "profile": "B", "split": ""})
        return records

    def test_all_records_get_valid_split(self):
        registry: dict = {}
        scene_ids = [f"SCENE_{i:03d}" for i in range(30)]
        for sid in scene_ids:
            register_scene(registry, sid, cloud_mode="clear", operator="test")
        records = self._make_records(scene_ids)
        assign_splits_from_registry(records, registry)
        for rec in records:
            assert rec["split"] in (
                "train",
                "val",
                "test",
            ), f"Invalid split {rec['split']!r} for {rec['scene_id']}"

    def test_scene_level_consistency(self):
        """All records for the same scene_id get the same split."""
        registry: dict = {}
        scene_ids = [f"SCENE_{i:03d}" for i in range(10)]
        for sid in scene_ids:
            register_scene(registry, sid, cloud_mode="clear", operator="test")

        # 3 records per scene (3 profiles)
        records = []
        for sid in scene_ids:
            for profile in ("A", "B", "C"):
                records.append(
                    {"scene_id": sid, "label": 0, "profile": profile, "split": ""}
                )

        assign_splits_from_registry(records, registry)

        scene_splits: dict = {}
        for rec in records:
            sid = rec["scene_id"]
            if sid in scene_splits:
                assert (
                    scene_splits[sid] == rec["split"]
                ), f"Scene {sid} has inconsistent splits across profiles"
            scene_splits[sid] = rec["split"]

    def test_stable_across_incremental_runs(self):
        """Splits assigned in run 1 are unchanged after run 2 adds more scenes."""
        registry: dict = {}
        initial_ids = [f"SCENE_{i:03d}" for i in range(20)]
        for sid in initial_ids:
            register_scene(registry, sid, cloud_mode="clear", operator="test")

        records_run1 = self._make_records(initial_ids)
        assign_splits_from_registry(records_run1, registry)
        splits_run1 = {rec["scene_id"]: rec["split"] for rec in records_run1}

        # Add more scenes (simulating a second operator run)
        for i in range(20, 50):
            register_scene(
                registry, f"SCENE_{i:03d}", cloud_mode="cloudy", operator="op2"
            )

        records_run2 = self._make_records(initial_ids)
        assign_splits_from_registry(records_run2, registry)
        splits_run2 = {rec["scene_id"]: rec["split"] for rec in records_run2}

        assert splits_run1 == splits_run2, (
            "Split assignments changed after adding more scenes — "
            "hash-based split must be stable"
        )

    def test_empty_registry_falls_back_gracefully(self):
        """An empty registry triggers assign_split fallback without crashing."""
        records = self._make_records([f"SCENE_{i:03d}" for i in range(10)])
        assign_splits_from_registry(records, {})
        for rec in records:
            assert rec["split"] in ("train", "val", "test")
