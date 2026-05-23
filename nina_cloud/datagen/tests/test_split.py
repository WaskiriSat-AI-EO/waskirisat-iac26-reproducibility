"""Tests for labeling/split.py."""

import pytest

from nina_cloud.datagen.labeling.split import scene_level_split


def _make_scenes(n_cloud_scenes: int, n_clear_scenes: int, tiles_per_scene: int = 10):
    """Build a synthetic scene → labels dict."""
    data = {}
    for i in range(n_cloud_scenes):
        data[f"cloud_scene_{i:03d}"] = [1] * tiles_per_scene
    for i in range(n_clear_scenes):
        data[f"clear_scene_{i:03d}"] = [0] * tiles_per_scene
    return data


class TestSceneLevelSplit:
    def test_all_scenes_assigned_exactly_once(self):
        data = _make_scenes(20, 20)
        splits = scene_level_split(data, seed=42)
        all_assigned = splits["train"] + splits["val"] + splits["test"]
        assert set(all_assigned) == set(data.keys())
        assert len(all_assigned) == len(data)

    def test_no_scene_in_multiple_splits(self):
        data = _make_scenes(30, 30)
        splits = scene_level_split(data, seed=42)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_fractions_sum_error_raises(self):
        data = _make_scenes(10, 10)
        with pytest.raises(ValueError, match="sum to 1.0"):
            scene_level_split(data, train_frac=0.6, val_frac=0.2, test_frac=0.3)

    def test_deterministic_with_same_seed(self):
        data = _make_scenes(20, 20)
        s1 = scene_level_split(data, seed=7)
        s2 = scene_level_split(data, seed=7)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_different_seeds_give_different_splits(self):
        data = _make_scenes(40, 40)
        s1 = scene_level_split(data, seed=1)
        s2 = scene_level_split(data, seed=2)
        assert s1["train"] != s2["train"]

    def test_both_cloud_tiers_represented_in_each_split(self):
        """High-cloud and low-cloud scenes must appear in all three splits."""
        data = _make_scenes(n_cloud_scenes=30, n_clear_scenes=30)
        splits = scene_level_split(data, seed=42)
        for split_name, sids in splits.items():
            has_cloud = any(s.startswith("cloud") for s in sids)
            has_clear = any(s.startswith("clear") for s in sids)
            assert has_cloud, f"{split_name} has no high-cloud scenes"
            assert has_clear, f"{split_name} has no low-cloud scenes"

    def test_ambiguous_tiles_do_not_affect_stratification(self):
        """Ambiguous labels (-1) must be ignored when computing cloud fraction."""
        data = {
            "scene_A": [1, 1, -1, -1],  # 2 cloud, 2 ambiguous → cloud frac = 1.0
            "scene_B": [0, 0, -1, -1],  # 2 clear, 2 ambiguous → cloud frac = 0.0
        }
        # Should not raise and should still assign both scenes
        splits = scene_level_split(
            data, train_frac=0.5, val_frac=0.25, test_frac=0.25, seed=0
        )
        all_assigned = splits["train"] + splits["val"] + splits["test"]
        assert set(all_assigned) == {"scene_A", "scene_B"}

    def test_approximate_split_sizes(self):
        """With 100 scenes the split sizes should be near 70 / 15 / 15."""
        data = _make_scenes(50, 50)
        splits = scene_level_split(data, seed=0)
        assert 65 <= len(splits["train"]) <= 75
        assert 10 <= len(splits["val"]) <= 20
        assert 10 <= len(splits["test"]) <= 20
