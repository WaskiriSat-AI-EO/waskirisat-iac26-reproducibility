"""Tests for dataset/balance.py."""


from nina_cloud.datagen.dataset.balance import build_balanced_dataset


def _make_records(n_cloud: int, n_clear: int, n_ambig: int = 0):
    records = []
    for i in range(n_cloud):
        records.append({"label": 1, "scene_id": f"sc_{i}", "tile_idx": i})
    for i in range(n_clear):
        records.append({"label": 0, "scene_id": f"sc_clear_{i}", "tile_idx": i})
    for i in range(n_ambig):
        records.append({"label": -1, "scene_id": f"sc_ambig_{i}", "tile_idx": i})
    return records


class TestBuildBalancedDataset:
    def test_cloud_tiles_never_removed(self):
        records = _make_records(50, 200)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=42)
        n_cloud = sum(1 for r in balanced if r["label"] == 1)
        assert n_cloud == 50

    def test_achieved_cloud_fraction_near_target(self):
        records = _make_records(100, 1000)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=42)
        n_cloud = sum(1 for r in balanced if r["label"] == 1)
        total = len(balanced)
        achieved = n_cloud / total
        assert abs(achieved - 0.5) < 0.01

    def test_deterministic_with_same_seed(self):
        records = _make_records(50, 200)
        b1 = build_balanced_dataset(records, seed=7)
        b2 = build_balanced_dataset(records, seed=7)
        assert [r["scene_id"] for r in b1] == [r["scene_id"] for r in b2]

    def test_different_seeds_give_different_results(self):
        records = _make_records(50, 500)
        b1 = build_balanced_dataset(records, seed=1)
        b2 = build_balanced_dataset(records, seed=2)
        assert [r["scene_id"] for r in b1] != [r["scene_id"] for r in b2]

    def test_ambiguous_tiles_excluded(self):
        records = _make_records(n_cloud=20, n_clear=20, n_ambig=10)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=0)
        labels = [r["label"] for r in balanced]
        assert -1 not in labels

    def test_no_cloud_tiles_returns_all_clear(self):
        records = _make_records(n_cloud=0, n_clear=50)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=0)
        assert len(balanced) == 50
        assert all(r["label"] == 0 for r in balanced)

    def test_output_is_shuffled(self):
        """Output should not be trivially ordered (all cloud then all clear)."""
        records = _make_records(n_cloud=100, n_clear=100)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=42)
        labels = [r["label"] for r in balanced]
        # If perfectly shuffled, it should not be sorted
        assert labels != sorted(
            labels, reverse=True
        ), "Output appears unshuffled (all cloud first)"

    def test_clear_cannot_exceed_cloud_when_supply_is_limited(self):
        """When there are fewer clear than needed, use all available clear."""
        records = _make_records(n_cloud=100, n_clear=30)
        balanced = build_balanced_dataset(records, cloud_ratio=0.5, seed=0)
        n_clear = sum(1 for r in balanced if r["label"] == 0)
        assert n_clear == 30  # all available clear tiles are kept
