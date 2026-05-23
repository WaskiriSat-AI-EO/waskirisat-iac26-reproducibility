"""Tests for labeling/split.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import pytest

from nina_thermal.datagen.labeling.split import granule_level_split
from nina_thermal.datagen import config


def _make_granules(n_high_fire: int, n_low_fire: int, seed: int = 0) -> dict:
    """Build a synthetic granule dict with the requested tier sizes.

    High-fire granules have 20 % fire labels; low-fire granules have 1 %.
    Each granule has 100 patches.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    granules = {}

    for i in range(n_high_fire):
        # fire fraction ~20 % > HIGH_FIRE_GRANULE_THRESH (0.05)
        labels = (rng.random(100) < 0.20).astype(int).tolist()
        granules[f"hf_{i:03d}"] = labels

    for i in range(n_low_fire):
        # fire fraction ~1 % <= HIGH_FIRE_GRANULE_THRESH
        labels = (rng.random(100) < 0.01).astype(int).tolist()
        granules[f"lf_{i:03d}"] = labels

    return granules


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def medium_dataset():
    """40 granules: 10 high-fire, 30 low-fire."""
    return _make_granules(n_high_fire=10, n_low_fire=30)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestGranuleLevelSplit:
    def test_no_granule_appears_in_more_than_one_split(self, medium_dataset):
        """CRITICAL: every granule_id must appear in exactly one split.

        Any overlap between splits would cause data leakage and
        invalidate validation / test metrics.
        """
        result = granule_level_split(medium_dataset, seed=42)

        train_set = set(result["train"])
        val_set = set(result["val"])
        test_set = set(result["test"])

        # Pairwise intersection must be empty
        assert train_set.isdisjoint(
            val_set
        ), "Leakage: granule(s) in both train and val"
        assert train_set.isdisjoint(
            test_set
        ), "Leakage: granule(s) in both train and test"
        assert val_set.isdisjoint(test_set), "Leakage: granule(s) in both val and test"

        # Union must equal the full input set
        all_assigned = train_set | val_set | test_set
        assert all_assigned == set(
            medium_dataset.keys()
        ), "Not all granules were assigned to a split"

    def test_each_split_contains_granules_from_both_tiers(self, medium_dataset):
        """Every split must receive at least one high-fire and one low-fire granule.

        This verifies the stratification step of the algorithm.
        """
        result = granule_level_split(medium_dataset, seed=42)

        for split_name, gids in result.items():
            has_high = any(g.startswith("hf_") for g in gids)
            has_low = any(g.startswith("lf_") for g in gids)
            assert has_high, f"Split '{split_name}' has no high-fire granules"
            assert has_low, f"Split '{split_name}' has no low-fire granules"

    def test_fractions_sum_to_one_within_rounding(self, medium_dataset):
        """The total number of assigned granules must equal the input count.

        Because fractions are rounded per-tier, the sum may differ by at
        most a handful of granules — we assert that EVERY granule is
        assigned exactly once (see also test_no_granule_appears_in_more_than_one_split).
        """
        result = granule_level_split(medium_dataset, seed=42)
        total_assigned = sum(len(gids) for gids in result.values())
        assert total_assigned == len(medium_dataset)

    def test_deterministic_with_same_seed(self, medium_dataset):
        """Two calls with the same seed must produce identical splits."""
        r1 = granule_level_split(medium_dataset, seed=7)
        r2 = granule_level_split(medium_dataset, seed=7)

        assert r1["train"] == r2["train"]
        assert r1["val"] == r2["val"]
        assert r1["test"] == r2["test"]

        # Different seed must differ (probabilistically guaranteed with large dataset)
        r3 = granule_level_split(medium_dataset, seed=99)
        assert (
            r1["train"] != r3["train"]
            or r1["val"] != r3["val"]
            or r1["test"] != r3["test"]
        ), "Different seeds produced identical splits — RNG seeding may be broken"
