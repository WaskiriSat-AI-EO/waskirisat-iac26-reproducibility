"""Tests for dataset/balance.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import pytest

from nina_thermal.datagen.dataset.balance import build_balanced_dataset
from nina_thermal.datagen import config


def _make_records(n_fire: int, n_nofire: int) -> list:
    """Minimal record dicts for testing."""
    records = []
    for i in range(n_fire):
        records.append({"label": 1, "granule_id": f"g_{i}", "patch": f"fire_{i:04d}.png"})
    for i in range(n_nofire):
        records.append({"label": 0, "granule_id": f"g_{i}", "patch": f"nofire_{i:04d}.png"})
    return records


class TestBuildBalancedDataset:
    def test_fire_patches_never_removed(self):
        """Every fire record in the input must appear in the output."""
        records = _make_records(n_fire=50, n_nofire=500)
        result = build_balanced_dataset(records, fire_ratio=0.20, seed=42)

        fire_in  = {r["patch"] for r in records if r["label"] == 1}
        fire_out = {r["patch"] for r in result  if r["label"] == 1}

        assert fire_in == fire_out, (
            f"Fire patches were removed: missing {fire_in - fire_out}"
        )

    def test_output_fire_fraction_within_1pct_of_target(self):
        """Achieved fire fraction must be within 1 percentage point of the target."""
        fire_ratio = config.FIRE_RATIO   # 0.20
        records = _make_records(n_fire=200, n_nofire=2000)
        result = build_balanced_dataset(records, fire_ratio=fire_ratio, seed=42)

        n_fire  = sum(1 for r in result if r["label"] == 1)
        achieved = n_fire / len(result)

        assert abs(achieved - fire_ratio) <= 0.01, (
            f"Fire fraction {achieved:.4f} deviates from target {fire_ratio} by more than 1 %"
        )

    def test_deterministic_with_same_seed(self):
        """Two calls with the same seed must return records in the same order."""
        records = _make_records(n_fire=100, n_nofire=1000)

        r1 = build_balanced_dataset(records, fire_ratio=0.20, seed=7)
        r2 = build_balanced_dataset(records, fire_ratio=0.20, seed=7)

        patches1 = [r["patch"] for r in r1]
        patches2 = [r["patch"] for r in r2]
        assert patches1 == patches2, "Same seed produced different orderings"

        # Different seed should (probabilistically) differ
        r3 = build_balanced_dataset(records, fire_ratio=0.20, seed=99)
        patches3 = [r["patch"] for r in r3]
        assert patches1 != patches3, (
            "Different seeds produced identical output — RNG seeding may be broken"
        )
