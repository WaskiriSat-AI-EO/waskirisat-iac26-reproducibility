"""Tests for labeling/cloud_label.py."""

import numpy as np
import pytest

from nina_cloud.datagen.labeling.cloud_label import (
    classify_tile_type,
    compute_tile_cloud_label,
)


class TestComputeTileCloudLabel:
    def test_all_vegetation_is_clear(self):
        scl = np.full((100, 100), 4, dtype=np.uint8)  # vegetation
        label, frac = compute_tile_cloud_label(scl)
        assert label == 0
        assert frac == pytest.approx(0.0)

    def test_all_cloud_high_prob_is_cloud(self):
        scl = np.full((100, 100), 9, dtype=np.uint8)  # cloud high prob
        label, frac = compute_tile_cloud_label(scl)
        assert label == 1
        assert frac == pytest.approx(1.0)

    def test_all_cloud_medium_prob_is_cloud(self):
        scl = np.full((50, 50), 8, dtype=np.uint8)  # cloud medium prob
        label, frac = compute_tile_cloud_label(scl)
        assert label == 1

    def test_all_thin_cirrus_is_cloud(self):
        scl = np.full((50, 50), 10, dtype=np.uint8)  # thin cirrus
        label, frac = compute_tile_cloud_label(scl)
        assert label == 1

    def test_fifty_percent_is_ambiguous(self):
        scl = np.zeros((100, 100), dtype=np.uint8)
        scl[:50, :] = 9  # 50% cloud high prob
        scl[50:, :] = 4  # 50% vegetation
        label, frac = compute_tile_cloud_label(scl)
        assert label == -1
        assert frac == pytest.approx(0.5)

    def test_snow_does_not_increment_cloud_frac(self):
        """SCL=11 (snow/ice) is not a cloud class; should yield clear label."""
        scl = np.full((100, 100), 11, dtype=np.uint8)
        label, frac = compute_tile_cloud_label(scl)
        assert label == 0
        assert frac == pytest.approx(0.0)

    def test_nodata_excluded_from_denominator(self):
        """SCL=0 (nodata) pixels must not appear in the denominator."""
        scl = np.zeros((100, 100), dtype=np.uint8)  # all nodata
        label, frac = compute_tile_cloud_label(scl)
        # All pixels are invalid → denominator is 0 → ambiguous
        assert label == -1
        assert frac == pytest.approx(0.0)

    def test_nodata_mixed_with_clear(self):
        """50% nodata + 50% vegetation → cloud_frac = 0.0 → clear."""
        scl = np.full((100, 100), 4, dtype=np.uint8)
        scl[:50, :] = 0  # nodata
        label, frac = compute_tile_cloud_label(scl)
        assert label == 0
        assert frac == pytest.approx(0.0)

    def test_saturated_excluded_from_denominator(self):
        """SCL=1 (saturated) pixels must not appear in the denominator."""
        scl = np.ones((100, 100), dtype=np.uint8)  # all saturated
        label, frac = compute_tile_cloud_label(scl)
        assert label == -1  # empty denominator → ambiguous

    def test_above_high_threshold_is_cloud(self):
        """80% cloud → cloud_frac > 0.70 → label 1."""
        scl = np.full((100, 100), 4, dtype=np.uint8)
        scl[:80, :] = 9  # 80% cloud
        label, frac = compute_tile_cloud_label(scl)
        assert label == 1

    def test_below_low_threshold_is_clear(self):
        """10% cloud → cloud_frac < 0.30 → label 0."""
        scl = np.full((100, 100), 4, dtype=np.uint8)
        scl[:10, :] = 9  # 10% cloud
        label, frac = compute_tile_cloud_label(scl)
        assert label == 0

    def test_cloud_shadow_not_counted_as_cloud(self):
        """SCL=3 (cloud shadow) must NOT increment the cloud numerator."""
        scl = np.full((100, 100), 3, dtype=np.uint8)  # all cloud shadow
        label, frac = compute_tile_cloud_label(scl)
        assert label == 0
        assert frac == pytest.approx(0.0)


class TestClassifyTileType:
    def test_label_1_is_cloud(self):
        assert classify_tile_type(1, 0.85) == "cloud"

    def test_label_0_is_clear(self):
        assert classify_tile_type(0, 0.10) == "clear"

    def test_label_minus1_is_ambiguous(self):
        assert classify_tile_type(-1, 0.50) == "ambiguous"
