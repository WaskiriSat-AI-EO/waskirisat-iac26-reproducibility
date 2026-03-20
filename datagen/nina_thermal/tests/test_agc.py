"""Tests for processing/agc.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import numpy as np
import pytest

from nina_thermal.processing.agc import apply_agc, agc_to_uint8, compute_global_percentiles
from nina_thermal import config


T_MIN = config.FIXED_TEMP_MIN_K   # 258.15 K
T_MAX = config.FIXED_TEMP_MAX_K   # 358.15 K


class TestApplyAgc:
    def test_values_within_range_map_to_zero_one(self):
        """Values strictly between t_min and t_max must map to (0, 1)."""
        bt = np.array([T_MIN, (T_MIN + T_MAX) / 2.0, T_MAX], dtype=np.float32)
        result = apply_agc(bt, T_MIN, T_MAX)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_values_below_t_min_clip_to_zero(self):
        """BT values below t_min must be clipped to exactly 0.0."""
        bt = np.array([T_MIN - 50.0, T_MIN - 1.0, T_MIN - 0.001], dtype=np.float32)
        result = apply_agc(bt, T_MIN, T_MAX)
        assert np.all(result == 0.0), f"Expected all zeros, got {result}"

    def test_values_above_t_max_clip_to_one(self):
        """BT values above t_max must be clipped to exactly 1.0."""
        bt = np.array([T_MAX + 0.001, T_MAX + 10.0, T_MAX + 500.0], dtype=np.float32)
        result = apply_agc(bt, T_MIN, T_MAX)
        assert np.all(result == 1.0), f"Expected all ones, got {result}"


class TestAgcToUint8:
    def test_uint8_cast_is_lossless_at_uint8_precision(self):
        """Round-trip float->uint8->float must be lossless at uint8 precision.

        For any value v = k/255 (k integer in [0,255]), agc_to_uint8 must
        recover exactly k. This is the lossless condition at uint8 precision.
        """
        k_values = np.arange(0, 256, dtype=np.float32)
        normalized = k_values / 255.0
        result = agc_to_uint8(normalized)
        expected = k_values.astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_clipping_behavior_on_out_of_range_floats(self):
        """Values outside [0,1] due to floating point must be clipped safely."""
        # Slightly above 1.0 and below 0.0 should not produce overflow
        normalized = np.array([-0.01, 0.0, 0.5, 1.0, 1.01], dtype=np.float32)
        result = agc_to_uint8(normalized)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 128   # round(0.5 * 255) = round(127.5) = 128
        assert result[3] == 255
        assert result[4] == 255
