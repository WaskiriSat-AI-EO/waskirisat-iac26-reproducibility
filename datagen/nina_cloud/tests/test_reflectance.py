"""Tests for processing/reflectance.py."""

import numpy as np
import pytest

from nina_cloud.processing.reflectance import normalize_reflectance, reflectance_to_uint8


class TestNormalizeReflectance:
    def test_full_scale_dn_maps_to_one(self):
        assert normalize_reflectance(np.array([10000])) == pytest.approx(1.0)

    def test_zero_dn_maps_to_zero(self):
        assert normalize_reflectance(np.array([0])) == pytest.approx(0.0)

    def test_above_scale_clips_to_one(self):
        result = normalize_reflectance(np.array([12000, 20000]))
        np.testing.assert_array_equal(result, np.array([1.0, 1.0], dtype=np.float32))

    def test_negative_clips_to_zero(self):
        result = normalize_reflectance(np.array([-100]))
        assert result[0] == 0.0

    def test_midpoint(self):
        result = normalize_reflectance(np.array([5000]))
        assert result[0] == pytest.approx(0.5)

    def test_output_dtype_is_float32(self):
        result = normalize_reflectance(np.array([5000], dtype=np.int16))
        assert result.dtype == np.float32

    def test_3d_rgb_array(self):
        rgb = np.full((4, 4, 3), 5000, dtype=np.int16)
        result = normalize_reflectance(rgb)
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 0.5, atol=1e-5)


class TestReflectanceToUint8:
    def test_zero_maps_to_zero(self):
        assert reflectance_to_uint8(np.array([0.0]))[0] == 0

    def test_one_maps_to_255(self):
        assert reflectance_to_uint8(np.array([1.0]))[0] == 255

    def test_half_maps_to_128(self):
        assert reflectance_to_uint8(np.array([0.5]))[0] == 128

    def test_output_dtype_is_uint8(self):
        result = reflectance_to_uint8(np.array([0.5], dtype=np.float32))
        assert result.dtype == np.uint8

    def test_above_one_clips_to_255(self):
        result = reflectance_to_uint8(np.array([1.5, 2.0]))
        np.testing.assert_array_equal(result, [255, 255])

    def test_below_zero_clips_to_zero(self):
        result = reflectance_to_uint8(np.array([-0.1, -1.0]))
        np.testing.assert_array_equal(result, [0, 0])

    def test_round_trip_precision(self):
        """Round-trip at uint8 precision: float → uint8 → float error < 1/255."""
        values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        u8 = reflectance_to_uint8(values)
        recovered = u8.astype(np.float32) / 255.0
        np.testing.assert_allclose(recovered, values, atol=1.0 / 255.0)
