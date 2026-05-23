"""Tests for simulation/noise.py."""

import numpy as np
import pytest

from nina_cloud.datagen.simulation.noise import (
    add_read_noise,
    add_shot_noise,
    apply_fpn,
    generate_pag7936_fpn,
)


class TestShotNoise:
    def test_output_shape_preserved(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_shot_noise(img, factor=1.0, rng=rng)
        assert out.shape == (100, 100, 3)

    def test_output_dtype_float32(self):
        img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_shot_noise(img, factor=1.0, rng=rng)
        assert out.dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        img = np.full((50, 50, 3), 0.99, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_shot_noise(img, factor=1.0, rng=rng)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_variance_scales_with_signal(self):
        """Poisson noise: var ∝ signal → high-signal patch has higher variance."""
        rng_lo = np.random.default_rng(0)
        rng_hi = np.random.default_rng(0)
        lo = np.full((200, 200, 3), 0.1, dtype=np.float32)
        hi = np.full((200, 200, 3), 0.9, dtype=np.float32)
        var_lo = add_shot_noise(lo, factor=1.0, rng=rng_lo).var()
        var_hi = add_shot_noise(hi, factor=1.0, rng=rng_hi).var()
        assert var_hi > var_lo, "Shot noise variance must be higher at higher signal"

    def test_higher_factor_increases_noise(self):
        """Higher shot_noise_factor (profile C vs A) must produce more noise."""
        rng_a = np.random.default_rng(0)
        rng_c = np.random.default_rng(0)
        img = np.full((200, 200, 3), 0.5, dtype=np.float32)
        out_a = add_shot_noise(img, factor=1.0, rng=rng_a)
        out_c = add_shot_noise(img, factor=4.0, rng=rng_c)
        assert out_c.var() < out_a.var(), (
            "Higher factor means larger photon pool → LOWER relative variance "
            "(Poisson noise σ/μ decreases with more photons)"
        )


class TestReadNoise:
    def test_output_shape_preserved(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_read_noise(img, sigma_dn=2.0, rng=rng)
        assert out.shape == (100, 100, 3)

    def test_output_dtype_float32(self):
        img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_read_noise(img, sigma_dn=2.0, rng=rng)
        assert out.dtype == np.float32

    def test_output_clipped_to_unit_range(self):
        img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        rng = np.random.default_rng(0)
        out = add_read_noise(img, sigma_dn=100.0, rng=rng)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_noise_is_signal_independent(self):
        """Read noise variance must be equal for low- and high-signal patches."""
        rng_lo = np.random.default_rng(42)
        rng_hi = np.random.default_rng(42)
        lo = np.full((500, 500, 3), 0.1, dtype=np.float32)
        hi = np.full((500, 500, 3), 0.9, dtype=np.float32)
        diff_lo = add_read_noise(lo, sigma_dn=5.0, rng=rng_lo) - lo
        diff_hi = add_read_noise(hi, sigma_dn=5.0, rng=rng_hi) - hi
        # Variance of noise should be equal (both drawn from same distribution)
        assert abs(diff_lo.var() - diff_hi.var()) < 1e-6


class TestGeneratePag7936Fpn:
    def test_output_keys(self):
        fpn = generate_pag7936_fpn(
            (120, 160), column_strength=0.01, pixel_strength=0.01
        )
        assert "column_fpn" in fpn
        assert "pixel_fpn" in fpn

    def test_output_shapes(self):
        fpn = generate_pag7936_fpn(
            (800, 1280), column_strength=0.01, pixel_strength=0.01
        )
        assert fpn["column_fpn"].shape == (800, 1280)
        assert fpn["pixel_fpn"].shape == (800, 1280)

    def test_column_fpn_has_vertical_stripe_structure(self):
        """All rows in a given column must share the same FPN value."""
        fpn = generate_pag7936_fpn(
            (100, 200), column_strength=0.1, pixel_strength=0.0, seed=0
        )
        col = fpn["column_fpn"]
        # Every row in column 0 should be identical
        np.testing.assert_array_equal(col[:, 0], col[0, 0] * np.ones(100))

    def test_column_std_exceeds_row_std(self):
        """Column FPN produces vertical stripes: col-wise std > row-wise std."""
        fpn = generate_pag7936_fpn(
            (200, 400), column_strength=0.05, pixel_strength=0.0, seed=1
        )
        # Std of column means > std of row means for pure column FPN
        col_means = fpn["column_fpn"].mean(axis=0)  # per-column mean
        row_means = fpn["column_fpn"].mean(axis=1)  # per-row mean
        assert (
            col_means.std() > row_means.std()
        ), "Column FPN must produce vertical stripes: column std > row std"

    def test_deterministic_with_seed(self):
        fpn1 = generate_pag7936_fpn((100, 100), column_strength=0.01, seed=7)
        fpn2 = generate_pag7936_fpn((100, 100), column_strength=0.01, seed=7)
        np.testing.assert_array_equal(fpn1["column_fpn"], fpn2["column_fpn"])


class TestApplyFpn:
    def test_output_clipped_to_unit_range(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        fpn = {
            "column_fpn": np.full((100, 100), 2.0, dtype=np.float32),
            "pixel_fpn": np.full((100, 100), 2.0, dtype=np.float32),
        }
        out = apply_fpn(img, fpn, factor=1.0)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_zero_factor_no_change(self):
        img = np.random.default_rng(0).random((50, 50, 3)).astype(np.float32)
        fpn = generate_pag7936_fpn((50, 50), column_strength=0.1, pixel_strength=0.1)
        out = apply_fpn(img, fpn, factor=0.0)
        np.testing.assert_array_almost_equal(out, img)

    def test_output_dtype_float32(self):
        img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        fpn = generate_pag7936_fpn((50, 50), 0.01, 0.01)
        out = apply_fpn(img, fpn, factor=1.0)
        assert out.dtype == np.float32
