"""Tests for simulation/noise.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import numpy as np
import pytest

from nina_thermal.datagen.simulation.noise import (
    calculate_thermal_noise_sigma,
    generate_lepton_fpn,
    apply_fpn,
)
from nina_thermal.datagen import config


class TestCalculateThermalNoiseSigma:
    def test_sigma_scales_with_netd_and_agc_range(self):
        """sigma = (netd_mk / 1000) / agc_range_k — verify the formula directly."""
        netd_mk = 50.0  # mK
        agc_range = 100.0  # K
        expected = (netd_mk / 1000.0) / agc_range  # 0.0005

        result = calculate_thermal_noise_sigma(netd_mk=netd_mk, agc_range_k=agc_range)
        assert result == pytest.approx(expected)

    def test_sigma_doubles_when_netd_doubles(self):
        """Doubling NETD must double sigma (linearity check)."""
        s1 = calculate_thermal_noise_sigma(netd_mk=50.0, agc_range_k=100.0)
        s2 = calculate_thermal_noise_sigma(netd_mk=100.0, agc_range_k=100.0)
        assert s2 == pytest.approx(2 * s1)

    def test_sigma_halves_when_agc_range_doubles(self):
        """Doubling the AGC range must halve sigma (inverse linearity check)."""
        s1 = calculate_thermal_noise_sigma(netd_mk=50.0, agc_range_k=100.0)
        s2 = calculate_thermal_noise_sigma(netd_mk=50.0, agc_range_k=200.0)
        assert s2 == pytest.approx(s1 / 2)

    def test_defaults_use_config_values(self):
        """Calling with no args must use config.LEPTON_NETD_MK and FIXED_TEMP_RANGE_K."""
        expected = (config.LEPTON_NETD_MK / 1000.0) / config.FIXED_TEMP_RANGE_K
        assert calculate_thermal_noise_sigma() == pytest.approx(expected)


class TestColumnFpnStructure:
    """Column FPN must produce vertical stripe structure.

    Physical basis: column-level readout amplifier variation is constant
    along each column but varies across columns. This means:
      - std across rows (per-column variation) should be HIGH
        (different columns have different offsets)
      - std along rows (per-row variation) should be LOW
        (all pixels in the same row share the same column offsets)

    Equivalently: std of column means > std of row means.
    """

    def test_column_fpn_vertical_stripe_structure(self):
        """std of column means must exceed std of row means for column FPN."""
        fpn = generate_lepton_fpn(
            resolution=(160, 120),
            column_strength=0.02,
            pixel_strength=0.0,  # isolate column component
            seed=0,
        )
        col_map = fpn["column_fpn"]  # (H=120, W=160)

        # Column means: mean over rows → (W,) — should vary strongly across columns
        col_means = col_map.mean(axis=0)  # shape (W,)
        # Row means: mean over columns → (H,) — should be near-constant
        row_means = col_map.mean(axis=1)  # shape (H,)

        assert col_means.std() > row_means.std(), (
            f"Column FPN does not show vertical stripe structure: "
            f"col_mean_std={col_means.std():.6f}, row_mean_std={row_means.std():.6f}"
        )

    def test_pixel_fpn_has_no_dominant_column_structure(self):
        """Pure pixel FPN must NOT produce systematic column variation."""
        fpn = generate_lepton_fpn(
            resolution=(160, 120),
            column_strength=0.0,  # isolate pixel component
            pixel_strength=0.02,
            seed=1,
        )
        pix_map = fpn["pixel_fpn"]

        col_means = pix_map.mean(axis=0)
        row_means = pix_map.mean(axis=1)

        # With pure pixel noise the column and row mean stds should be comparable
        ratio = col_means.std() / (row_means.std() + 1e-12)
        assert (
            ratio < 3.0
        ), f"Pixel FPN unexpectedly shows column structure (ratio={ratio:.2f})"


class TestApplyFpn:
    def test_apply_fpn_clips_to_0_1(self):
        """Output of apply_fpn must always be clipped to [0, 1]."""
        fpn = generate_lepton_fpn(
            resolution=(160, 120),
            column_strength=0.1,
            pixel_strength=0.1,
            seed=2,
        )
        # Use an image that is already near the edges so FPN can push outside [0,1]
        img = np.full((120, 160), 0.98, dtype=np.float32)
        result = apply_fpn(img, fpn, factor=2.0)

        assert result.dtype == np.float32
        assert result.min() >= 0.0, f"apply_fpn output below 0: {result.min()}"
        assert result.max() <= 1.0, f"apply_fpn output above 1: {result.max()}"
