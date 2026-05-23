"""Tests for processing/spatial.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.datagen.processing.spatial import (
    aggregate_bt_to_lepton_gsd,
    tile_granule_to_lepton_patches,
)


class TestRadianceConsistentAggregation:
    """Verify that radiance-consistent aggregation exceeds temperature averaging
    for a synthetic fire pixel scenario."""

    def test_fire_pixel_radiance_gt_temperature_average(self):
        """One 380 K pixel surrounded by 15 × 295 K pixels in a 4×4 block.

        Radiance-consistent output must exceed temperature-averaged output,
        and the difference must be > 1.0 K (physically expected ~1.8 K).
        """
        # Build a 4×4 block: one fire pixel, 15 ambient
        block = np.full((4, 4), 295.0, dtype=np.float64)
        block[0, 0] = 380.0

        # Radiance-consistent aggregation via our function
        agg_bt = aggregate_bt_to_lepton_gsd(
            block,
            agg_factor=4,
            wavelength_m=config.SLSTR_F2_WAVELENGTH_M,
        )
        radiance_agg_bt = float(agg_bt[0, 0])

        # Naive temperature average
        temperature_avg_bt = float(block.mean())

        assert radiance_agg_bt > temperature_avg_bt, (
            f"Radiance-consistent BT ({radiance_agg_bt:.4f} K) must exceed "
            f"temperature average ({temperature_avg_bt:.4f} K)"
        )

        diff = radiance_agg_bt - temperature_avg_bt
        assert diff > 1.0, f"Difference must be > 1.0 K, got {diff:.4f} K"

    def test_uniform_block_radiance_equals_temperature_average(self):
        """For a uniform temperature field the two methods must agree."""
        block = np.full((4, 4), 300.0, dtype=np.float64)
        agg_bt = aggregate_bt_to_lepton_gsd(
            block,
            agg_factor=4,
            wavelength_m=config.SLSTR_F2_WAVELENGTH_M,
        )
        assert abs(float(agg_bt[0, 0]) - 300.0) < 0.01


class TestOutputShape:
    """Verify output shape is (H//agg_factor, W//agg_factor)."""

    def test_shape_2400x1500_agg4(self):
        H, W = 2400, 1500
        bt = np.full((H, W), 295.0, dtype=np.float32)
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert agg.shape == (
            H // 4,
            W // 4,
        ), f"Expected ({H//4}, {W//4}), got {agg.shape}"

    def test_shape_non_multiple_cropped(self):
        """Non-multiple dimensions should be cropped before aggregation."""
        H, W = 2401, 1503  # not exact multiples of 4
        bt = np.full((H, W), 295.0, dtype=np.float32)
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert agg.shape == (H // 4, W // 4)

    def test_output_dtype_float32(self):
        bt = np.full((8, 8), 300.0, dtype=np.float64)
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert agg.dtype == np.float32


class TestNaNHandling:
    """Verify NaN propagation and block-level NaN output."""

    def test_all_nan_block_produces_nan_output(self):
        """A 4×4 block that is entirely NaN must produce a NaN output pixel."""
        bt = np.full((4, 4), 295.0, dtype=np.float64)
        bt[:, :] = np.nan  # all NaN
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert np.isnan(agg[0, 0]), "All-NaN block must yield NaN output"

    def test_partial_nan_block_not_nan(self):
        """A block with some NaN pixels (but not all) must not produce NaN."""
        bt = np.full((4, 4), 295.0, dtype=np.float64)
        bt[0, 0] = np.nan  # only one NaN
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert not np.isnan(
            agg[0, 0]
        ), "Partial-NaN block should not produce NaN (nanmean used)"

    def test_mixed_nan_and_valid_blocks(self):
        """Array with one all-NaN block and one valid block."""
        bt = np.full((4, 8), 300.0, dtype=np.float64)
        bt[:, :4] = np.nan  # left block all NaN
        agg = aggregate_bt_to_lepton_gsd(bt, agg_factor=4)
        assert np.isnan(agg[0, 0])
        assert not np.isnan(agg[0, 1])


class TestTileGranuleToLeptonPatches:
    """Tests for tile_granule_to_lepton_patches."""

    def test_patches_have_correct_shape(self):
        H, W = 2400, 1500
        bt = np.full((H, W), 295.0, dtype=np.float32)
        patches = tile_granule_to_lepton_patches(
            bt, agg_factor=4, lepton_h=120, lepton_w=160
        )
        for p in patches:
            assert p.shape == (120, 160)

    def test_patches_are_float32(self):
        bt = np.full((480, 640), 295.0, dtype=np.float64)
        patches = tile_granule_to_lepton_patches(
            bt, agg_factor=4, lepton_h=120, lepton_w=160
        )
        for p in patches:
            assert p.dtype == np.float32

    def test_nan_patch_rejected_above_threshold(self):
        """Patches with NaN fraction > max_nan_fraction must be discarded."""
        bt = np.full((480, 640), 295.0, dtype=np.float32)
        # Aggregated shape: (120, 160) — exactly one patch
        # Make more than 15% of pixels NaN in the aggregated array
        # Set rows 0-18 (19 of 120) as NaN in source → >15% of patch
        bt[:76, :] = np.nan  # 76 SLSTR rows → 19 agg rows out of 120 = 15.8%
        patches = tile_granule_to_lepton_patches(
            bt, agg_factor=4, lepton_h=120, lepton_w=160, max_nan_fraction=0.15
        )
        assert len(patches) == 0, "Patch with NaN fraction > 0.15 should be rejected"

    def test_nan_patch_filled_below_threshold(self):
        """Patches with NaN fraction <= max_nan_fraction must be kept and filled."""
        bt = np.full((480, 640), 300.0, dtype=np.float32)
        # Set 4 SLSTR rows → 1 agg row of NaN out of 120 = 0.83% < 15%
        bt[:4, :] = np.nan
        patches = tile_granule_to_lepton_patches(
            bt, agg_factor=4, lepton_h=120, lepton_w=160, max_nan_fraction=0.15
        )
        assert len(patches) == 1
        assert not np.isnan(
            patches[0]
        ).any(), "NaN pixels below threshold should be filled"

    def test_no_patches_when_array_too_small(self):
        """An array smaller than one patch after aggregation yields no patches."""
        bt = np.full((4, 4), 295.0, dtype=np.float32)
        patches = tile_granule_to_lepton_patches(
            bt, agg_factor=4, lepton_h=120, lepton_w=160
        )
        assert len(patches) == 0
