"""Tests for processing/spatial.py."""

import numpy as np
import pytest

from nina_cloud.datagen import config
from nina_cloud.datagen.processing.spatial import tile_scene_to_patches, upsample_scl_to_rgb_scale


class TestUpsampleSclToRgbScale:
    def test_output_shape_is_double_input(self):
        scl = np.zeros((100, 200), dtype=np.uint8)
        out = upsample_scl_to_rgb_scale(scl)
        assert out.shape == (200, 400)

    def test_categorical_values_preserved(self):
        scl = np.array([[4, 9], [8, 0]], dtype=np.uint8)
        out = upsample_scl_to_rgb_scale(scl)
        # Top-left 2×2 block should all be 4
        assert out[0, 0] == 4
        assert out[0, 1] == 4
        assert out[1, 0] == 4
        assert out[1, 1] == 4
        # Top-right block should all be 9
        assert out[0, 2] == 9
        assert out[0, 3] == 9

    def test_output_dtype_is_uint8(self):
        scl = np.zeros((10, 10), dtype=np.uint8)
        out = upsample_scl_to_rgb_scale(scl)
        assert out.dtype == np.uint8


class TestTileSceneToPatchesShape:
    def test_patch_shapes_correct(self):
        H, W = 1600, 2560
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        scl = np.zeros((H // 2, W // 2), dtype=np.uint8)
        rgb_patches, scl_patches, origins = tile_scene_to_patches(
            rgb, scl, tile_w=1280, tile_h=800, stride_fraction=0.5
        )
        for p in rgb_patches:
            assert p.shape == (800, 1280, 3)
        for s in scl_patches:
            assert s.shape == (800, 1280)

    def test_patch_dtype_float32(self):
        H, W = 800, 1280
        rgb = np.zeros((H, W, 3), dtype=np.float64)
        scl = np.zeros((H // 2, W // 2), dtype=np.uint8)
        rgb_patches, _, _ = tile_scene_to_patches(rgb, scl, tile_w=1280, tile_h=800)
        for p in rgb_patches:
            assert p.dtype == np.float32

    def test_no_patches_when_scene_too_small(self):
        rgb = np.zeros((400, 640, 3), dtype=np.float32)
        scl = np.zeros((200, 320), dtype=np.uint8)
        rgb_patches, scl_patches, origins = tile_scene_to_patches(
            rgb, scl, tile_w=1280, tile_h=800
        )
        assert len(rgb_patches) == 0

    def test_origins_are_within_bounds(self):
        H, W = 1600, 2560
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        scl = np.zeros((H // 2, W // 2), dtype=np.uint8)
        _, _, origins = tile_scene_to_patches(rgb, scl, tile_w=1280, tile_h=800)
        for r, c in origins:
            assert 0 <= r <= H - 800
            assert 0 <= c <= W - 1280


class TestNodataRejection:
    def test_tile_rejected_above_threshold(self):
        """A tile where >10% of SCL pixels are nodata (class 0) must be rejected."""
        H, W = 800, 1280
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        scl = np.zeros((H // 2, W // 2), dtype=np.uint8)
        # Set 15% of SCL pixels to nodata (class 0)
        n_nodata = int(0.15 * scl.size)
        scl.ravel()[:n_nodata] = 0  # class 0 = nodata (already 0)
        # All pixels are class 0 = nodata; that is 100% nodata
        rgb_patches, _, _ = tile_scene_to_patches(
            rgb, scl, tile_w=1280, tile_h=800, max_nodata_frac=0.10
        )
        assert len(rgb_patches) == 0

    def test_tile_kept_below_threshold(self):
        """A tile where <10% of SCL pixels are nodata must be accepted."""
        H, W = 800, 1280
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        scl = np.full((H // 2, W // 2), 4, dtype=np.uint8)  # vegetation — valid
        # Add 5% nodata
        n_nodata = int(0.05 * scl.size)
        scl.ravel()[:n_nodata] = 0
        rgb_patches, _, _ = tile_scene_to_patches(
            rgb, scl, tile_w=1280, tile_h=800, max_nodata_frac=0.10
        )
        assert len(rgb_patches) == 1

    def test_scl_patch_contains_upsampled_values(self):
        """SCL values in returned patches match upsampled source."""
        H, W = 800, 1280
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        scl = np.full((H // 2, W // 2), 9, dtype=np.uint8)  # cloud high prob
        _, scl_patches, _ = tile_scene_to_patches(
            rgb, scl, tile_w=1280, tile_h=800
        )
        assert len(scl_patches) == 1
        assert np.all(scl_patches[0] == 9)
