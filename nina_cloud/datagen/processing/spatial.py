"""Spatial tiling and SCL upsampling for Sentinel-2 cloud dataset generation."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from nina_cloud.datagen import config

logger = logging.getLogger(__name__)


def upsample_scl_to_rgb_scale(scl_uint8: np.ndarray) -> np.ndarray:
    """Upsample an SCL array from 20m to 10m resolution (2× in each dimension).

    Uses nearest-neighbour interpolation to preserve categorical class labels.

    Args:
        scl_uint8: uint8 (H, W) SCL array at 20m resolution.

    Returns:
        uint8 (H*2, W*2) SCL array at 10m resolution.
    """
    h, w = scl_uint8.shape
    return cv2.resize(
        scl_uint8,
        (w * 2, h * 2),
        interpolation=cv2.INTER_NEAREST,
    )


def tile_scene_to_patches(
    rgb_float: np.ndarray,
    scl_uint8: np.ndarray,
    tile_w: Optional[int] = None,
    tile_h: Optional[int] = None,
    stride_fraction: float = config.TILE_STRIDE_FRACTION,
    max_nodata_frac: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """Tile a Sentinel-2 scene into overlapping PAG7936-sized patches.

    Slides a (tile_h × tile_w) window over the scene with the given stride
    fraction.  Tiles with SCL nodata fraction exceeding max_nodata_frac are
    discarded; all others are returned with no further modification (cloud
    labeling is handled by labeling.cloud_label).

    The SCL array must be at 20m resolution (half the RGB pixel dimensions).
    It is upsampled 2× with nearest-neighbour before tiling so that SCL and
    RGB tile shapes are compatible.

    Args:
        rgb_float: float32 (H_s2, W_s2, 3) BOA reflectance at 10m.
        scl_uint8: uint8 (H_scl, W_scl) SCL at 20m (i.e. H_scl ≈ H_s2 / 2).
        tile_w: Tile width in pixels; default TILE_WIDTH_PX (1280).
        tile_h: Tile height in pixels; default TILE_HEIGHT_PX (800).
        stride_fraction: Step size as a fraction of tile size; default 0.5.
        max_nodata_frac: Discard tiles with nodata fraction above this value;
            default MAX_NODATA_FRACTION (0.10).

    Returns:
        Tuple of (rgb_patches, scl_patches, patch_origins) where:
          - rgb_patches: list of float32 (tile_h, tile_w, 3) arrays
          - scl_patches: list of uint8 (tile_h, tile_w) arrays (at 10m)
          - patch_origins: list of (row, col) top-left corner in S2 pixels
    """
    tile_w = tile_w or config.TILE_WIDTH_PX
    tile_h = tile_h or config.TILE_HEIGHT_PX
    max_nodata_frac = (
        max_nodata_frac if max_nodata_frac is not None else config.MAX_NODATA_FRACTION
    )

    scl_10m = upsample_scl_to_rgb_scale(scl_uint8)

    # Align SCL to RGB dimensions — after 2× upsample minor off-by-one can occur
    h_rgb, w_rgb = rgb_float.shape[:2]
    scl_10m = scl_10m[:h_rgb, :w_rgb]

    stride_h = max(1, int(tile_h * stride_fraction))
    stride_w = max(1, int(tile_w * stride_fraction))

    rgb_patches: List[np.ndarray] = []
    scl_patches: List[np.ndarray] = []
    origins: List[Tuple[int, int]] = []

    n_accepted = n_rejected = 0

    for r in range(0, h_rgb - tile_h + 1, stride_h):
        for c in range(0, w_rgb - tile_w + 1, stride_w):
            scl_tile = scl_10m[r : r + tile_h, c : c + tile_w]

            nodata_mask = np.isin(scl_tile, list(config.SCL_INVALID_CLASSES))
            nodata_frac = nodata_mask.mean()

            if nodata_frac > max_nodata_frac:
                n_rejected += 1
                continue

            rgb_tile = rgb_float[r : r + tile_h, c : c + tile_w].astype(np.float32)
            rgb_patches.append(rgb_tile)
            scl_patches.append(scl_tile)
            origins.append((r, c))
            n_accepted += 1

    logger.info(
        "tiling: tile=%dx%d  stride=%dx%d  accepted=%d  rejected_nodata=%d",
        tile_w,
        tile_h,
        stride_w,
        stride_h,
        n_accepted,
        n_rejected,
    )
    return rgb_patches, scl_patches, origins
