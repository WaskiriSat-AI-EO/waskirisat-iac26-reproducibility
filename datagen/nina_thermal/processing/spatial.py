from typing import List

import numpy as np

from nina_thermal import config
from nina_thermal.processing.radiometry import bt_to_radiance, radiance_to_bt


def aggregate_bt_to_lepton_gsd(bt_array, agg_factor=None, wavelength_m=None):
    """Aggregate SLSTR BT array to Lepton GSD using radiance-consistent averaging.

    CRITICAL: Uses radiance-domain averaging, not temperature averaging.
    Sensors integrate radiance over their footprint; temperature averaging
    underestimates hot sub-pixel fire target contributions.

    Args:
        bt_array: float array of brightness temperatures [K], shape (H, W)
        agg_factor: pixels per side to aggregate (default SLSTR_AGG_FACTOR=4)
        wavelength_m: center wavelength [m] (default SLSTR_F2_WAVELENGTH_M)

    Returns:
        float32 array of shape (H//agg_factor, W//agg_factor)
    """
    if agg_factor is None:
        agg_factor = config.SLSTR_AGG_FACTOR
    if wavelength_m is None:
        wavelength_m = config.SLSTR_F2_WAVELENGTH_M

    bt_array = np.asarray(bt_array, dtype=np.float64)
    H, W = bt_array.shape

    # Step 1: Crop to exact multiple of agg_factor
    H_crop = (H // agg_factor) * agg_factor
    W_crop = (W // agg_factor) * agg_factor
    bt_crop = bt_array[:H_crop, :W_crop]

    # Step 2: Convert to radiance
    L = bt_to_radiance(bt_crop, wavelength_m)

    # Step 3: Reshape to (H//agg, agg, W//agg, agg)
    H_out = H_crop // agg_factor
    W_out = W_crop // agg_factor
    L_reshaped = L.reshape(H_out, agg_factor, W_out, agg_factor)

    # Step 4: nanmean over the aggregation axes
    with np.errstate(all="ignore"):
        L_mean = np.nanmean(L_reshaped, axis=(1, 3))

    # Step 5: Convert mean radiance back to BT
    bt_out = radiance_to_bt(L_mean, wavelength_m)

    return bt_out.astype(np.float32)


def tile_granule_to_lepton_patches(
    bt_array,
    agg_factor=None,
    lepton_w=None,
    lepton_h=None,
    max_nan_fraction=None,
    stride_fraction=0.5,
):
    """Aggregate and tile a BT granule into Lepton-sized patches.

    Args:
        bt_array: float array of brightness temperatures [K], shape (H, W)
        agg_factor: passed through to aggregate_bt_to_lepton_gsd
        lepton_w: patch width in pixels (default LEPTON_RESOLUTION[0]=160)
        lepton_h: patch height in pixels (default LEPTON_RESOLUTION[1]=120)
        max_nan_fraction: maximum NaN fraction to accept a patch
                          (default MAX_NAN_FRACTION)
        stride_fraction: stride as a fraction of patch size (default 0.5 = 50%
                         overlap). Use 1.0 for no overlap (original behaviour).

    Returns:
        List of float32 arrays, each of shape (lepton_h, lepton_w)
    """
    if lepton_w is None:
        lepton_w = config.LEPTON_RESOLUTION[0]
    if lepton_h is None:
        lepton_h = config.LEPTON_RESOLUTION[1]
    if max_nan_fraction is None:
        max_nan_fraction = config.MAX_NAN_FRACTION

    # Step 1: Aggregate to Lepton GSD
    agg = aggregate_bt_to_lepton_gsd(bt_array, agg_factor=agg_factor)

    H, W = agg.shape
    patches = []

    stride_y = max(1, int(lepton_h * stride_fraction))
    stride_x = max(1, int(lepton_w * stride_fraction))

    # Step 2: Tile with configurable stride (50% overlap by default)
    for row_start in range(0, H - lepton_h + 1, stride_y):
        for col_start in range(0, W - lepton_w + 1, stride_x):
            patch = agg[row_start:row_start + lepton_h,
                        col_start:col_start + lepton_w].copy()

            # Step 3a: Compute NaN fraction
            nan_fraction = np.isnan(patch).mean()

            # Step 3b: Reject if too many NaNs
            if nan_fraction > max_nan_fraction:
                continue

            # Step 3c: Fill residual NaNs with patch nanmedian
            if nan_fraction > 0:
                median_val = np.nanmedian(patch)
                patch[np.isnan(patch)] = median_val

            patches.append(patch.astype(np.float32))

    return patches
