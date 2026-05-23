import logging
from typing import List

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.datagen.processing.radiometry import bt_to_radiance, radiance_to_bt

logger = logging.getLogger(__name__)


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
    extra_arrays=None,
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
        extra_arrays: optional dict ``{name: array}`` of additional arrays to
            co-tile alongside ``bt_array``.  Extra arrays are aggregated using
            simple nanmean (NOT Planck-domain) — appropriate for quantities
            like delta_T that are temperature differences, not absolute
            radiances.  Accepted/rejected decisions are driven solely by
            ``bt_array``; extra arrays follow the same mask.

    Returns:
        If ``extra_arrays`` is None: list of float32 BT patches.
        If ``extra_arrays`` is provided: tuple ``(bt_patches, extra_patches)``
        where ``extra_patches`` is a dict ``{name: list_of_patches}``.
    """
    if lepton_w is None:
        lepton_w = config.LEPTON_RESOLUTION[0]
    if lepton_h is None:
        lepton_h = config.LEPTON_RESOLUTION[1]
    if max_nan_fraction is None:
        max_nan_fraction = config.MAX_NAN_FRACTION
    if agg_factor is None:
        agg_factor = config.SLSTR_AGG_FACTOR

    # Step 1: Aggregate BT to Lepton GSD (Planck-domain)
    agg = aggregate_bt_to_lepton_gsd(bt_array, agg_factor=agg_factor)
    H, W = agg.shape

    # Step 1b: Aggregate extra arrays with simple nanmean
    agg_extras: dict = {}
    if extra_arrays:
        for name, arr in extra_arrays.items():
            arr = np.asarray(arr, dtype=np.float32)
            Hx, Wx = arr.shape
            Hx_crop = (Hx // agg_factor) * agg_factor
            Wx_crop = (Wx // agg_factor) * agg_factor
            Hx_out  = Hx_crop // agg_factor
            Wx_out  = Wx_crop // agg_factor
            with np.errstate(all="ignore"):
                agg_extras[name] = np.nanmean(
                    arr[:Hx_crop, :Wx_crop]
                    .reshape(Hx_out, agg_factor, Wx_out, agg_factor),
                    axis=(1, 3),
                ).astype(np.float32)

    patches: List = []
    extra_patches: dict = {name: [] for name in agg_extras}

    stride_y = max(1, int(lepton_h * stride_fraction))
    stride_x = max(1, int(lepton_w * stride_fraction))

    n_rejected   = 0
    n_nan_filled = 0

    # Step 2: Tile with configurable stride (50% overlap by default)
    for row_start in range(0, H - lepton_h + 1, stride_y):
        for col_start in range(0, W - lepton_w + 1, stride_x):
            patch = agg[row_start:row_start + lepton_h,
                        col_start:col_start + lepton_w].copy()

            nan_fraction = np.isnan(patch).mean()

            if nan_fraction > max_nan_fraction:
                n_rejected += 1
                continue

            if nan_fraction > 0:
                patch[np.isnan(patch)] = np.nanmedian(patch)
                n_nan_filled += 1

            patches.append(patch.astype(np.float32))

            # Co-tile extras for the same accepted window
            for name, agg_extra in agg_extras.items():
                ep = agg_extra[row_start:row_start + lepton_h,
                               col_start:col_start + lepton_w].copy()
                if np.isnan(ep).any():
                    fill = np.nanmedian(ep)
                    ep[np.isnan(ep)] = fill if np.isfinite(fill) else 0.0
                extra_patches[name].append(ep.astype(np.float32))

    n_accepted = len(patches)
    n_total    = n_accepted + n_rejected
    rejection_rate = 100.0 * n_rejected / n_total if n_total > 0 else 0.0
    logger.info(
        "  tiling: accepted=%d  rejected=%d (%.1f%%)  nan_filled=%d",
        n_accepted, n_rejected, rejection_rate, n_nan_filled,
    )

    if extra_arrays is not None:
        return patches, extra_patches
    return patches
