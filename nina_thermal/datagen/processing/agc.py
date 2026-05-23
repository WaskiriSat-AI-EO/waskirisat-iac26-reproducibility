"""Automatic Gain Control (AGC) functions for thermal patch normalization."""

import logging
from typing import Tuple, Optional, List

import numpy as np

from nina_thermal.datagen import config

logger = logging.getLogger(__name__)


def apply_agc(bt_patch: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Normalize a BT patch to [0, 1] using fixed temperature bounds.

    Formula: clip((BT - t_min) / (t_max - t_min), 0, 1)

    Args:
        bt_patch: Brightness temperature array [K], any shape.
        t_min: Lower bound temperature [K].
        t_max: Upper bound temperature [K].

    Returns:
        float32 array clipped to [0, 1].
    """
    normalized = (bt_patch.astype(np.float32) - t_min) / (t_max - t_min)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def compute_global_percentiles(
    all_patches: List[np.ndarray],
    percentiles: Tuple[float, float] = (2.0, 98.0),
    max_samples_per_patch: int = 5000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute global low/high BT percentiles across a collection of patches.

    Samples up to max_samples_per_patch finite values per patch to keep
    memory bounded for large datasets.

    Args:
        all_patches: List of float32 BT arrays [K].
        percentiles: (low_pct, high_pct) percentile pair.
        max_samples_per_patch: Maximum number of finite values sampled per patch.
        seed: Optional RNG seed for reproducible sampling.

    Returns:
        (T_low, T_high) in Kelvin.
    """
    rng = np.random.default_rng(seed)
    samples: List[np.ndarray] = []

    for patch in all_patches:
        finite_vals = patch[np.isfinite(patch)].ravel()
        if finite_vals.size == 0:
            continue
        if finite_vals.size > max_samples_per_patch:
            idx = rng.choice(
                finite_vals.size, size=max_samples_per_patch, replace=False
            )
            finite_vals = finite_vals[idx]
        samples.append(finite_vals)

    n_skipped = len(all_patches) - len(samples)
    if not samples:
        raise ValueError("No finite values found across all patches.")

    all_values = np.concatenate(samples)
    t_low  = float(np.percentile(all_values, percentiles[0]))
    t_high = float(np.percentile(all_values, percentiles[1]))

    logger.info(
        "global percentiles: patches_used=%d/%d (%d skipped all-NaN)  "
        "total_samples=%d  T_low=%.2fK  T_high=%.2fK",
        len(samples), len(all_patches), n_skipped,
        len(all_values), t_low, t_high,
    )
    return t_low, t_high


def agc_to_uint8(normalized: np.ndarray) -> np.ndarray:
    """Convert a float32 [0, 1] normalized array to uint8 [0, 255].

    Formula: clip(round(normalized * 255), 0, 255).astype(uint8)

    This is the final quantization step before model input. It is lossless
    relative to the float32 representation at uint8 precision.

    Args:
        normalized: float32 array with values in [0, 1].

    Returns:
        uint8 array with values in [0, 255].
    """
    return np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)
