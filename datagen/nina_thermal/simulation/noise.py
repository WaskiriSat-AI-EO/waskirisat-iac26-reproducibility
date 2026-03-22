"""Thermal noise and Fixed Pattern Noise (FPN) simulation.

Models the two dominant noise sources in a VOx microbolometer:

* **Thermal noise** (NETD-derived Gaussian) — a temperature uncertainty
  that must be added in normalised BT space, before any rendering step.
* **Fixed Pattern Noise (FPN)** — column-level readout amplifier
  variation (dominant) plus residual per-pixel non-uniformity after
  factory NUC.  Applied after the optical path (post-uint8 cast).
"""

from typing import Dict, Optional

import numpy as np

from nina_thermal import config


def calculate_thermal_noise_sigma(
    netd_mk: float = None,
    agc_range_k: float = None,
) -> float:
    """Convert NETD to noise sigma in normalised [0, 1] BT space.

    Formula::

        sigma = (netd_mk / 1000.0) / agc_range_k

    Args:
        netd_mk: Noise Equivalent Temperature Difference [mK].
            Defaults to ``config.LEPTON_NETD_MK``.
        agc_range_k: Temperature range spanned by the AGC window [K].
            Defaults to ``config.FIXED_TEMP_RANGE_K``.

    Returns:
        Noise standard deviation in normalised BT units.
    """
    if netd_mk is None:
        netd_mk = config.LEPTON_NETD_MK
    if agc_range_k is None:
        agc_range_k = config.FIXED_TEMP_RANGE_K
    return (netd_mk / 1000.0) / agc_range_k


def add_thermal_noise(
    bt_normalized: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian thermal noise to a normalised BT array.

    Noise is added in normalised [0, 1] BT space — before ``agc_to_uint8``
    and before any colormap rendering.  This matches physical reality:
    microbolometer NETD is a temperature uncertainty, so it must be applied
    in temperature space.  Adding noise after colormap rendering would
    create artefacts whose magnitude depends on the colormap gradient,
    which is physically incorrect.

    Args:
        bt_normalized: float32 array of normalised brightness temperatures
            in [0, 1].
        sigma: Gaussian noise standard deviation (in normalised BT units).
        rng: ``numpy.random.Generator`` instance for reproducible noise.

    Returns:
        float32 array clipped to [0, 1].
    """
    noise = rng.standard_normal(bt_normalized.shape).astype(np.float32) * sigma
    return np.clip(bt_normalized + noise, 0.0, 1.0).astype(np.float32)


def generate_lepton_fpn(
    resolution: tuple = None,
    column_strength: float = 0.0,
    pixel_strength: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a two-component Fixed Pattern Noise map for a Lepton sensor.

    Two components matching VOx microbolometer readout architecture:

    * **column_fpn** — shape ``(H, W)``, one random offset per column
      broadcast over all rows.  Models column-level readout amplifier
      variation, the dominant FPN artifact.
    * **pixel_fpn** — shape ``(H, W)``, independent per-pixel random
      offset.  Models residual non-uniformity after factory NUC.

    Both components are zero-mean unit-variance Gaussian maps scaled by
    their respective strength parameters.

    Args:
        resolution: ``(width, height)`` in pixels.
            Defaults to ``config.LEPTON_RESOLUTION`` (160, 120).
        column_strength: Standard deviation of column-level offsets.
        pixel_strength: Standard deviation of per-pixel offsets.
        seed: RNG seed for reproducibility.

    Returns:
        ``{"column_fpn": ndarray (H, W), "pixel_fpn": ndarray (H, W)}``
        — both float32.
    """
    if resolution is None:
        resolution = config.LEPTON_RESOLUTION

    w, h = resolution
    rng = np.random.default_rng(seed)

    # Column FPN: one offset per column, broadcast to full (H, W) shape
    col_offsets = rng.standard_normal(w).astype(np.float32) * column_strength
    column_fpn = np.broadcast_to(col_offsets, (h, w)).copy()

    # Pixel FPN: independent per-pixel
    pixel_fpn = (
        rng.standard_normal((h, w)).astype(np.float32) * pixel_strength
    )

    return {"column_fpn": column_fpn, "pixel_fpn": pixel_fpn}


def apply_fpn(
    img_float: np.ndarray,
    fpn: Dict[str, np.ndarray],
    factor: float,
) -> np.ndarray:
    """Apply combined column + pixel FPN to a float32 [0, 1] image.

    Formula::

        combined = (fpn["column_fpn"] + fpn["pixel_fpn"]) * factor
        output   = clip(img_float + combined, 0, 1)

    Must be called **after** ``agc_to_uint8()`` cast: the caller converts
    the uint8 image to float32, calls this function, then casts back to
    uint8.  FPN is a readout artifact that occurs after the optical path.

    Args:
        img_float: float32 array in [0, 1].
        fpn: Dict with keys ``"column_fpn"`` and ``"pixel_fpn"``
            (output of ``generate_lepton_fpn``).
        factor: Scaling factor for FPN amplitude (from degradation profile).

    Returns:
        float32 array clipped to [0, 1].
    """
    combined = (fpn["column_fpn"] + fpn["pixel_fpn"]) * factor
    return np.clip(img_float + combined, 0.0, 1.0).astype(np.float32)
