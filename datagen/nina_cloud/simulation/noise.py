"""Shot noise, read noise, and Fixed Pattern Noise for the PAG7936 CMOS sensor.

PAG7936 is a global shutter CMOS sensor — there is no rolling-shutter row FPN.
The FPN model has two components:

* **column_fpn** — column-level readout amplifier variation (dominant in CMOS)
* **pixel_fpn**  — residual per-pixel non-uniformity after factory calibration

Shot noise is Poisson-distributed and scales with signal (photon statistics).
Read noise is Gaussian and signal-independent.
"""

from typing import Dict

import numpy as np

from nina_cloud import config


def add_shot_noise(
    rgb_float: np.ndarray,
    factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Poisson shot noise to a float32 [0, 1] reflectance image.

    Shot noise models photon counting statistics: variance ∝ signal.

    The full-well electron count is derived from PAG7936 SNR = 36 dB:
      SNR_linear = 10^(36/10) ≈ 3981 → N_electrons ≈ 3981 at full well.

    Sampling model:
      counts = Poisson(rgb × N_eff × factor)
      output = counts / (N_eff × factor)

    where N_eff = PAG7936_FULL_WELL_ELECTRONS.  Higher factor (profiles B,
    C) inflates shot noise to model elevated dark current on-orbit.

    Args:
        rgb_float: float32 (H, W, 3) image in [0, 1].
        factor: Noise scale factor from degradation profile.
        rng: numpy.random.Generator for reproducible sampling.

    Returns:
        float32 (H, W, 3) array clipped to [0, 1].
    """
    n_eff = config.PAG7936_FULL_WELL_ELECTRONS * factor
    counts = rng.poisson(np.clip(rgb_float, 0.0, 1.0) * n_eff).astype(np.float32)
    noisy = counts / n_eff
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_read_noise(
    rgb_float: np.ndarray,
    sigma_dn: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian read noise to a float32 [0, 1] image.

    Read noise is signal-independent (CMOS readout amplifier floor).

    Args:
        rgb_float: float32 (H, W, 3) image in [0, 1].
        sigma_dn: Noise standard deviation in DN units (0–255 scale).
        rng: numpy.random.Generator for reproducible sampling.

    Returns:
        float32 (H, W, 3) array clipped to [0, 1].
    """
    sigma_norm = sigma_dn / 255.0
    noise = rng.standard_normal(rgb_float.shape).astype(np.float32) * sigma_norm
    return np.clip(rgb_float + noise, 0.0, 1.0).astype(np.float32)


def generate_pag7936_fpn(
    shape: tuple,
    column_strength: float = 0.0,
    pixel_strength: float = 0.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a two-component FPN map for the PAG7936 global shutter sensor.

    Components:
    * **column_fpn**: one random offset per column, broadcast to (H, W).
      Models column-level readout amplifier variation.
    * **pixel_fpn**: independent per-pixel offsets (H, W).
      Models residual non-uniformity.

    Unlike rolling-shutter CMOS, PAG7936 uses global shutter — there is no
    row-dependent readout timing and therefore no row FPN component.

    Args:
        shape: (H, W) image dimensions.
        column_strength: Std-dev of column offsets.
        pixel_strength: Std-dev of pixel offsets.
        seed: RNG seed for reproducibility.

    Returns:
        {"column_fpn": float32 (H, W), "pixel_fpn": float32 (H, W)}
    """
    h, w = shape
    rng = np.random.default_rng(seed)

    col_offsets = rng.standard_normal(w).astype(np.float32) * column_strength
    column_fpn  = np.broadcast_to(col_offsets, (h, w)).copy()

    pixel_fpn = rng.standard_normal((h, w)).astype(np.float32) * pixel_strength

    return {"column_fpn": column_fpn, "pixel_fpn": pixel_fpn}


def apply_fpn(
    img_float: np.ndarray,
    fpn: Dict[str, np.ndarray],
    factor: float,
) -> np.ndarray:
    """Apply combined column + pixel FPN to a float32 [0, 1] image.

    Formula: output = clip(img_float + (column_fpn + pixel_fpn) × factor, 0, 1)

    Must be called after quantisation to uint8 (and float32 conversion back):
    FPN is a readout electronics artefact that occurs after photon capture.

    Args:
        img_float: float32 array in [0, 1] (any shape compatible with fpn arrays).
        fpn: Dict with "column_fpn" and "pixel_fpn" arrays from generate_pag7936_fpn.
        factor: Amplitude scaling factor from the degradation profile.

    Returns:
        float32 array clipped to [0, 1].
    """
    combined = (fpn["column_fpn"] + fpn["pixel_fpn"]) * factor
    # Expand to (H, W, 1) so it broadcasts over RGB channels if img is (H, W, 3)
    if img_float.ndim == 3 and combined.ndim == 2:
        combined = combined[:, :, np.newaxis]
    return np.clip(img_float + combined, 0.0, 1.0).astype(np.float32)
