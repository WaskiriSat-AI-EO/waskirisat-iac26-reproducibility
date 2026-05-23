"""Optical PSF simulation for the PAG7936 sensor at 562.5m GSD.

The PAG7936 has a 2.8mm f/2.0 lens at 525km SSO altitude, yielding a GSD of
562.5m/px.  At S2 10m resolution, this corresponds to a 56.25× scale ratio.

The Gaussian PSF sigma is derived as:

    sigma = GSD_ratio / (2 × sqrt(2)) ≈ 19.9 S2 pixels

This suppresses spatial frequencies above the PAG7936 Nyquist limit
(1 / (2 × 562.5m) = 8.9 × 10⁻⁴ cycles/m) when applied to S2 10m imagery.

Applied FIRST in the degradation chain — optical averaging acts on the
incoming scene radiance/reflectance before any electronic noise or ADC step.
"""

from typing import Optional

import numpy as np

from nina_cloud.datagen import config
from shared.optics import apply_psf_blur


def apply_pag7936_psf(
    rgb_float: np.ndarray,
    sigma_px: Optional[float] = None,
) -> np.ndarray:
    """Apply Gaussian PSF blur simulating the PAG7936 optics at S2 10m scale.

    Wraps nina_thermal's apply_psf_blur with the PAG7936-calibrated sigma.
    The same function handles large and small sigma values correctly.

    Physical rationale: blurring at σ ≈ 20 S2 pixels is equivalent to
    integrating scene radiance over a 562.5m GSD footprint, correctly
    suppressing sub-GSD spatial frequencies before sensor noise is added.

    Args:
        rgb_float: float32 (H, W, 3) BOA reflectance at S2 10m resolution.
        sigma_px: PSF Gaussian sigma in S2 pixels.  Defaults to
            config.PAG7936_PSF_SIGMA_S2_PX (≈ 19.9 px).

    Returns:
        float32 (H, W, 3) blurred reflectance array.
    """
    if sigma_px is None:
        sigma_px = config.PAG7936_PSF_SIGMA_S2_PX

    blurred = apply_psf_blur(rgb_float, sigma_px=sigma_px)
    return blurred.astype(np.float32)
