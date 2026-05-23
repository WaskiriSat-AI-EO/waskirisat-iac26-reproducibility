"""Reflectance normalisation for Sentinel-2 L2A BOA imagery."""

import numpy as np

from nina_cloud.datagen import config


def normalize_reflectance(
    band_dn: np.ndarray,
    scale: float = config.S2_BOA_SCALE,
) -> np.ndarray:
    """Convert S2 L2A BOA digital numbers to float32 reflectance in [0, 1].

    S2 L2A BOA DNs are stored as integers scaled by 10000.  Values above
    10000 arise from specular surfaces and are clipped to 1.0.

    Args:
        band_dn: Integer or float array of BOA DNs (any shape).
        scale: DN-to-reflectance divisor; default S2_BOA_SCALE = 10000.

    Returns:
        float32 array clipped to [0, 1].
    """
    refl = band_dn.astype(np.float32) / float(scale)
    return np.clip(refl, 0.0, 1.0).astype(np.float32)


def reflectance_to_uint8(refl_float: np.ndarray) -> np.ndarray:
    """Map float32 [0, 1] reflectance to uint8 [0, 255].

    Formula: clip(round(refl * 255), 0, 255).astype(uint8)

    Identical pattern to nina_thermal.processing.agc.agc_to_uint8; kept
    local so this module has no thermal dependency.

    Args:
        refl_float: float32 array with values in [0, 1].

    Returns:
        uint8 array with values in [0, 255].
    """
    return np.clip(np.round(refl_float * 255.0), 0, 255).astype(np.uint8)
