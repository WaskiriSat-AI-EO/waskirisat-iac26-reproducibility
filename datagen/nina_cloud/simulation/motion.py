"""Motion blur calculation and kernel generation for PAG7936 at LEO altitude.

At PAG7936's 562.5m GSD and 5ms exposure the along-track smear is:
  smear = 7600 m/s × 0.005 s = 38 m → 38 / 562.5 = 0.067 px

This is effectively zero.  The module is included for completeness,
reproducibility, and to support future sensors with different GSD or
longer exposure times.
"""

import numpy as np

from nina_cloud import config


def calculate_motion_blur_px(
    ground_speed_m_s: float = config.LEO_GROUND_SPEED_M_S,
    exposure_s: float = config.PAG7936_EXPOSURE_S,
    gsd_m: float = config.PAG7936_GSD_M,
) -> float:
    """Compute along-track motion blur in pixels.

    Formula: smear_px = (ground_speed × exposure) / GSD

    Args:
        ground_speed_m_s: Ground-projected satellite speed [m/s].
        exposure_s: Sensor integration time [s].
        gsd_m: Ground sampling distance [m/px].

    Returns:
        Motion blur in pixels.  For PAG7936 at nominal mission params ≈ 0.067.
    """
    return (ground_speed_m_s * exposure_s) / gsd_m


def motion_blur_kernel(length_px: float, angle_deg: float = 0.0) -> np.ndarray:
    """Build a 1-D motion blur convolution kernel.

    When length_px < 1 the motion is sub-pixel and an identity kernel (1×1)
    is returned — no blur is applied.

    Args:
        length_px: Blur length in pixels (along the motion direction).
        angle_deg: Rotation angle of the blur axis in degrees; 0 = horizontal
            (along-track for a north-facing satellite).

    Returns:
        float32 normalised 2-D kernel of shape (size, size) where
        size = ceil(length_px) | 1 (odd for symmetric convolution).
        Identity kernel (shape (1, 1)) when length_px < 1.
    """
    if length_px < 1.0:
        return np.ones((1, 1), dtype=np.float32)

    size = int(np.ceil(length_px))
    if size % 2 == 0:
        size += 1  # ensure odd size for symmetric padding

    kernel = np.zeros((size, size), dtype=np.float32)
    centre = size // 2
    kernel[centre, :] = 1.0  # horizontal line

    if angle_deg != 0.0:
        import cv2
        M = cv2.getRotationMatrix2D((centre, centre), angle_deg, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))

    total = kernel.sum()
    if total > 0:
        kernel /= total
    else:
        kernel[centre, centre] = 1.0

    return kernel
