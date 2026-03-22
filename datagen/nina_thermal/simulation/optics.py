"""Optical PSF blur simulation for the FLIR Lepton 3.5 lens.

Models the f/1.1 optical point-spread function as a Gaussian blur.
Applied to the uint8-rendered image (after AGC quantisation) because
optical blur acts on the spatial distribution of pixel values as they
would be recorded by the focal plane array.
"""

import cv2
import numpy as np


def apply_psf_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Apply a Gaussian PSF blur approximating the Lepton f/1.1 optics.

    Kernel size is derived from sigma to capture at least ±2σ::

        ksize = int(2 * round(2 * sigma_px) + 1)   # always odd

    Args:
        img: Input image array.  Any dtype; the same dtype is returned.
        sigma_px: Gaussian blur standard deviation in pixels.
            If ≤ 0, the input is returned unchanged.

    Returns:
        Blurred array with the same dtype and shape as *img*.
    """
    if sigma_px <= 0:
        return img

    ksize = int(2 * round(2 * sigma_px) + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)
    return blurred.astype(img.dtype)
