"""Image quality and cloud fraction validation metrics.

Used for the --validation flag in run_simulation.py to verify that simulated
images meet quality criteria before being included in the dataset.
"""

import numpy as np


def compute_snr_laplacian(img_uint8: np.ndarray) -> float:
    """Estimate image sharpness (SNR proxy) using the variance of the Laplacian.

    A higher value indicates more texture / sharper edges.  Used to flag
    images that have been over-blurred or are essentially featureless.

    Args:
        img_uint8: uint8 (H, W) or (H, W, 3) image.

    Returns:
        Variance of the Laplacian (float).
    """
    import cv2

    if img_uint8.ndim == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_entropy(img_uint8: np.ndarray) -> float:
    """Compute Shannon entropy of the pixel value histogram.

    Higher entropy indicates more information content.

    Args:
        img_uint8: uint8 array of any shape.

    Returns:
        Shannon entropy in bits.
    """
    hist, _ = np.histogram(img_uint8.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_cloud_frac_from_scl(scl_uint8: np.ndarray, cloud_classes=None) -> float:
    """Compute cloud fraction from an SCL tile.

    Convenience wrapper; mirrors compute_tile_cloud_label but returns only
    the fraction (without label inference).

    Args:
        scl_uint8: uint8 SCL array.
        cloud_classes: SCL classes counted as cloud.  Defaults to {8, 9, 10}.

    Returns:
        Cloud fraction in [0, 1].
    """
    from nina_cloud import config

    if cloud_classes is None:
        cloud_classes = config.SCL_CLOUD_CLASSES

    invalid_classes = config.SCL_INVALID_CLASSES
    flat = scl_uint8.ravel()
    valid_mask = ~np.isin(flat, list(invalid_classes))
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return 0.0
    n_cloud = int(np.isin(flat[valid_mask], list(cloud_classes)).sum())
    return n_cloud / n_valid
