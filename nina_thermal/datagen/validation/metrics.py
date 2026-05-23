"""Image quality metrics for dataset validation.

Used by pipeline scripts and the paper's validation figures.
These are pure functions with no file I/O (Rule 2).
"""

from typing import Dict

import numpy as np


def calculate_snr_db(image: np.ndarray) -> float:
    """Estimate image SNR in dB using a Laplacian-based noise estimator.

    The Laplacian estimator is robust to scene content: it measures
    high-frequency energy (noise) independently of the low-frequency
    signal structure.  The signal power is estimated as the total image
    variance minus the estimated noise variance.

    Estimator (from Immerkær 1996)::

        sigma_noise = sqrt(pi/2) * mean(|Laplacian(I)|) / 6
        SNR_dB = 10 * log10(signal_var / noise_var)
                 where signal_var = max(total_var - noise_var, epsilon)

    Args:
        image: 2-D numeric array (any dtype).  Float images are used as-is;
            uint8 images are normalised to [0, 1] before estimation.

    Returns:
        SNR estimate in decibels.  Returns 0.0 if the image is uniform
        (zero variance) or noise dominates entirely.
    """
    img = image.astype(np.float64)
    if img.max() > 1.0 and image.dtype == np.uint8:
        img = img / 255.0

    total_var = img.var()
    if total_var == 0.0:
        return 0.0

    # Laplacian kernel
    laplacian = _laplacian(img)
    # Immerkær estimator for sigma of additive white Gaussian noise
    sigma_noise = np.sqrt(np.pi / 2.0) * np.mean(np.abs(laplacian)) / 6.0
    noise_var = sigma_noise**2

    signal_var = max(total_var - noise_var, 1e-12)
    return float(10.0 * np.log10(signal_var / noise_var))


def calculate_histogram_stats(image: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics of pixel value distribution.

    Operates on finite values only; NaN and Inf are excluded.

    Args:
        image: Numeric array of any shape and dtype.

    Returns:
        Dict with keys:
            ``"mean"``, ``"std"``, ``"min"``, ``"max"``,
            ``"p2"`` (2nd percentile), ``"p25"``, ``"p50"`` (median),
            ``"p75"``, ``"p98"`` (98th percentile).
        All values are Python floats.  If there are no finite values,
        all entries are ``float("nan")``.
    """
    flat = image.ravel().astype(np.float64)
    finite = flat[np.isfinite(flat)]

    if finite.size == 0:
        nan = float("nan")
        return {
            k: nan
            for k in ("mean", "std", "min", "max", "p2", "p25", "p50", "p75", "p98")
        }

    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p2": float(np.percentile(finite, 2)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p98": float(np.percentile(finite, 98)),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _laplacian(img: np.ndarray) -> np.ndarray:
    """Apply discrete Laplacian kernel to a 2-D float64 image.

    Uses the 3×3 kernel::

        [ 0  1  0 ]
        [ 1 -4  1 ]
        [ 0  1  0 ]

    Boundary rows/columns are excluded (valid convolution).

    Args:
        img: 2-D float64 array.

    Returns:
        2-D float64 array of Laplacian responses (interior pixels only).
    """
    # Manual valid convolution avoids a scipy dependency
    c = img[1:-1, 1:-1]
    lap = (
        img[:-2, 1:-1]  # top
        + img[2:, 1:-1]  # bottom
        + img[1:-1, :-2]  # left
        + img[1:-1, 2:]  # right
        - 4.0 * c
    )
    return lap
