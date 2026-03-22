"""Noise floor validation for simulated Lepton images.

Compares measured noise in a simulated uniform-scene image against the
theoretical NETD-derived sigma.  Produces the validation figure data
for the paper's methodology section.
"""

from typing import Dict

import numpy as np

from nina_thermal import config


def validate_noise_floor(
    simulated_flat_image: np.ndarray,
    expected_netd_mk: float,
    agc_range_k: float,
) -> Dict:
    """Validate the noise floor of a simulated uniform-scene image.

    Takes a simulated image of a spatially uniform thermal scene (a flat
    field), measures the actual pixel-level noise standard deviation in
    normalised [0, 1] BT units, and compares it against the theoretical
    sigma derived from NETD.

    Theoretical sigma::

        sigma_theoretical = expected_netd_mk / (1000.0 × agc_range_k)

    The ratio ``measured / theoretical`` should equal the profile's
    ``thermal_noise_factor`` (1.0 for profile A, 2.0 for B, 4.0 for C).
    Validation passes when the ratio is within ±20 % of the nearest
    integer noise factor.

    Args:
        simulated_flat_image: uint8 or float32 (H, W) image produced by
            ``simulate_lepton()`` on a spatially uniform input.  If uint8,
            it is converted to normalised float32 before measurement.
        expected_netd_mk: NETD of the base sensor (before profile scaling)
            in millikelvin.  Typically ``config.LEPTON_NETD_MK``.
        agc_range_k: Temperature range of the AGC window [K].

    Returns:
        Dict with keys:

        ``"measured_sigma"`` (float)
            Measured noise std in normalised [0, 1] BT units.

        ``"theoretical_sigma"`` (float)
            NETD-derived theoretical sigma.

        ``"ratio"`` (float)
            ``measured_sigma / theoretical_sigma``.  Should be near the
            profile's ``thermal_noise_factor``.

        ``"pass"`` (bool)
            ``True`` when ``ratio`` is within ±20 % of the nearest
            whole-number noise factor (1, 2, or 4).

        ``"expected_netd_mk"`` (float)
            Echo of the input argument.

        ``"agc_range_k"`` (float)
            Echo of the input argument.
    """
    img = simulated_flat_image

    # Normalise to [0, 1] if uint8
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    measured_sigma = float(img.std())
    theoretical_sigma = expected_netd_mk / (1000.0 * agc_range_k)

    ratio = measured_sigma / theoretical_sigma if theoretical_sigma > 0 else float("inf")

    # Pass criterion: ratio within ±20 % of the nearest recognised noise factor
    recognised_factors = [1.0, 2.0, 4.0]
    nearest = min(recognised_factors, key=lambda f: abs(ratio - f))
    passed = abs(ratio - nearest) / nearest <= 0.20

    return {
        "measured_sigma":    measured_sigma,
        "theoretical_sigma": theoretical_sigma,
        "ratio":             ratio,
        "pass":              passed,
        "expected_netd_mk":  expected_netd_mk,
        "agc_range_k":       agc_range_k,
    }
