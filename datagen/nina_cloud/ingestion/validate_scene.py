"""Pre-flight data quality checks for a loaded S2 scene dict."""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

_MIN_FINITE_FRACTION = 0.50


def validate_scene(scene: Dict[str, np.ndarray]) -> bool:
    """Run pre-flight checks on a loaded S2 scene dict.

    Checks performed (in order):

    1. Required keys present: ``"rgb"`` and ``"scl"``.
    2. ``rgb`` has shape (H, W, 3) and ``scl`` has shape (H//2, W//2)
       (S2 BOA at 10m, SCL at 20m — half resolution).
    3. ``rgb`` dtype is float32; ``scl`` dtype is uint8.
    4. ``rgb`` finite fraction > MIN_FINITE_FRACTION.

    Args:
        scene: Dict produced by :func:`load_s2_scene`.

    Returns:
        True if all checks pass, False otherwise.
        Failures are logged at WARNING level with a specific reason.
    """
    for key in ("rgb", "scl"):
        if key not in scene:
            logger.warning("validate_scene: missing key '%s'", key)
            return False

    rgb = scene["rgb"]
    scl = scene["scl"]

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        logger.warning(
            "validate_scene: rgb must be (H, W, 3), got %s", rgb.shape
        )
        return False

    h_rgb, w_rgb = rgb.shape[:2]
    expected_scl_shape = (h_rgb // 2, w_rgb // 2)
    if scl.shape != expected_scl_shape:
        logger.warning(
            "validate_scene: scl shape %s does not match expected %s "
            "(half of rgb %s)",
            scl.shape, expected_scl_shape, (h_rgb, w_rgb),
        )
        return False

    if rgb.dtype != np.float32:
        logger.warning(
            "validate_scene: rgb dtype is %s, expected float32", rgb.dtype
        )
        return False

    if scl.dtype != np.uint8:
        logger.warning(
            "validate_scene: scl dtype is %s, expected uint8", scl.dtype
        )
        return False

    finite_frac = float(np.isfinite(rgb).mean())
    if finite_frac < _MIN_FINITE_FRACTION:
        logger.warning(
            "validate_scene: rgb finite fraction %.3f < %.3f",
            finite_frac, _MIN_FINITE_FRACTION,
        )
        return False

    logger.info(
        "validate OK: rgb=%s  scl=%s  dtype=float32/uint8  finite_frac=%.3f",
        rgb.shape, scl.shape, finite_frac,
    )
    return True
