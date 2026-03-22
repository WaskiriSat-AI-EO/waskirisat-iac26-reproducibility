"""Pre-flight data quality checks for a loaded SLSTR granule dict.

Validates that the dict returned by :func:`load_slstr_granule` is
well-formed before it enters the processing pipeline.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("F2_BT", "S8_BT", "delta_T", "lat", "lon")
_MIN_FINITE_FRACTION = 0.20


def validate_granule(granule: Dict[str, np.ndarray]) -> bool:
    """Run pre-flight checks on a granule dict.

    Checks performed (in order):

    1. All five required keys are present:
       ``"F2_BT"``, ``"S8_BT"``, ``"delta_T"``, ``"lat"``, ``"lon"``.
    2. All arrays share the same shape.
    3. All arrays have ``dtype == float32``.
    4. ``F2_BT`` has a finite-value fraction > 0.20 (granule has usable
       data). SLSTR nadir arrays include large buffer zones and scan gaps
       outside the field of view filled with NaN, so fractions around
       0.35–0.45 are normal and not indicative of a bad granule.

    Args:
        granule: Dict produced by :func:`load_slstr_granule`.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
        Failures are logged at WARNING level with a specific reason.

    Example::

        >>> import numpy as np
        >>> g = {
        ...     "F2_BT":   np.full((10, 10), 300.0, dtype=np.float32),
        ...     "S8_BT":   np.full((10, 10), 298.0, dtype=np.float32),
        ...     "delta_T": np.full((10, 10),   2.0, dtype=np.float32),
        ...     "lat":     np.zeros((10, 10), dtype=np.float32),
        ...     "lon":     np.zeros((10, 10), dtype=np.float32),
        ... }
        >>> validate_granule(g)
        True

        >>> bad = dict(g)
        >>> del bad["delta_T"]
        >>> validate_granule(bad)
        False
    """
    # 1. Required keys
    for key in _REQUIRED_KEYS:
        if key not in granule:
            logger.warning("validate_granule: missing key '%s'", key)
            return False

    arrays = [granule[k] for k in _REQUIRED_KEYS]
    ref_shape = arrays[0].shape

    # 2. Shape consistency
    for key, arr in zip(_REQUIRED_KEYS, arrays):
        if arr.shape != ref_shape:
            logger.warning(
                "validate_granule: shape mismatch — '%s' is %s, expected %s",
                key, arr.shape, ref_shape,
            )
            return False

    # 3. dtype
    for key, arr in zip(_REQUIRED_KEYS, arrays):
        if arr.dtype != np.float32:
            logger.warning(
                "validate_granule: dtype mismatch — '%s' is %s, expected float32",
                key, arr.dtype,
            )
            return False

    # 4. Finite fraction
    f2 = granule["F2_BT"]
    finite_fraction = float(np.isfinite(f2).mean())
    if finite_fraction <= _MIN_FINITE_FRACTION:
        logger.warning(
            "validate_granule: finite fraction %.3f <= %.3f "
            "(granule too masked; SLSTR buffer zones account for ~50-60%% of array)",
            finite_fraction, _MIN_FINITE_FRACTION,
        )
        return False

    return True
