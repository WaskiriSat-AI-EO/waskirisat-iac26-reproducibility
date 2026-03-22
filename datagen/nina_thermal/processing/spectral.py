import numpy as np
from nina_thermal import config


def apply_broadband_correction(bt_k, ambient_k=None, alpha=None):
    """Apply first-order linear broadband correction to brightness temperature.

    Corrects from SLSTR F2 narrowband (10.85 µm) to Lepton broadband (8–14 µm).
    Formula: BT_corrected = BT + alpha × max(BT − ambient_k, 0)

    Effect is zero for BT <= ambient_k (ambient pixels unchanged).
    Effect grows linearly for BT > ambient_k (fire pixels upward adjusted).
    NaN inputs produce NaN outputs.
    """
    if ambient_k is None:
        ambient_k = config.AMBIENT_REFERENCE_K
    if alpha is None:
        alpha = config.BROADBAND_ALPHA

    # Copy to float64 before in-place updates.
    # Avoids Python 3.14 LOAD_FAST_BORROW memory-reuse: any expression of the
    # form  bt_k OP scalar  may alias bt_k's buffer for large arrays, corrupting
    # it before the return expression reads it.  Fancy-index in-place writes
    # (result[mask] += ...) operate on a sub-array copy so no aliasing occurs.
    result = np.asarray(bt_k, dtype=np.float64).copy()
    with np.errstate(invalid="ignore"):
        mask = result > ambient_k
    result[mask] += alpha * (result[mask] - ambient_k)
    return result
