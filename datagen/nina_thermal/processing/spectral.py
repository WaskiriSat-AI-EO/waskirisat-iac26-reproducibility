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

    bt_k = np.asarray(bt_k, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        delta = np.maximum(bt_k - ambient_k, 0.0)
    return bt_k + alpha * delta
