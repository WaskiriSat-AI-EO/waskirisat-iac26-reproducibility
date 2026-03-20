import numpy as np
from nina_thermal import config


def bt_to_radiance(bt_k, wavelength_m):
    """Convert brightness temperature [K] to spectral radiance [W sr⁻¹ m⁻² m⁻¹].

    Uses Planck's law. NaN inputs produce NaN outputs.
    No clipping — caller is responsible for valid input range.
    """
    bt_k = np.asarray(bt_k, dtype=np.float64)
    c1 = 2.0 * config.H_PLANCK * config.C_LIGHT**2
    c2 = config.H_PLANCK * config.C_LIGHT / config.K_BOLTZMANN
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        L = c1 / (wavelength_m**5 * (np.exp(c2 / (wavelength_m * bt_k)) - 1.0))
    return L


def radiance_to_bt(L, wavelength_m):
    """Inverse Planck: spectral radiance [W sr⁻¹ m⁻² m⁻¹] → brightness temperature [K].

    NaN inputs and non-positive L produce NaN outputs.
    Handles overflow in exp() gracefully via np.errstate.
    """
    L = np.asarray(L, dtype=np.float64)
    c1 = 2.0 * config.H_PLANCK * config.C_LIGHT**2
    c2 = config.H_PLANCK * config.C_LIGHT / config.K_BOLTZMANN
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        arg = c1 / (wavelength_m**5 * L) + 1.0
        bt = c2 / (wavelength_m * np.log(arg))
    # Non-positive L or NaN already propagate through; enforce NaN for L <= 0
    bt = np.where(np.asarray(L) <= 0, np.nan, bt)
    return bt
