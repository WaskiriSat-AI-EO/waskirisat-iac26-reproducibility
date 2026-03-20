import numpy as np
import pytest
from nina_thermal import config
from nina_thermal.processing.radiometry import bt_to_radiance, radiance_to_bt

WAVELENGTH = config.SLSTR_F2_WAVELENGTH_M  # 10.85e-6 m


def test_round_trip():
    """BT → radiance → BT within 0.01 K for T in [250, 400] K."""
    temps = np.linspace(250, 400, 50)
    radiances = bt_to_radiance(temps, WAVELENGTH)
    recovered = radiance_to_bt(radiances, WAVELENGTH)
    np.testing.assert_allclose(recovered, temps, atol=0.01)


def test_known_planck_value():
    """At T=300 K, λ=10.85 µm, L ≈ 9.88 W sr⁻¹ m⁻² µm⁻¹.

    Function returns W/sr/m²/m; divide by 1e6 to get W/sr/m²/µm.
    Tolerance of 5% accounts for rounded physical constants in config.
    """
    L = bt_to_radiance(300.0, WAVELENGTH)
    L_per_um = float(L) * 1e-6  # convert W/sr/m²/m → W/sr/m²/µm
    assert (
        abs(L_per_um - 9.88) / 9.88 < 0.05
    ), f"Expected ~9.88 W/sr/m²/µm, got {L_per_um:.4f}"


def test_nan_propagation():
    """NaN input → NaN output in both directions."""
    nan_arr = np.array([np.nan, 300.0, np.nan])

    L = bt_to_radiance(nan_arr, WAVELENGTH)
    assert np.isnan(L[0]), "bt_to_radiance: NaN input should give NaN output"
    assert np.isfinite(L[1]), "bt_to_radiance: valid input should give finite output"
    assert np.isnan(L[2]), "bt_to_radiance: NaN input should give NaN output"

    L_with_nan = np.array([np.nan, bt_to_radiance(300.0, WAVELENGTH), np.nan])
    bt = radiance_to_bt(L_with_nan, WAVELENGTH)
    assert np.isnan(bt[0]), "radiance_to_bt: NaN input should give NaN output"
    assert np.isfinite(bt[1]), "radiance_to_bt: valid input should give finite output"
    assert np.isnan(bt[2]), "radiance_to_bt: NaN input should give NaN output"

    # Non-positive L → NaN
    bt_neg = radiance_to_bt(np.array([-1.0, 0.0]), WAVELENGTH)
    assert np.all(np.isnan(bt_neg)), "radiance_to_bt: non-positive L should give NaN"
