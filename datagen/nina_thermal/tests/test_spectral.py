import numpy as np
import pytest
from nina_thermal import config
from nina_thermal.processing.spectral import apply_broadband_correction

AMBIENT = config.AMBIENT_REFERENCE_K  # 295.0 K


def test_correction_zero_at_ambient():
    """Correction is zero at and below ambient temperature."""
    bt_at_ambient = np.array([AMBIENT - 10.0, AMBIENT, AMBIENT - 0.001])
    corrected = apply_broadband_correction(bt_at_ambient)
    np.testing.assert_array_equal(corrected, bt_at_ambient)


def test_correction_positive_for_fire():
    """Correction is positive (upward) for fire temperatures above ambient."""
    fire_bt = 380.0
    corrected = float(apply_broadband_correction(fire_bt))
    assert (
        corrected > fire_bt
    ), f"Expected correction > 0 at {fire_bt} K, got {corrected - fire_bt:.4f} K"
    expected_delta = config.BROADBAND_ALPHA * (fire_bt - AMBIENT)
    np.testing.assert_allclose(corrected, fire_bt + expected_delta, rtol=1e-9)


def test_nan_safe():
    """NaN inputs produce NaN outputs; finite neighbours are unaffected."""
    bt = np.array([np.nan, AMBIENT, 380.0, np.nan])
    corrected = apply_broadband_correction(bt)
    assert np.isnan(corrected[0]), "NaN input should give NaN output"
    assert np.isfinite(corrected[1]), "Finite ambient should give finite output"
    assert np.isfinite(corrected[2]), "Finite fire temp should give finite output"
    assert np.isnan(corrected[3]), "NaN input should give NaN output"
