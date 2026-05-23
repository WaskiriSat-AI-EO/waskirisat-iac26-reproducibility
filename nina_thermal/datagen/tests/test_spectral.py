import numpy as np
import pytest
from nina_thermal.datagen import config
from nina_thermal.datagen.processing.spectral import apply_broadband_correction

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


def test_correction_correct_on_large_array():
    """
    Regression test for Python 3.14 LOAD_FAST_BORROW memory reuse bug.
    Must use an array >= 200x200 to trigger the original failure mode.
    """
    # 300x400 array — above the ~200x200 threshold that triggered the bug
    bt = np.full((300, 400), 350.0, dtype=np.float32)  # fire temperature
    result = apply_broadband_correction(bt)

    # At 350 K with alpha=0.015, ambient=295 K:
    # delta = 350 - 295 = 55 K
    # correction = 0.015 * 55 = 0.825 K
    # expected = 350 + 0.825 = 350.825 K
    expected = 350.0 + config.BROADBAND_ALPHA * (350.0 - config.AMBIENT_REFERENCE_K)
    assert np.allclose(
        result, expected, atol=0.01
    ), f"Expected {expected:.3f} K, got {result.mean():.3f} K — possible buffer aliasing"

    # Also verify ambient pixels are unchanged (correction = 0)
    bt_ambient = np.full((300, 400), 295.0, dtype=np.float32)
    result_ambient = apply_broadband_correction(bt_ambient)
    assert np.allclose(
        result_ambient, 295.0, atol=0.01
    ), "Ambient pixels should not be modified"
