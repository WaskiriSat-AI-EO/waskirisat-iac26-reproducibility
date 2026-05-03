"""Tests for labeling/fire_label.py — Section 8 of CLAUDE_CODE_BRIEF.md."""

import numpy as np
import pytest

from nina_thermal.labeling.fire_label import compute_patch_label, classify_patch_surface_type
from nina_thermal import config


# Convenience constants
_H, _W = 20, 20
_AMBIENT_K  = config.AMBIENT_REFERENCE_K        # 295 K
_FIRE_F2    = config.FIRE_F2_ABS_THRESHOLD_K   # 310 K
_FIRE_DT    = config.FIRE_DELTA_T_THRESHOLD_K  # effectively -inf (disabled)
_WARM_THRESH = config.WARM_SURFACE_THRESH_K    # 310 K


def _ambient_patch():
    """(H, W) patch at ambient temperature — no fire pixels."""
    return np.full((_H, _W), _AMBIENT_K, dtype=np.float32)


def _ambient_delta():
    """(H, W) delta_T patch well below the fire threshold."""
    return np.full((_H, _W), 1.0, dtype=np.float32)


class TestComputePatchLabel:
    def test_all_ambient_patch_gives_label_0(self):
        """A purely ambient patch must receive label 0."""
        f2 = _ambient_patch()
        dt = _ambient_delta()
        assert compute_patch_label(f2, dt) == 0

    def test_enough_fire_pixels_gives_label_1(self):
        """A patch with >0.5 % of pixels meeting both criteria must get label 1.

        We set 2 % of pixels (> MIN_FIRE_FRACTION=0.5 %) to fire-level
        temperatures to guarantee the label flips to 1.
        """
        f2 = _ambient_patch()
        dt = _ambient_delta()

        # Mark 2 % of pixels as fire (dt threshold is -inf so any finite value qualifies)
        n_fire = max(1, int(0.02 * _H * _W))
        f2.ravel()[:n_fire] = _FIRE_F2 + 10.0   # above absolute threshold
        dt.ravel()[:n_fire] = 5.0               # finite → satisfies disabled -inf criterion

        assert compute_patch_label(f2, dt) == 1

    def test_warm_pixels_below_delta_t_give_label_0(self):
        """Pixels that are warm but do NOT meet the delta-T criterion stay label 0.

        This tests the dual-criterion logic: high F2_BT alone is not enough.
        """
        f2 = np.full((_H, _W), _FIRE_F2 + 20.0, dtype=np.float32)  # warm
        dt = np.full((_H, _W), _FIRE_DT - 1.0,  dtype=np.float32)  # below threshold

        assert compute_patch_label(f2, dt) == 0

    def test_312k_pixel_gives_label_1_with_lowered_threshold(self):
        """312 K mixed pixel (previously missed at 320 K) must now label as fire.

        A 1 km pixel with ~10% flaming coverage at 700 K + 300 K background
        yields ~312-318 K mixed BT.  This was silently rejected at the old
        320 K threshold; the corrected 310 K threshold must recover it.
        """
        f2 = _ambient_patch()
        dt = _ambient_delta()
        n_fire = max(1, int(0.02 * _H * _W))
        f2.ravel()[:n_fire] = 312.0   # above 310 K, below old 320 K threshold
        assert compute_patch_label(f2, dt) == 1

    def test_nan_pixels_do_not_crash(self):
        """Patches containing NaN values must not raise any exception."""
        f2 = _ambient_patch()
        dt = _ambient_delta()

        # Scatter NaNs across both arrays
        f2[0, :] = np.nan
        dt[:, 0] = np.nan

        # Should return 0 or 1 without raising
        result = compute_patch_label(f2, dt)
        assert result in (0, 1)


class TestClassifyPatchSurfaceType:
    def test_label_1_returns_fire(self):
        f2 = _ambient_patch()
        dt = _ambient_delta()
        assert classify_patch_surface_type(f2, dt, label=1) == "fire"

    def test_warm_surface_when_f2_above_thresh_and_label_0(self):
        """Patch whose max F2_BT exceeds WARM_SURFACE_THRESH_K gets 'warm_surface'."""
        f2 = np.full((_H, _W), _WARM_THRESH + 5.0, dtype=np.float32)
        dt = _ambient_delta()
        assert classify_patch_surface_type(f2, dt, label=0) == "warm_surface"

    def test_nominal_for_cool_patch(self):
        f2 = _ambient_patch()
        dt = _ambient_delta()
        assert classify_patch_surface_type(f2, dt, label=0) == "nominal"
