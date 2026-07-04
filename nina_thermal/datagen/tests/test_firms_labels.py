"""Tests for ingestion/firms_labels.py.

Covers:
- VIIRS string confidence filtering (low / nominal / high)
- MODIS numeric confidence — boundary cases at official C6/6.1 breakpoints
  (29 → low, 30 → nominal, 79 → nominal, 80 → high) to lock the correct bins
  and prevent regression to the incorrect 32/65 cut-points.
- KDTree snapping: inside-tolerance → snapped, outside → dropped.
- Cosine-corrected tolerance: tighter lon tolerance is used at non-zero lat.
- Date-window filter: time_window_days=0 same-day only, =1 includes adjacent days.
- FRP detectability filter: below floor dropped; missing column → warn + skip.
- extract_sensing_date: valid SLSTR granule ID parsed; malformed returns None.
"""

import logging
from datetime import date

import numpy as np
import pytest

from nina_thermal.datagen.ingestion.firms_labels import (
    extract_sensing_date,
    load_firms_csv,
    rasterise_firms_to_slstr,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df(**cols):
    """Build a minimal pandas DataFrame from keyword columns."""
    import pandas as pd  # noqa: PLC0415
    return pd.DataFrame(cols)


def _single_point_df(lat, lon, confidence, acq_date="2023-08-15", frp=None):
    """Return a one-row DataFrame with the minimum required FIRMS columns."""
    data = {
        "latitude": [lat],
        "longitude": [lon],
        "acq_date": [acq_date],
        "confidence": [confidence],
    }
    if frp is not None:
        data["frp"] = [frp]
    return _make_df(**data)


# 5×5 grid centred around (lat=0, lon=0), 1° spacing
_GRID_H, _GRID_W = 5, 5
_LAT_GRID = np.tile(
    np.linspace(2.0, -2.0, _GRID_H)[:, None], (1, _GRID_W)
).astype(np.float32)
_LON_GRID = np.tile(
    np.linspace(-2.0, 2.0, _GRID_W)[None, :], (_GRID_H, 1)
).astype(np.float32)
_SENSING_DATE = "2023-08-15"


# ---------------------------------------------------------------------------
# VIIRS string confidence tests
# ---------------------------------------------------------------------------

class TestVirsStringConfidence:
    """VIIRS fire products use string confidence: low / nominal / high."""

    def test_high_confidence_accepted_above_nominal_threshold(self):
        df = _single_point_df(0.0, 0.0, "high")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.any(), "high-confidence detection should pass nominal filter"

    def test_nominal_confidence_accepted_at_nominal_threshold(self):
        df = _single_point_df(0.0, 0.0, "nominal")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.any(), "nominal-confidence detection should pass nominal filter"

    def test_low_confidence_rejected_at_nominal_threshold(self):
        df = _single_point_df(0.0, 0.0, "low")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert not mask.any(), "low-confidence detection should be rejected at nominal"

    def test_low_confidence_accepted_at_low_threshold(self):
        df = _single_point_df(0.0, 0.0, "low")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="low"
        )
        assert mask.any(), "low-confidence detection should pass low filter"

    def test_high_confidence_required_at_high_threshold(self):
        df = _single_point_df(0.0, 0.0, "nominal")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="high"
        )
        assert not mask.any(), "nominal-confidence should be rejected at high threshold"

    def test_case_insensitive_confidence_values(self):
        df = _single_point_df(0.0, 0.0, "NOMINAL")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.any(), "confidence values should be matched case-insensitively"


# ---------------------------------------------------------------------------
# MODIS numeric confidence tests
# Official MODIS C6/6.1 breakpoints: low 0–29, nominal 30–79, high 80–100
# (Giglio et al., MODIS Active Fire Product User's Guide)
# ---------------------------------------------------------------------------

class TestModisNumericConfidence:
    """Verify the MODIS 0–100 numeric confidence is mapped to the correct tiers.

    Boundary cases at 29/30 and 79/80 lock the official breakpoints and
    prevent regression to the incorrect 32/65 values used before this fix.
    """

    @pytest.mark.parametrize("conf_value", [0, 15, 29])
    def test_low_confidence_range_rejected_at_nominal(self, conf_value):
        df = _single_point_df(0.0, 0.0, conf_value)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert not mask.any(), (
            f"MODIS confidence={conf_value} should map to 'low' and be "
            "rejected at the nominal threshold"
        )

    @pytest.mark.parametrize("conf_value", [30, 50, 79])
    def test_nominal_confidence_range_accepted_at_nominal(self, conf_value):
        df = _single_point_df(0.0, 0.0, conf_value)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.any(), (
            f"MODIS confidence={conf_value} should map to 'nominal' and be "
            "accepted at the nominal threshold"
        )

    @pytest.mark.parametrize("conf_value", [80, 90, 100])
    def test_high_confidence_range_accepted_at_high(self, conf_value):
        df = _single_point_df(0.0, 0.0, conf_value)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="high"
        )
        assert mask.any(), (
            f"MODIS confidence={conf_value} should map to 'high' and be "
            "accepted at the high threshold"
        )

    def test_boundary_29_is_low(self):
        """Value 29 must map to 'low', not 'nominal' — locks the 30-boundary."""
        df = _single_point_df(0.0, 0.0, 29)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert not mask.any(), "MODIS confidence=29 should be 'low', rejected at nominal"

    def test_boundary_30_is_nominal(self):
        """Value 30 must map to 'nominal', not 'low' — locks the 30-boundary."""
        df = _single_point_df(0.0, 0.0, 30)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.any(), "MODIS confidence=30 should be 'nominal', accepted at nominal"

    def test_boundary_79_is_nominal(self):
        """Value 79 must map to 'nominal', not 'high' — locks the 80-boundary."""
        df = _single_point_df(0.0, 0.0, 79)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="high"
        )
        assert not mask.any(), "MODIS confidence=79 should be 'nominal', rejected at high"

    def test_boundary_80_is_high(self):
        """Value 80 must map to 'high', not 'nominal' — locks the 80-boundary."""
        df = _single_point_df(0.0, 0.0, 80)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="high"
        )
        assert mask.any(), "MODIS confidence=80 should be 'high', accepted at high"


# ---------------------------------------------------------------------------
# KDTree snapping tests
# ---------------------------------------------------------------------------

class TestKDTreeSnapping:
    """Fire points within tolerance snap to the nearest grid cell; beyond are dropped."""

    def test_point_inside_tolerance_snaps_to_correct_cell(self):
        """A detection placed at grid cell (2, 2) — lat=0, lon=0 — should snap there."""
        df = _single_point_df(0.0, 0.0, "nominal", frp=50.0)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE,
            tolerance_km=1.5, min_confidence="nominal"
        )
        # Cell (2, 2) in the 5×5 grid centred at (0, 0)
        assert mask[2, 2], "Detection at (0,0) should snap to centre cell [2,2]"
        assert mask.sum() == 1, "Only one cell should be flagged"

    def test_point_far_outside_tolerance_is_dropped(self):
        """A detection 50 km from the nearest grid cell should be dropped."""
        # Move the point 50 km north of the grid extent (~0.45°)
        df = _single_point_df(2.5, 0.0, "nominal")  # beyond the 2° grid edge
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE,
            tolerance_km=1.5, min_confidence="nominal"
        )
        assert not mask.any(), "Detection outside tolerance should be dropped"

    def test_empty_dataframe_returns_zero_mask(self):
        """An all-low-confidence DataFrame leaves the mask all-False."""
        df = _single_point_df(0.0, 0.0, "low")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_confidence="nominal"
        )
        assert mask.dtype == bool
        assert not mask.any()


# ---------------------------------------------------------------------------
# Cosine-corrected tolerance tests
# ---------------------------------------------------------------------------

class TestCosineCorrectTolerance:
    """At non-zero latitude the longitude tolerance is tighter than the lat tolerance."""

    def test_tighter_tolerance_applied_at_high_latitude(self):
        """A point just within the naïve degree tolerance but outside the cosine-
        corrected one must be dropped.

        At lat=60° N, cos(60°)=0.5, so 1° of longitude ≈ 55.5 km.
        A detection placed 1.4 km north along longitude from the nearest cell
        crosses the tighter lon tolerance (~0.025°) but not the lat tolerance.
        We test that the detection is rejected, confirming the cosine correction
        is applied rather than a single isotropic degree tolerance.
        """
        # 3×3 grid at lat 60° N
        lat_60 = np.array([[60.0, 60.0, 60.0],
                            [59.0, 59.0, 59.0],
                            [58.0, 58.0, 58.0]], dtype=np.float32)
        lon_60 = np.array([[-1.0, 0.0, 1.0],
                            [-1.0, 0.0, 1.0],
                            [-1.0, 0.0, 1.0]], dtype=np.float32)

        import math
        tol_km = 1.5
        cos_lat = math.cos(math.radians(60.0))
        tol_deg_lon = tol_km / (111.0 * cos_lat)   # ~0.027° at 60°N

        # Place the detection just beyond tol_deg_lon east of (59°N, 0°E)
        offset_lon = tol_deg_lon * 1.1
        df = _single_point_df(59.0, 0.0 + offset_lon, "high")

        mask = rasterise_firms_to_slstr(
            df, lat_60, lon_60, _SENSING_DATE,
            tolerance_km=tol_km, min_confidence="nominal"
        )
        # The cosine-corrected (tighter) tolerance must reject this point
        assert not mask.any(), (
            "Detection just beyond cosine-corrected lon tolerance should be rejected"
        )

    def test_true_positive_within_ground_tolerance_accepted_at_high_latitude(self):
        """A detection genuinely within tolerance_km on the ground, offset purely
        in longitude at high latitude, must be ACCEPTED.

        At lat=60°N, cos(60°)=0.5, so 1° of longitude ≈ 55.5 km. A detection
        1.0 km east of the grid cell (well within the 1.5 km tolerance) is
        offset by only ~0.018° of longitude. A naive isotropic degree
        tolerance (~0.0135°, the latitude-only value) would wrongly reject
        this point since 0.018 > 0.0135 in raw degree space. Only a proper
        cosine rescale of the longitude axis correctly accepts it.
        """
        lat_60 = np.array([[60.0, 60.0, 60.0]], dtype=np.float32)
        lon_60 = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

        import math
        tol_km = 1.5
        cos_lat = math.cos(math.radians(60.0))
        true_ground_km = 1.0  # within tol_km
        offset_lon = true_ground_km / (111.0 * cos_lat)  # ~0.018°

        df = _single_point_df(60.0, 0.0 + offset_lon, "high")

        mask = rasterise_firms_to_slstr(
            df, lat_60, lon_60, _SENSING_DATE,
            tolerance_km=tol_km, min_confidence="nominal"
        )
        assert mask.any(), (
            "Detection genuinely within tolerance_km on the ground should be "
            "accepted once longitude is correctly rescaled by cos(lat)"
        )


# ---------------------------------------------------------------------------
# Date-window filter tests
# ---------------------------------------------------------------------------

class TestDateWindowFilter:
    """time_window_days=0 → same day only; =1 → includes adjacent calendar days."""

    def test_same_day_detection_included_at_window_zero(self):
        df = _single_point_df(0.0, 0.0, "nominal", acq_date=_SENSING_DATE)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, time_window_days=0
        )
        assert mask.any(), "Same-day detection must be included at time_window_days=0"

    def test_next_day_detection_excluded_at_window_zero(self):
        df = _single_point_df(0.0, 0.0, "nominal", acq_date="2023-08-16")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, time_window_days=0
        )
        assert not mask.any(), "Next-day detection must be excluded at time_window_days=0"

    def test_next_day_detection_included_at_window_one(self):
        df = _single_point_df(0.0, 0.0, "nominal", acq_date="2023-08-16")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, time_window_days=1
        )
        assert mask.any(), "Next-day detection must be included at time_window_days=1"

    def test_two_day_gap_excluded_at_window_one(self):
        df = _single_point_df(0.0, 0.0, "nominal", acq_date="2023-08-17")
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, time_window_days=1
        )
        assert not mask.any(), "Two-day gap detection must be excluded at time_window_days=1"


# ---------------------------------------------------------------------------
# FRP detectability filter tests
# ---------------------------------------------------------------------------

class TestFrpFilter:
    """Detections below min_frp_mw are dropped; missing frp column → warn + skip."""

    def test_detection_above_frp_floor_accepted(self):
        df = _single_point_df(0.0, 0.0, "nominal", frp=50.0)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_frp_mw=10.0
        )
        assert mask.any(), "Detection with FRP=50 MW should pass min_frp_mw=10 filter"

    def test_detection_below_frp_floor_rejected(self):
        df = _single_point_df(0.0, 0.0, "nominal", frp=5.0)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_frp_mw=10.0
        )
        assert not mask.any(), "Detection with FRP=5 MW should be dropped by min_frp_mw=10"

    def test_detection_at_exact_frp_floor_accepted(self):
        """Boundary: FRP == floor must be included (>= comparison)."""
        df = _single_point_df(0.0, 0.0, "nominal", frp=10.0)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_frp_mw=10.0
        )
        assert mask.any(), "Detection exactly at FRP floor should be accepted"

    def test_missing_frp_column_skips_filter_and_warns(self, caplog):
        """When frp column is absent the filter is skipped and a warning is logged."""
        df = _single_point_df(0.0, 0.0, "nominal")  # no frp column
        with caplog.at_level(logging.WARNING, logger="nina_thermal.datagen.ingestion.firms_labels"):
            mask = rasterise_firms_to_slstr(
                df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_frp_mw=10.0
            )
        assert mask.any(), "Detection should still be accepted when frp column is missing"
        assert any("frp" in r.message.lower() for r in caplog.records), (
            "A warning mentioning 'frp' should be logged when the column is absent"
        )

    def test_zero_frp_floor_disables_filter(self):
        """min_frp_mw=0.0 (the default) must not drop any detections."""
        df = _single_point_df(0.0, 0.0, "nominal", frp=0.1)
        mask = rasterise_firms_to_slstr(
            df, _LAT_GRID, _LON_GRID, _SENSING_DATE, min_frp_mw=0.0
        )
        assert mask.any(), "min_frp_mw=0.0 should not drop any detection"


# ---------------------------------------------------------------------------
# extract_sensing_date tests
# ---------------------------------------------------------------------------

class TestExtractSensingDate:
    """extract_sensing_date parses SLSTR granule IDs into ISO date strings."""

    def test_valid_slstr_granule_id(self):
        granule = "S3A_SL_1_RBT____20230815T101530_20230815T101830_20230815T123456_0180_103_051_2520_PS2_O_NR_003.SEN3"
        result = extract_sensing_date(granule)
        assert result == "2023-08-15", f"Expected '2023-08-15', got {result!r}"

    def test_minimal_timestamp_format(self):
        result = extract_sensing_date("S3B_SL_1_RBT____20231231T235959")
        assert result == "2023-12-31"

    def test_malformed_granule_id_returns_none(self):
        assert extract_sensing_date("not_a_granule_id") is None

    def test_empty_string_returns_none(self):
        assert extract_sensing_date("") is None
