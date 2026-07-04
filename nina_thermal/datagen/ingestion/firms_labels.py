"""Load and rasterise FIRMS active-fire CSV detections onto a SLSTR 1 km grid."""

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"low": 0, "nominal": 1, "high": 2}
# VIIRS NRT products use single-letter codes; map to the canonical words used
# throughout this module so that MODIS and VIIRS frames can be safely merged.
_VIIRS_SHORT_CONF = {"l": "low", "n": "nominal", "h": "high"}
_REQUIRED_COLS = {"latitude", "longitude", "acq_date", "confidence"}
# Optional columns loaded when present — frp for the FRP detectability filter,
# acq_time reserved for future sub-day temporal filtering.
_OPTIONAL_COLS = {"acq_time", "frp"}


def _normalise_confidence(conf_col):
    """Return a Series of 'low'/'nominal'/'high' strings from any FIRMS confidence column.

    FIRMS delivers confidence in three formats depending on product:
    - MODIS C6/C6.1: integer 0–100  (0–29 low, 30–79 nominal, 80–100 high)
    - VIIRS standard: strings 'low' / 'nominal' / 'high'
    - VIIRS NRT:      single-letter 'l' / 'n' / 'h'

    Normalising to canonical strings in load_firms_csv means merged frames
    always have a uniform object column that rasterise_firms_to_slstr can
    map with a single .map(_CONFIDENCE_RANK) call.
    """
    import pandas as pd  # type: ignore

    if pd.api.types.is_numeric_dtype(conf_col):
        return pd.cut(
            conf_col,
            bins=[-1, 29, 79, 100],
            labels=["low", "nominal", "high"],
        ).astype(str)

    normalised = conf_col.str.lower().str.strip()
    normalised = normalised.map(lambda v: _VIIRS_SHORT_CONF.get(v, v))
    return normalised


def load_firms_csv(csv_path: Path):
    """Load a FIRMS CSV and return a pandas DataFrame, or None on failure."""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        logger.error("pandas is required for FIRMS CSV loading.")
        return None

    try:
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in (_REQUIRED_COLS | _OPTIONAL_COLS),
        )
    except Exception as exc:
        logger.error("Failed to read FIRMS CSV %s: %s", csv_path, exc)
        return None

    if not _REQUIRED_COLS.issubset(df.columns):
        logger.warning(
            "FIRMS CSV %s missing columns: %s",
            csv_path,
            _REQUIRED_COLS - set(df.columns),
        )
        return None

    df["confidence"] = _normalise_confidence(df["confidence"])
    return df


def rasterise_firms_to_slstr(
    firms_df,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sensing_date: str,
    tolerance_km: float = 1.5,
    time_window_days: int = 0,
    min_confidence: str = "nominal",
    min_frp_mw: float = 0.0,
) -> np.ndarray:
    """Return a bool (H, W) fire mask by snapping FIRMS points to the SLSTR 1 km grid.

    Only detections within ±time_window_days of sensing_date, at or above
    min_confidence, and (when the ``frp`` column is present) with FRP ≥
    min_frp_mw are used.  Points farther than tolerance_km from any grid cell
    centre are dropped.

    Args:
        firms_df: pandas DataFrame with at minimum columns ``latitude``,
            ``longitude``, ``acq_date``, and ``confidence``.  The optional
            ``frp`` column (Fire Radiative Power, MW) enables the detectability
            filter via ``min_frp_mw``.
        lat_grid: float32 (H, W) latitude array for the SLSTR granule.
        lon_grid: float32 (H, W) longitude array for the SLSTR granule.
        sensing_date: ISO date string ``YYYY-MM-DD`` of the SLSTR overpass.
        tolerance_km: Maximum distance (km) from a FIRMS point to a grid cell
            centre for it to be snapped.  Default 1.5 km (slightly above the
            1 km SLSTR pixel spacing to catch edge cases).
        time_window_days: Number of calendar days around ``sensing_date`` to
            include.  Default 0 = same day only.  Set to 1 for sparse regions
            where same-day coverage is incomplete.  Using ±1 day can introduce
            fire detections from the day before/after the overpass.
        min_confidence: Minimum confidence tier to accept.  One of
            ``"low"``, ``"nominal"`` (default), or ``"high"``.
            Handles both VIIRS string confidence (``"low"``/``"nominal"``/
            ``"high"``) and MODIS numeric confidence (0–100 integer).
            MODIS values are mapped per the MODIS Collection 6/6.1 active-fire
            product specification (Giglio et al.):
            0–29 → low, 30–79 → nominal, 80–100 → high.
        min_frp_mw: Minimum Fire Radiative Power [MW] for a detection to be
            accepted.  Default 0.0 (no FRP filter).  When set to a positive
            value (e.g. ``config.FIRMS_MIN_FRP_MW = 10.0``), detections below
            this floor are dropped — they are too weak to produce a detectable
            LWIR thermal anomaly at the 3563 m Lepton GSD and would introduce
            contradictory (unlearnable) positive labels.  Requires the ``frp``
            column to be present; if absent, a warning is logged and the filter
            is skipped.
    """
    try:
        from scipy.spatial import KDTree  # type: ignore
    except ImportError:
        logger.error("scipy is required for FIRMS rasterisation.")
        return np.zeros(lat_grid.shape, dtype=bool)

    fire_mask = np.zeros(lat_grid.shape, dtype=bool)

    # Confidence filter. load_firms_csv() already normalises this column to
    # 'low'/'nominal'/'high', but rasterise_firms_to_slstr() is also called
    # directly (tests, other callers) with un-normalised frames, so normalise
    # again here — idempotent on already-normalised input.
    min_rank = _CONFIDENCE_RANK.get(min_confidence.lower(), 1)
    conf_rank = _normalise_confidence(firms_df["confidence"]).map(_CONFIDENCE_RANK)
    df = firms_df[conf_rank >= min_rank].copy()
    if df.empty:
        logger.debug("No FIRMS detections at or above confidence='%s'.", min_confidence)
        return fire_mask

    # FRP detectability filter — drop detections too weak to produce a thermal
    # anomaly detectable by the Lepton at 3563 m GSD.
    if min_frp_mw > 0.0:
        if "frp" in df.columns:
            n_before = len(df)
            df = df[df["frp"] >= min_frp_mw]
            n_dropped = n_before - len(df)
            if n_dropped:
                logger.debug(
                    "FRP filter (>= %.1f MW): dropped %d/%d detections.",
                    min_frp_mw,
                    n_dropped,
                    n_before,
                )
            if df.empty:
                logger.debug("All detections dropped by FRP filter.")
                return fire_mask
        else:
            logger.warning(
                "min_frp_mw=%.1f requested but 'frp' column absent — filter skipped.",
                min_frp_mw,
            )

    # Date filter — default same-day only (time_window_days=0)
    center = date.fromisoformat(sensing_date)
    date_min = (center - timedelta(days=time_window_days)).isoformat()
    date_max = (center + timedelta(days=time_window_days)).isoformat()
    df = df[(df["acq_date"] >= date_min) & (df["acq_date"] <= date_max)]
    if df.empty:
        return fire_mask

    # KD-tree nearest-neighbour snap in degree space.
    # Longitude degrees are narrower than latitude degrees away from the
    # equator (1° lon ≈ 111*cos(lat) km vs. 1° lat ≈ 111 km) — this matters at
    # Bolivia's latitude (~15°S, cos(15°) ≈ 0.97).  The KD-tree computes plain
    # Euclidean distance in degree space, so to get an approximately circular
    # ground-distance tolerance from a single scalar radius, rescale longitude
    # into "equivalent latitude-degree" units (lon * cos_lat) before building
    # the tree, and use the latitude-only tolerance as the isotropic radius in
    # that rescaled space: ground_dist_lon = Δlon_deg*111*cos_lat =
    # (Δlon_deg*cos_lat)*111, i.e. the same km-per-degree scale as latitude.
    mean_lat_rad = np.deg2rad(float(np.nanmean(lat_grid)))
    cos_lat = max(float(np.cos(mean_lat_rad)), 1e-6)
    tol_deg = tolerance_km / 111.0

    W = lat_grid.shape[1]
    grid_pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel() * cos_lat])
    tree = KDTree(grid_pts)
    firms_pts = np.column_stack(
        [df["latitude"].values, df["longitude"].values * cos_lat]
    )

    dists, idxs = tree.query(firms_pts, distance_upper_bound=tol_deg)
    valid = dists < tol_deg

    for flat_idx in idxs[valid]:
        r, c = divmod(int(flat_idx), W)
        fire_mask[r, c] = True

    return fire_mask


def extract_sensing_date(granule_id: str) -> Optional[str]:
    """Extract YYYY-MM-DD from an SLSTR granule ID (e.g. S3A_SL_1_RBT____20230601T…)."""
    m = re.search(r"(\d{8})T\d{6}", granule_id)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return None


def load_firms_for_granule(
    firms_dir: Path,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sensing_date: str,
    time_window_days: int = 0,
    min_confidence: str = "nominal",
    min_frp_mw: float = 0.0,
) -> Optional[np.ndarray]:
    """Find all FIRMS CSVs in firms_dir that overlap sensing_date and rasterise.

    Args:
        firms_dir: Directory containing FIRMS CSV files.
        lat_grid: float32 (H, W) latitude array for the SLSTR granule.
        lon_grid: float32 (H, W) longitude array for the SLSTR granule.
        sensing_date: ISO date string ``YYYY-MM-DD`` of the SLSTR overpass.
        time_window_days: Calendar-day window around ``sensing_date``.
            Default 0 = same day only.  The previous default of ±1 day
            introduced fire detections from the day before/after the overpass,
            creating false-positive labels for fires that had started or ended
            by the time of the SLSTR overpass.
        min_confidence: Minimum FIRMS confidence tier.  Forwarded to
            :func:`rasterise_firms_to_slstr`.  Default ``"nominal"``.
        min_frp_mw: Minimum Fire Radiative Power [MW].  Forwarded to
            :func:`rasterise_firms_to_slstr`.  Default 0.0 = no filter.

    Returns:
        A bool (H, W) fire mask, or None if no matching detections were found.
    """
    import pandas as pd  # type: ignore

    firms_dir = Path(firms_dir)
    csv_files = sorted(firms_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No .csv files found in FIRMS dir %s.", firms_dir)
        return None

    center = date.fromisoformat(sensing_date)
    date_min = (center - timedelta(days=time_window_days)).isoformat()
    date_max = (center + timedelta(days=time_window_days)).isoformat()

    frames = []
    for csv_path in csv_files:
        df = load_firms_csv(csv_path)
        if df is None:
            continue
        df_win = df[(df["acq_date"] >= date_min) & (df["acq_date"] <= date_max)]
        if not df_win.empty:
            frames.append(df_win)

    if not frames:
        logger.debug(
            "No FIRMS detections in date window %s ± %dd.", sensing_date, time_window_days
        )
        return None

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "  FIRMS: %d detection(s) in window %s ± %dd.",
        len(merged),
        sensing_date,
        time_window_days,
    )

    return rasterise_firms_to_slstr(
        merged,
        lat_grid,
        lon_grid,
        sensing_date,
        time_window_days=time_window_days,
        min_confidence=min_confidence,
        min_frp_mw=min_frp_mw,
    )
