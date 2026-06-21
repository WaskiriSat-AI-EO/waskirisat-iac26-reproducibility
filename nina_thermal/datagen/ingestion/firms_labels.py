"""Load and rasterise FIRMS active-fire CSV detections onto a SLSTR 1 km grid."""

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"low": 0, "nominal": 1, "high": 2}
_REQUIRED_COLS = {"latitude", "longitude", "acq_date", "confidence"}


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
            usecols=lambda c: c in (_REQUIRED_COLS | {"acq_time"}),
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

    return df


def rasterise_firms_to_slstr(
    firms_df,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sensing_date: str,
    tolerance_km: float = 1.5,
    time_window_days: int = 0,
    min_confidence: str = "nominal",
) -> np.ndarray:
    """Return a bool (H, W) fire mask by snapping FIRMS points to the SLSTR 1 km grid.

    Only detections within ±time_window_days of sensing_date and at or above
    min_confidence are used.  Points farther than tolerance_km from any grid
    cell centre are dropped.

    Args:
        firms_df: pandas DataFrame with at minimum columns ``latitude``,
            ``longitude``, ``acq_date``, and ``confidence``.
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
            ``"high"``) and MODIS numeric confidence (0–100 integer) —
            MODIS values are mapped to tiers as 0–32 → low, 33–65 → nominal,
            66–100 → high.
    """
    try:
        import pandas as pd  # type: ignore
        from scipy.spatial import KDTree  # type: ignore
    except ImportError:
        logger.error("scipy and pandas are required for FIRMS rasterisation.")
        return np.zeros(lat_grid.shape, dtype=bool)

    fire_mask = np.zeros(lat_grid.shape, dtype=bool)

    # Confidence filter — support both VIIRS string and MODIS numeric confidence
    min_rank = _CONFIDENCE_RANK.get(min_confidence.lower(), 1)
    conf_col = firms_df["confidence"]
    if pd.api.types.is_numeric_dtype(conf_col):
        # MODIS: numeric 0–100 → string tier mapping
        str_conf = pd.cut(
            conf_col,
            bins=[-1, 32, 65, 100],
            labels=["low", "nominal", "high"],
        )
        conf_rank = str_conf.map(_CONFIDENCE_RANK)
    else:
        conf_rank = conf_col.str.lower().map(_CONFIDENCE_RANK)
    df = firms_df[conf_rank >= min_rank].copy()
    if df.empty:
        logger.debug("No FIRMS detections at or above confidence='%s'.", min_confidence)
        return fire_mask

    # Date filter — default same-day only (time_window_days=0)
    center = date.fromisoformat(sensing_date)
    date_min = (center - timedelta(days=time_window_days)).isoformat()
    date_max = (center + timedelta(days=time_window_days)).isoformat()
    df = df[(df["acq_date"] >= date_min) & (df["acq_date"] <= date_max)]
    if df.empty:
        return fire_mask

    # KD-tree nearest-neighbour snap in degree space.
    # Use the mean latitude of the scene to compute a cosine-corrected longitude
    # tolerance so that the snap radius is approximately circular on the ground
    # rather than ellipsoidal in degree space.  This matters at Bolivia's
    # latitude (~15°S) where longitude degrees are ~cos(15°) ≈ 0.97× as wide
    # as latitude degrees; using a single degree tolerance would produce a
    # ~3 % error in the longitude direction.  We take the more conservative
    # (tighter) of the two tolerances.
    mean_lat_rad = np.deg2rad(float(np.nanmean(lat_grid)))
    cos_lat = max(float(np.cos(mean_lat_rad)), 1e-6)
    tol_deg_lat = tolerance_km / 111.0
    tol_deg_lon = tolerance_km / (111.0 * cos_lat)
    tol_deg = min(tol_deg_lat, tol_deg_lon)  # conservative (tighter circle)

    W = lat_grid.shape[1]
    grid_pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    tree = KDTree(grid_pts)
    firms_pts = np.column_stack([df["latitude"].values, df["longitude"].values])

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
        merged, lat_grid, lon_grid, sensing_date, time_window_days=time_window_days
    )
