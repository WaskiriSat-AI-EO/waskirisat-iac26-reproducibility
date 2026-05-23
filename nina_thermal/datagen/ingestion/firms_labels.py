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
    time_window_days: int = 1,
    min_confidence: str = "nominal",
) -> np.ndarray:
    """Return a bool (H, W) fire mask by snapping FIRMS points to the SLSTR 1 km grid.

    Only detections within ±time_window_days of sensing_date and at or above
    min_confidence are used.  Points farther than tolerance_km from any grid
    cell centre are dropped.
    """
    try:
        from scipy.spatial import KDTree  # type: ignore
    except ImportError:
        logger.error("scipy is required for FIRMS rasterisation.")
        return np.zeros(lat_grid.shape, dtype=bool)

    fire_mask = np.zeros(lat_grid.shape, dtype=bool)

    # Confidence filter
    min_rank = _CONFIDENCE_RANK.get(min_confidence.lower(), 1)
    conf_rank = firms_df["confidence"].str.lower().map(_CONFIDENCE_RANK)
    df = firms_df[conf_rank >= min_rank].copy()
    if df.empty:
        return fire_mask

    # Date filter
    center = date.fromisoformat(sensing_date)
    date_min = (center - timedelta(days=time_window_days)).isoformat()
    date_max = (center + timedelta(days=time_window_days)).isoformat()
    df = df[(df["acq_date"] >= date_min) & (df["acq_date"] <= date_max)]
    if df.empty:
        return fire_mask

    # KD-tree nearest-neighbour snap (degrees; 1° lat ≈ 111 km)
    W = lat_grid.shape[1]
    grid_pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    tree = KDTree(grid_pts)
    firms_pts = np.column_stack([df["latitude"].values, df["longitude"].values])
    tol_deg = tolerance_km / 111.0

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
) -> Optional[np.ndarray]:
    """Find all FIRMS CSVs in firms_dir that overlap sensing_date ± 1 day and rasterise.

    Returns a bool (H, W) fire mask, or None if no matching detections were found.
    """
    import pandas as pd  # type: ignore

    firms_dir = Path(firms_dir)
    csv_files = sorted(firms_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No .csv files found in FIRMS dir %s.", firms_dir)
        return None

    center = date.fromisoformat(sensing_date)
    date_min = (center - timedelta(days=1)).isoformat()
    date_max = (center + timedelta(days=1)).isoformat()

    frames = []
    for csv_path in csv_files:
        df = load_firms_csv(csv_path)
        if df is None:
            continue
        df_win = df[(df["acq_date"] >= date_min) & (df["acq_date"] <= date_max)]
        if not df_win.empty:
            frames.append(df_win)

    if not frames:
        logger.debug("No FIRMS detections in date window %s ± 1d.", sensing_date)
        return None

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "  FIRMS: %d detection(s) in window %s ± 1d.", len(merged), sensing_date
    )

    return rasterise_firms_to_slstr(merged, lat_grid, lon_grid, sensing_date)
