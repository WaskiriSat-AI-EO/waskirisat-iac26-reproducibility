"""NASA FIRMS REST API active-fire query integration.

Fetches active fire detections for a given bounding box and date range,
parses the CSV response, and returns unique dates containing fire alerts
above the confidence and FRP thresholds.
"""

import io
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd  # type: ignore

from nina_thermal.datagen import config

logger = logging.getLogger(__name__)

_CONFIDENCE_RANK = {"low": 0, "nominal": 1, "high": 2}
_VIIRS_SHORT_CONF = {"l": "low", "n": "nominal", "h": "high"}


def _normalise_confidence(conf_series: pd.Series) -> pd.Series:
    """Normalize confidence values to low/nominal/high strings."""
    if pd.api.types.is_numeric_dtype(conf_series):
        return pd.cut(
            conf_series,
            bins=[-1, 29, 79, 100],
            labels=["low", "nominal", "high"],
        ).astype(str)

    normalised = conf_series.astype(str).str.lower().str.strip()
    return normalised.map(lambda v: _VIIRS_SHORT_CONF.get(v, v))


def query_firms_fire_dates(
    firms_key: str,
    bbox: tuple,  # (west, south, east, north)
    start_date_str: str,  # YYYYMMDD
    end_date_str: str,  # YYYYMMDD
    source: str = "VIIRS_SNPP_NRT",
    min_confidence: str = "nominal",
    min_frp_mw: float = 10.0,
) -> list:
    """Query NASA FIRMS REST API and return unique fire dates (YYYYMMDD).

    Args:
        firms_key: The NASA FIRMS API Map Key.
        bbox: Bounding box as (west, south, east, north) in degrees.
        start_date_str: Start date in YYYYMMDD format.
        end_date_str: End date in YYYYMMDD format.
        source: Satellite sensor data source.
        min_confidence: Minimum confidence level ('low', 'nominal', 'high').
        min_frp_mw: Minimum Fire Radiative Power in MW.

    Returns:
        Sorted list of unique YYYYMMDD date strings with active fires.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        end_date = datetime.strptime(end_date_str, "%Y%m%d")
    except ValueError as exc:
        logger.error("Invalid date format (expected YYYYMMDD): %s", exc)
        return []

    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    min_rank = _CONFIDENCE_RANK.get(min_confidence.lower(), 1)

    fire_dates = set()
    current_date = start_date

    while current_date <= end_date:
        days_left = (end_date - current_date).days + 1
        chunk_size = min(days_left, config.FIRMS_API_MAX_CHUNKS)
        date_param = current_date.strftime("%Y-%m-%d")

        url = f"{config.FIRMS_API_BASE_URL}/{firms_key}/{source}/{bbox_str}/{chunk_size}/{date_param}"
        logger.info(
            "Querying FIRMS API: %s (range: %d days) ...", date_param, chunk_size
        )

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            csv_data = response.text
        except Exception as exc:
            logger.error("FIRMS API request failed for date %s: %s", date_param, exc)
            # Proceed to next chunk in case of temporary failure
            current_date += timedelta(days=chunk_size)
            continue

        # If the response contains error messages (e.g. invalid map key, bad format)
        # rather than valid CSV header containing 'latitude', handle it.
        if "latitude" not in csv_data.lower():
            logger.error(
                "FIRMS API returned invalid response: %s", csv_data.strip()[:200]
            )
            current_date += timedelta(days=chunk_size)
            continue

        try:
            df = pd.read_csv(io.StringIO(csv_data))
        except Exception as exc:
            logger.error("Failed to parse FIRMS CSV response: %s", exc)
            current_date += timedelta(days=chunk_size)
            continue

        if df.empty:
            current_date += timedelta(days=chunk_size)
            continue

        required_cols = {"latitude", "longitude", "acq_date", "confidence"}
        if not required_cols.issubset(df.columns):
            logger.warning("FIRMS API CSV missing columns. Found: %s", list(df.columns))
            current_date += timedelta(days=chunk_size)
            continue

        # Normalise confidence and rank filter
        conf_ranks = _normalise_confidence(df["confidence"]).map(_CONFIDENCE_RANK)
        filtered_df = df[conf_ranks >= min_rank].copy()

        # FRP filter
        if min_frp_mw > 0.0:
            if "frp" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["frp"] >= min_frp_mw]
            else:
                logger.warning(
                    "'frp' column missing in FIRMS response; skipping FRP filter."
                )

        if not filtered_df.empty:
            dates = filtered_df["acq_date"].dropna().unique()
            for d in dates:
                # Convert YYYY-MM-DD to YYYYMMDD
                fire_dates.add(str(d).replace("-", ""))

        current_date += timedelta(days=chunk_size)

    result = sorted(list(fire_dates))
    logger.info("Found active fire detections on %d unique date(s).", len(result))
    return result
