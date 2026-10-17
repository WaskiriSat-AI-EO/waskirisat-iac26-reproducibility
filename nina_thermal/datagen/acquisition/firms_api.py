"""NASA FIRMS REST API active-fire query integration.

Fetches active fire detections for a given bounding box and date range,
parses the CSV response, and returns unique dates containing fire alerts
above the confidence and FRP thresholds.
"""

import io
import logging
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


# ── Label-CSV download (run_fetch_firms) ─────────────────────────────────────


def plan_firms_chunks(
    start: date,
    end: date,
    source: str = config.FIRMS_LABEL_SOURCE,
    outage_source: str = config.FIRMS_OUTAGE_SOURCE,
    outages: Optional[List[Tuple[str, str]]] = None,
    margin_days: int = 1,
) -> List[Tuple[str, date, int]]:
    """Plan area-API requests covering ``[start − margin, end + margin]``.

    Labels use FIRMS detections within ±1 day of each granule, so the window
    is widened by *margin_days* on both sides.  Days inside a Suomi-NPP
    outage are fetched from *outage_source*; every other day from *source*,
    so each date has exactly one product.  Each run of same-source days is
    cut into requests of at most ``FIRMS_API_MAX_CHUNKS`` days.

    Returns:
        ``(source, first_day, n_days)`` per request, in date order.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if outages is None:
        outages = config.FIRMS_SNPP_OUTAGES
    spans = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in outages]

    def source_for(day: date) -> str:
        return outage_source if any(a <= day <= b for a, b in spans) else source

    first = start - timedelta(days=margin_days)
    last = end + timedelta(days=margin_days)
    chunks: List[Tuple[str, date, int]] = []
    day = first
    while day <= last:
        src = source_for(day)
        n = 1
        while (
            n < config.FIRMS_API_MAX_CHUNKS
            and day + timedelta(days=n) <= last
            and source_for(day + timedelta(days=n)) == src
        ):
            n += 1
        chunks.append((src, day, n))
        day += timedelta(days=n)
    return chunks


def firms_csv_name(source: str, first_day: date, n_days: int) -> str:
    """File name used in ``$ROOT/firms/`` (``firms_{source}_{date}_{n}d.csv``)."""
    return f"firms_{source}_{first_day.isoformat()}_{n_days}d.csv"


_CSV_NAME = re.compile(
    r"^firms_(?P<src>[A-Z0-9_]+)_(?P<day>\d{4}-\d{2}-\d{2})_(?P<n>\d+)d\.csv$"
)


def parse_firms_csv_name(name: str) -> Optional[Tuple[str, date, int]]:
    """Inverse of :func:`firms_csv_name`; None for other file names."""
    m = _CSV_NAME.match(name)
    if not m:
        return None
    return m["src"], date.fromisoformat(m["day"]), int(m["n"])


def source_conflicts(
    existing_names: List[str], chunks: List[Tuple[str, date, int]]
) -> List[str]:
    """Existing CSVs whose dates overlap *chunks* but come from another product.

    Mixing products on one date double-counts FRP (v3 label bug #3), so the
    fetch refuses to add a second product to a date that already has one.
    """
    conflicts = []
    for name in existing_names:
        parsed = parse_firms_csv_name(name)
        if parsed is None:
            continue
        src, day0, n = parsed
        days = {day0 + timedelta(days=i) for i in range(n)}
        for c_src, c_day0, c_n in chunks:
            c_days = {c_day0 + timedelta(days=i) for i in range(c_n)}
            if src != c_src and days & c_days:
                conflicts.append(name)
                break
    return conflicts


def fetch_area_csv(
    firms_key: str, source: str, bbox: tuple, first_day: date, n_days: int
) -> str:
    """Fetch one FIRMS area-API CSV; raise on HTTP errors or a non-CSV reply."""
    west, south, east, north = bbox
    bbox_str = f"{west:g},{south:g},{east:g},{north:g}"
    url = (
        f"{config.FIRMS_API_BASE_URL}/{firms_key}/{source}/{bbox_str}/"
        f"{n_days}/{first_day.isoformat()}"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    text = response.text
    header = text.splitlines()[0].lower() if text.strip() else ""
    if "latitude" not in header:
        raise RuntimeError(f"FIRMS returned a non-CSV response: {text.strip()[:200]}")
    return text


def fetch_data_availability(firms_key: str) -> Dict[str, Tuple[date, date]]:
    """Return ``{source: (min_date, max_date)}`` from the FIRMS availability API."""
    url = f"{config.FIRMS_AVAILABILITY_URL}/{firms_key}/ALL"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    return {
        str(row["data_id"]): (
            date.fromisoformat(str(row["min_date"])),
            date.fromisoformat(str(row["max_date"])),
        )
        for _, row in df.iterrows()
    }


def unavailable_chunks(
    chunks: List[Tuple[str, date, int]], availability: Dict[str, Tuple[date, date]]
) -> List[Tuple[str, date, int]]:
    """Chunks whose dates fall outside their product's published availability."""
    bad = []
    for src, day0, n in chunks:
        span = availability.get(src)
        last = day0 + timedelta(days=n - 1)
        if span is None or day0 < span[0] or last > span[1]:
            bad.append((src, day0, n))
    return bad
