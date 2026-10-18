"""CDSE OData catalog search for Sentinel-3 SLSTR L1B products.

Searches the Copernicus Data Space Ecosystem OData v1 catalog for
Sentinel-3 SL_1_RBT___ products intersecting a given footprint and
date range.
"""

import glob
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_COLLECTION = "SENTINEL-3"
_TIMEOUT_S = 60
_CHUNK_PAGES = 20  # max items per OData $top request


def search_slstr_products(
    access_token: str,
    footprint_wkt: str,
    start_date: str,
    end_date: str,
    product_type: str = "SL_1_RBT___",
    max_results: int = 50,
) -> List[Dict]:
    """Search the CDSE catalog for Sentinel-3 SLSTR L1B products.

    Builds an OData ``$filter`` query that constrains by:

    * Collection name ``SENTINEL-3``
    * Product type attribute (``productType`` eq *product_type*)
    * Sensing start date range
    * Geographic intersection with *footprint_wkt*

    Results are ordered by sensing start date ascending (oldest first)
    and limited to *max_results* entries.

    Args:
        access_token: Bearer token from :func:`get_access_token`.
        footprint_wkt: WKT geometry string of the area of interest
            (e.g. produced by ``sentinelsat.geojson_to_wkt``).
        start_date: Start of sensing window in ``"YYYYMMDD"`` format.
        end_date: End of sensing window in ``"YYYYMMDD"`` format.
        product_type: SLSTR product type string.
            Defaults to ``"SL_1_RBT___"``.
        max_results: Maximum number of product records to return.
            Defaults to 50.

    Returns:
        List of product metadata dicts as returned by the OData API.
        Each dict includes at minimum ``"Id"`` (UUID) and ``"Name"``.
        Returns an empty list on any failure.
    """
    try:
        start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        logger.error("Invalid date format (expected YYYYMMDD): %s", exc)
        return []

    filter_query = (
        f"Collection/Name eq '{_COLLECTION}' and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{product_type}') and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start le {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}')"
    )

    params: Dict = {
        "$filter": filter_query,
        "$top": min(max_results, _CHUNK_PAGES),
        "$orderby": "ContentDate/Start asc",
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    products: List[Dict] = []
    url: Optional[str] = _SEARCH_URL

    while url and len(products) < max_results:
        try:
            response = requests.get(
                url,
                params=params if url == _SEARCH_URL else None,
                headers=headers,
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
        except requests.HTTPError as exc:
            logger.error(
                "CDSE search HTTP error %s: %s",
                exc.response.status_code,
                exc,
            )
            break
        except Exception as exc:
            logger.error("CDSE search failed: %s", exc)
            break

        page = body.get("value", [])
        products.extend(page)

        # OData next-page link
        url = body.get("@odata.nextLink")

    result = products[:max_results]
    logger.info(
        "Found %d %s products (requested ≤ %d).",
        len(result),
        product_type,
        max_results,
    )
    return result


# ── Product selection (no network) ───────────────────────────────────────────


def granule_key(name: str) -> str:
    """Normalise a product or ``.SEN3`` directory name for comparison.

    CDSE product names already end in ``.SEN3``; inventory files list the
    directory names, which are the same.  Stripping the suffix makes either
    form (and a bare granule id) compare equal.
    """
    name = name.strip()
    return name[: -len(".SEN3")] if name.endswith(".SEN3") else name


def load_held_granules(globs: List[str]) -> Set[str]:
    """Read every ``granule_inventory.txt`` matched by *globs*.

    Args:
        globs: Glob patterns, relative to the working directory or absolute
            (e.g. ``"data*/01_raw/granule_inventory.txt"``).

    Returns:
        Set of normalised granule keys (see :func:`granule_key`).
    """
    held: Set[str] = set()
    for pattern in globs:
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.warning("--skip_inventory %s matched no file.", pattern)
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                keys = {granule_key(line) for line in fh if line.strip()}
            logger.info("Inventory %s: %d granules held.", path, len(keys))
            held |= keys
    return held


def drop_held_products(
    products: List[Dict], held: Set[str]
) -> Tuple[List[Dict], List[Dict]]:
    """Split *products* into those not yet held and those already held."""
    kept = [p for p in products if granule_key(p["Name"]) not in held]
    skipped = [p for p in products if granule_key(p["Name"]) in held]
    return kept, skipped


def _sensing_start(product: Dict) -> str:
    return product.get("ContentDate", {}).get("Start", "")


def select_products(
    products: List[Dict], max_products: int, sampling: str = "even"
) -> List[Dict]:
    """Choose at most *max_products* products from a full-window search.

    ``"oldest"`` keeps the first products in sensing order (the pre-v4
    behaviour, which left most months covering only days 1–9).  ``"even"``
    spreads the cap over the window: products are grouped by sensing day
    and taken round-robin, one per day per round, so every day with data
    contributes before any day contributes twice.

    Args:
        products: Search results (each with ``Name`` and ``ContentDate.Start``).
        max_products: Upper bound on the number returned.
        sampling: ``"even"`` or ``"oldest"``.

    Returns:
        The selected products in sensing order.
    """
    if sampling not in ("even", "oldest"):
        raise ValueError(f"sampling must be 'even' or 'oldest', got {sampling!r}")
    ordered = sorted(products, key=lambda p: (_sensing_start(p), p["Name"]))
    if sampling == "oldest" or len(ordered) <= max_products:
        return ordered[:max_products]

    by_day: Dict[str, List[Dict]] = defaultdict(list)
    for p in ordered:
        by_day[_sensing_start(p)[:10]].append(p)
    queues = [by_day[d] for d in sorted(by_day)]

    selected: List[Dict] = []
    depth = 0
    while len(selected) < max_products:
        row = [q[depth] for q in queues if depth < len(q)]
        if not row:
            break
        # A partial last round goes to days spread across the window, not
        # to the earliest ones.
        need = max_products - len(selected)
        if need < len(row):
            step = len(row) / need
            row = [row[int(i * step)] for i in range(need)]
        selected.extend(row)
        depth += 1
    return sorted(selected, key=lambda p: (_sensing_start(p), p["Name"]))


def products_per_day(products: List[Dict]) -> Dict[str, int]:
    """Count products per sensing day (``YYYY-MM-DD``), for logging."""
    counts: Dict[str, int] = defaultdict(int)
    for p in products:
        counts[_sensing_start(p)[:10]] += 1
    return dict(sorted(counts.items()))


_NAME = re.compile(
    r"^(?P<platform>S3[AB])_SL_1_RBT____(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_"
    r"(?P<created>\d{8}T\d{6})_(?P<orbit>\d{4}_\d{3}_\d{3}_\d{4})_"
    r"(?P<centre>[A-Z0-9]{3})_(?P<timeliness>[OR])_NT_(?P<baseline>\d{3})"
)


def acquisition_key(name: str) -> str:
    """Identify one SLSTR acquisition independent of how it was processed.

    The catalog can hold the same acquisition more than once: e.g. the 2019
    operational product (``…_LN2_O_NT_003``) and ESA's reprocessing
    (``…_LR1_R_NT_004``) share platform, sensing start/stop and orbit, and
    differ only in creation time, centre, timeliness and baseline.  Unknown
    name formats fall back to the full name.
    """
    m = _NAME.match(granule_key(name))
    if not m:
        return granule_key(name)
    return f"{m['platform']}_{m['start']}_{m['stop']}_{m['orbit']}"


def _version_rank(name: str) -> Tuple[int, int, str]:
    m = _NAME.match(granule_key(name))
    if not m:
        return (-1, -1, "")
    return (int(m["baseline"]), int(m["timeliness"] == "R"), m["created"])


def dedupe_acquisitions(products: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Keep one product per acquisition: highest baseline, then reprocessed
    (``R``) over operational (``O``), then the newest.

    Duplicates would put the same scene in the dataset twice, possibly in
    train and test.  Returns ``(kept, dropped)``.
    """
    best: Dict[str, Dict] = {}
    for p in products:
        key = acquisition_key(p["Name"])
        if key not in best or _version_rank(p["Name"]) > _version_rank(
            best[key]["Name"]
        ):
            best[key] = p
    kept_ids = {id(p) for p in best.values()}
    kept = [p for p in products if id(p) in kept_ids]
    dropped = [p for p in products if id(p) not in kept_ids]
    return kept, dropped
