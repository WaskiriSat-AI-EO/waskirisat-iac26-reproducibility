"""CDSE OData catalog search for Sentinel-3 SLSTR L1B products.

Searches the Copernicus Data Space Ecosystem OData v1 catalog for
Sentinel-3 SL_1_RBT___ products intersecting a given footprint and
date range.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_SEARCH_URL  = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_COLLECTION  = "SENTINEL-3"
_TIMEOUT_S   = 60
_CHUNK_PAGES = 20   # max items per OData $top request


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

    Results are ordered by sensing start date descending (most recent
    first) and limited to *max_results* entries.

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
        end   = datetime.strptime(end_date,   "%Y%m%d").strftime("%Y-%m-%d")
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
        "$filter":  filter_query,
        "$top":     min(max_results, _CHUNK_PAGES),
        "$orderby": "ContentDate/Start desc",
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    products: List[Dict] = []
    url: Optional[str] = _SEARCH_URL

    while url and len(products) < max_results:
        try:
            response = requests.get(
                url, params=params if url == _SEARCH_URL else None,
                headers=headers, timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
        except requests.HTTPError as exc:
            logger.error(
                "CDSE search HTTP error %s: %s",
                exc.response.status_code, exc,
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
        len(result), product_type, max_results,
    )
    return result
