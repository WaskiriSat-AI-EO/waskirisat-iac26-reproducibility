"""CDSE OData catalog search for Sentinel-2 L2A products.

Searches the Copernicus Data Space Ecosystem OData v1 catalog for
Sentinel-2 S2MSI2A products intersecting a given footprint and date range,
with a dual cloud-cover strategy to acquire both clear and cloudy scenes.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import requests

from nina_cloud.datagen import config

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_TIMEOUT_S = 60
_CHUNK_PAGES = 20  # max items per OData $top request


def search_s2_products(
    access_token: str,
    footprint_wkt: str,
    start_date: str,
    end_date: str,
    cloud_mode: str = "clear",
    max_results: int = 100,
) -> List[Dict]:
    """Search the CDSE catalog for Sentinel-2 L2A products.

    Builds an OData ``$filter`` query that constrains by:

    * Collection name ``SENTINEL-2``
    * Product type ``S2MSI2A`` (Level-2A BOA reflectance + SCL)
    * Sensing start date range
    * Geographic intersection with *footprint_wkt*
    * Cloud cover percentage (based on *cloud_mode*)

    Cloud mode mapping:
      ``"clear"``  — cloudCover ≤ ACQUISITION_CLEAR_MAX_PCT (30%)
      ``"cloudy"`` — cloudCover ≥ ACQUISITION_CLOUDY_MIN_PCT (70%)

    Results are ordered by sensing start date ascending (oldest first)
    and paginated via ``@odata.nextLink``.

    Args:
        access_token: Bearer token from nina_thermal.acquisition.cdse_auth.
        footprint_wkt: WKT geometry string of the area of interest.
        start_date: Start of sensing window in ``"YYYYMMDD"`` format.
        end_date: End of sensing window in ``"YYYYMMDD"`` format.
        cloud_mode: One of ``"clear"`` or ``"cloudy"``.
        max_results: Maximum number of product records to return.

    Returns:
        List of product metadata dicts.  Each dict includes at minimum
        ``"Id"`` (UUID) and ``"Name"``.  Returns empty list on failure.

    Raises:
        ValueError: If *cloud_mode* is not ``"clear"`` or ``"cloudy"``.
    """
    if cloud_mode not in ("clear", "cloudy"):
        raise ValueError(f"cloud_mode must be 'clear' or 'cloudy', got '{cloud_mode}'")

    try:
        start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        logger.error("Invalid date format (expected YYYYMMDD): %s", exc)
        return []

    if cloud_mode == "clear":
        cloud_filter = (
            f"Attributes/OData.CSC.DoubleAttribute/any("
            f"att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value le {config.ACQUISITION_CLEAR_MAX_PCT})"
        )
    else:
        cloud_filter = (
            f"Attributes/OData.CSC.DoubleAttribute/any("
            f"att:att/Name eq 'cloudCover' and "
            f"att/OData.CSC.DoubleAttribute/Value ge {config.ACQUISITION_CLOUDY_MIN_PCT})"
        )

    filter_query = (
        f"Collection/Name eq '{config.S2_COLLECTION}' and "
        f"Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{config.S2_PRODUCT_TYPE}') and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start le {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}') and "
        f"{cloud_filter}"
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
                "CDSE S2 search HTTP error %s: %s",
                exc.response.status_code,
                exc,
            )
            break
        except Exception as exc:
            logger.error("CDSE S2 search failed: %s", exc)
            break

        page = body.get("value", [])
        products.extend(page)
        url = body.get("@odata.nextLink")

    result = products[:max_results]
    logger.info(
        "Found %d S2MSI2A products  cloud_mode=%s  requested≤%d",
        len(result),
        cloud_mode,
        max_results,
    )
    return result
