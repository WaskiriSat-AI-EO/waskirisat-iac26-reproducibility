"""Pipeline: authenticate → search → download Sentinel-3 SLSTR products.

Usage::

    python -m nina_thermal.pipelines.run_acquisition \\
        --roi        resources/bolivia_roi.geojson \\
        --start      20230101 \\
        --end        20230630 \\
        --output_dir data/01_raw \\
        --max_products 10

Writes ``acquisition_manifest.json`` to ``output_dir`` on completion.
Credentials are read from ``$CDSE_USER`` and ``$CDSE_PASSWORD``
(set them in a ``.env`` file or as environment variables).
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sentinelsat import read_geojson, geojson_to_wkt  # type: ignore

from shared.cdse_auth import get_access_token
from nina_thermal.datagen.acquisition.cdse_search import search_slstr_products
from nina_thermal.datagen.acquisition.firms_api import query_firms_fire_dates
from nina_thermal.datagen.ingestion.roi_mask import load_roi_polygon
from shared.cdse_download import download_and_extract
from nina_thermal.datagen import config
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_acquisition",
        description="Download Sentinel-3 SLSTR L1B products from CDSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--roi",
        required=True,
        metavar="GEOJSON",
        help="Path to GeoJSON file defining the area of interest (e.g. Bolivia AOI).",
    )
    p.add_argument(
        "--start",
        default=config.DEFAULT_START_DATE,
        metavar="YYYYMMDD",
        help="Start of sensing window.",
    )
    p.add_argument(
        "--end",
        default=config.DEFAULT_END_DATE,
        metavar="YYYYMMDD",
        help="End of sensing window.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory for downloaded .SEN3 directories.",
    )
    p.add_argument(
        "--max_products",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of products to download.",
    )
    p.add_argument(
        "--product_type",
        default="SL_1_RBT___",
        metavar="TYPE",
        help="SLSTR L1B product type identifier.",
    )
    p.add_argument(
        "--use_firms",
        action="store_true",
        help="If set, query NASA FIRMS API to identify dates with active fires and filter acquisitions to those dates.",
    )
    p.add_argument(
        "--firms_key",
        default=None,
        metavar="KEY",
        help="NASA FIRMS Map Key. Falls back to $FIRMS_MAP_KEY environment variable.",
    )
    p.add_argument(
        "--firms_source",
        default="VIIRS_SNPP_NRT",
        choices=["MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"],
        help="FIRMS sensor data source to query.",
    )
    p.add_argument(
        "--log_dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for log files.  Defaults to "
            "data/logs/run_{start}-{end} so all stages of a run share one dir."
        ),
    )
    return p


def main(argv=None) -> int:
    """Run the acquisition pipeline.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    args = build_parser().parse_args(argv)
    log_dir = args.log_dir or f"data/logs/run_{args.start}-{args.end}"
    _, start_time = configure_logging(
        "run_acquisition",
        log_dir,
        extra_id=f"{args.start}-{args.end}",
        argv=sys.argv[1:],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auth ─────────────────────────────────────────────────────────────────
    logger.info("Authenticating with CDSE …")
    token = get_access_token()
    if not token:
        logger.error("Authentication failed — check CDSE_USER / CDSE_PASSWORD.")
        return log_run_footer(start_time, exit_code=1)

    # ── AOI → WKT ─────────────────────────────────────────────────────────────
    try:
        footprint_wkt = geojson_to_wkt(read_geojson(args.roi))
    except Exception as exc:
        logger.error("Failed to read AOI GeoJSON %s: %s", args.roi, exc)
        return log_run_footer(start_time, exit_code=1)

    # ── FIRMS Query (Optional Date Filtering) ────────────────────────────────
    firms_key = args.firms_key or os.getenv("FIRMS_MAP_KEY")
    if args.use_firms and not firms_key:
        logger.error("FIRMS Map Key is required when using --use_firms. Set $FIRMS_MAP_KEY or pass --firms_key.")
        return log_run_footer(start_time, exit_code=1)

    fire_dates = None
    if args.use_firms:
        try:
            roi_poly = load_roi_polygon(Path(args.roi))
            bbox = roi_poly.bounds  # (minx, miny, maxx, maxy)
            logger.info("Querying NASA FIRMS for active fires in bounds: %s ...", bbox)
            fire_dates = query_firms_fire_dates(
                firms_key=firms_key,
                bbox=bbox,
                start_date_str=args.start,
                end_date_str=args.end,
                source=args.firms_source,
                min_confidence=config.FIRMS_MIN_CONFIDENCE,
                min_frp_mw=config.FIRMS_MIN_FRP_MW,
            )
            if not fire_dates:
                logger.warning(
                    "No active fires found on FIRMS for the given range. "
                    "Option B: Falling back to download all images for the full date range."
                )
                fire_dates = None
        except Exception as exc:
            logger.error(
                "FIRMS query failed: %s. Falling back to downloading all images.",
                exc,
            )
            fire_dates = None

    # ── Search ────────────────────────────────────────────────────────────────
    if fire_dates is not None:
        logger.info("Filtering CDSE search to %d date(s) with active fires...", len(fire_dates))
        products = []
        for d in fire_dates:
            logger.info("Searching CDSE for date: %s ...", d)
            day_products = search_slstr_products(
                access_token=token,
                footprint_wkt=footprint_wkt,
                start_date=d,
                end_date=d,
                product_type=args.product_type,
                max_results=args.max_products - len(products),
            )
            products.extend(day_products)
            if len(products) >= args.max_products:
                logger.info("Reached maximum products limit (%d).", args.max_products)
                break
    else:
        logger.info(
            "Searching catalog: %s  %s – %s  (max %d products) …",
            args.product_type,
            args.start,
            args.end,
            args.max_products,
        )
        products = search_slstr_products(
            access_token=token,
            footprint_wkt=footprint_wkt,
            start_date=args.start,
            end_date=args.end,
            product_type=args.product_type,
            max_results=args.max_products,
        )

    if not products:
        logger.warning("No products found for the given criteria.")
        _write_manifest(output_dir, [], args)
        return log_run_footer(start_time, exit_code=0)

    logger.info("Found %d product(s).", len(products))

    # ── Download + extract ────────────────────────────────────────────────────
    # Refresh token before each download: CDSE bearer tokens expire in ~10 min
    # and a 50-product session easily outlives one token.
    downloaded = []
    for i, product in enumerate(products, 1):
        pid = product["Id"]
        name = product["Name"]
        date = product.get("ContentDate", {}).get("Start", "?")
        logger.info("[%d/%d] %s  (sensing: %s)", i, len(products), name, date)

        token = get_access_token()
        if not token:
            logger.error("Token refresh failed at product %d — aborting.", i)
            break

        ok = download_and_extract(
            access_token=token,
            product_id=pid,
            product_name=name,
            output_dir=output_dir,
            delete_zip=True,
        )
        if ok:
            # The extracted directory follows Sentinel-3 naming conventions
            sen3_dir = output_dir / f"{name}.SEN3"
            downloaded.append(
                {
                    "product_id": pid,
                    "product_name": name,
                    "sensing_start": date,
                    "sen3_dir": str(sen3_dir),
                }
            )
        else:
            logger.warning("Download failed for %s — skipped.", name)

    _write_manifest(output_dir, downloaded, args)
    logger.info(
        "Acquisition complete: %d/%d products downloaded.  "
        "Manifest → %s/acquisition_manifest.json",
        len(downloaded),
        len(products),
        output_dir,
    )
    return log_run_footer(start_time, exit_code=0)


def _write_manifest(output_dir: Path, records: list, args) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "roi": args.roi,
            "start": args.start,
            "end": args.end,
            "product_type": args.product_type,
            "max_products": args.max_products,
        },
        "granules": records,
        "total_downloaded": len(records),
    }
    path = output_dir / "acquisition_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
