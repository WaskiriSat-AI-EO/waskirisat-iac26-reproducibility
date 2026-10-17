"""Pipeline: authenticate → search → download Sentinel-3 SLSTR products.

Usage::

    python -m nina_thermal.pipelines.run_acquisition \\
        --roi        resources/bolivia_roi.geojson \\
        --start      20230101 \\
        --end        20230630 \\
        --output_dir data/01_raw \\
        --max_products 10

Writes ``acquisition_manifest.json`` and ``granule_inventory.txt`` to
``output_dir``.  Both are merged with what is already there, so an interrupted
run can be restarted with the same command.

v4 options (see ``docs/RUNBOOK_v4_data_expansion.md``):

* ``--sampling even`` (default) spreads ``--max_products`` over every day of
  the window instead of taking the oldest products.
* ``--skip_inventory GLOB`` (repeatable) drops granules already listed in
  another root's ``granule_inventory.txt``.
* ``--extract_dir DIR --delete_raw`` extracts each granule to ``.npy`` right
  after its download and deletes the ``.SEN3`` folder, so raw data never
  accumulates (~380 MB per granule vs ~27 MB extracted).
* ``--dry_run`` searches and prints the selection without downloading.

Credentials are read from ``$CDSE_USER`` and ``$CDSE_PASSWORD``
(set them in a ``.env`` file or as environment variables).
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sentinelsat import read_geojson, geojson_to_wkt  # type: ignore

from shared.cdse_auth import get_access_token
from nina_thermal.datagen.acquisition.cdse_search import (
    drop_held_products,
    granule_key,
    load_held_granules,
    products_per_day,
    search_slstr_products,
    select_products,
)
from nina_thermal.datagen.acquisition.firms_api import query_firms_fire_dates
from nina_thermal.datagen.ingestion.extract_channels import extract_thermal_channels
from nina_thermal.datagen.ingestion.roi_mask import load_roi_polygon
from shared.cdse_download import download_and_extract
from nina_thermal.datagen import config
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)

# Upper bound on one window's search results before sampling (a full month
# over the ROI is ~170 SL_1_RBT products).
_SEARCH_LIMIT = 5000
INVENTORY_FILE = "granule_inventory.txt"
_MIN_ROI_SPAN_DEG = 1.0
_DOWNLOAD_ATTEMPTS = 3
_RETRY_WAIT_S = 30
_EXTRACTED_SUFFIXES = ("F2_BT", "S8_BT", "lat", "lon")


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
        help="Maximum number of products to download (after skipping held ones).",
    )
    p.add_argument(
        "--sampling",
        default="even",
        choices=["even", "oldest"],
        help=(
            "How --max_products is applied to the window: 'even' takes products "
            "round-robin per sensing day; 'oldest' keeps the first ones (pre-v4)."
        ),
    )
    p.add_argument(
        "--skip_inventory",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "granule_inventory.txt file(s) of other roots; listed granules are not "
            "downloaded again.  Repeatable."
        ),
    )
    p.add_argument(
        "--extract_dir",
        default=None,
        metavar="DIR",
        help="Extract each granule to .npy here right after download (02_extracted).",
    )
    p.add_argument(
        "--delete_raw",
        action="store_true",
        help="Delete each .SEN3 folder once its .npy files exist (needs --extract_dir).",
    )
    p.add_argument(
        "--allow_small_roi",
        action="store_true",
        help="Accept an ROI narrower than 1° (normally a mistake: see bolivia_roi.geojson).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Search and print the selection per day; download nothing.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delete_raw and not args.extract_dir:
        parser.error("--delete_raw requires --extract_dir")
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
    if not args.allow_small_roi:
        west, south, east, north = load_roi_polygon(Path(args.roi)).bounds
        if min(east - west, north - south) < _MIN_ROI_SPAN_DEG:
            # resources/bolivia_roi.geojson is a 0.1° test box; the v3 roots
            # data_202208/02/09 were fetched with it and got ~40% of the
            # granules the country polygon yields.
            logger.error(
                "ROI %s spans only %.2f° × %.2f°.  Use "
                "nina_thermal/resources/bo_nina_roi_simplified.geojson for Bolivia, "
                "or pass --allow_small_roi.",
                args.roi,
                east - west,
                north - south,
            )
            return log_run_footer(start_time, exit_code=1)

    # ── FIRMS Query (Optional Date Filtering) ────────────────────────────────
    firms_key = args.firms_key or os.getenv("FIRMS_MAP_KEY")
    if args.use_firms and not firms_key:
        logger.error(
            "FIRMS Map Key is required when using --use_firms. Set $FIRMS_MAP_KEY or pass --firms_key."
        )
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

    # ── Search (full window, then skip held, then sample) ────────────────────
    if fire_dates is not None:
        logger.info(
            "Filtering CDSE search to %d date(s) with active fires...", len(fire_dates)
        )
        windows = [(d, d) for d in fire_dates]
    else:
        windows = [(args.start, args.end)]
    logger.info(
        "Searching catalog: %s  %s – %s  (%d window(s)) …",
        args.product_type,
        args.start,
        args.end,
        len(windows),
    )
    found: list = []
    for w_start, w_end in windows:
        found.extend(
            search_slstr_products(
                access_token=token,
                footprint_wkt=footprint_wkt,
                start_date=w_start,
                end_date=w_end,
                product_type=args.product_type,
                max_results=_SEARCH_LIMIT,
            )
        )
    unique = {granule_key(p["Name"]): p for p in found}
    found = list(unique.values())
    logger.info("Catalog: %d product(s) in the window.", len(found))

    held = load_held_granules(args.skip_inventory)
    # The root's own inventory: granules already fetched by an earlier,
    # interrupted run of this command are resumed, not skipped.
    own = _read_inventory(output_dir)
    held -= own
    candidates, skipped = drop_held_products(found, held)
    if skipped:
        logger.info("Skipped %d product(s) already held by other roots.", len(skipped))

    products = select_products(candidates, args.max_products, args.sampling)
    per_day = products_per_day(products)
    logger.info(
        "Selected %d product(s) over %d day(s) (sampling=%s): %s",
        len(products),
        len(per_day),
        args.sampling,
        ", ".join(f"{d[5:]}:{n}" for d, n in per_day.items()),
    )

    if args.dry_run:
        for p in products:
            status = "held" if granule_key(p["Name"]) in own else "new"
            print(f"{_sensing_start(p)}  {status:4s}  {p['Name']}")
        print(
            f"dry run: {len(found)} in window, {len(skipped)} held by other roots, "
            f"{len(products)} selected ({len(per_day)} days)"
        )
        return log_run_footer(start_time, exit_code=0)

    if not products:
        logger.warning("No products found for the given criteria.")
        _write_manifest(output_dir, [], args)
        return log_run_footer(start_time, exit_code=0)

    # ── Download + extract ────────────────────────────────────────────────────
    # Refresh token before each download: CDSE bearer tokens expire in ~10 min
    # and a 50-product session easily outlives one token.
    extract_dir = Path(args.extract_dir) if args.extract_dir else None
    if extract_dir is not None:
        extract_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    extracted = []
    failed = 0
    for i, product in enumerate(products, 1):
        pid = product["Id"]
        name = product["Name"]
        date = _sensing_start(product) or "?"
        sen3_dir = output_dir / _sen3_dirname(name)
        record = {
            "product_id": pid,
            "product_name": name,
            "sensing_start": date,
            "sen3_dir": str(sen3_dir),
        }
        logger.info("[%d/%d] %s  (sensing: %s)", i, len(products), name, date)

        if extract_dir is not None and _is_extracted(extract_dir, name):
            logger.info("Already extracted, skipping download: %s", name)
            downloaded.append(record)
            _append_inventory(output_dir, name)
            continue

        if not sen3_dir.is_dir():
            ok = False
            for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
                token = get_access_token()
                if not token:
                    logger.error("Token refresh failed at product %d.", i)
                    break
                ok = (
                    download_and_extract(
                        access_token=token,
                        product_id=pid,
                        product_name=name,
                        output_dir=output_dir,
                        delete_zip=True,
                    )
                    and sen3_dir.is_dir()
                )
                if ok:
                    break
                # The partial ZIP is removed on failure, so a retry starts
                # clean.  CDSE drops ~5% of long downloads (IncompleteRead).
                _remove_partial(output_dir, name)
                if attempt < _DOWNLOAD_ATTEMPTS:
                    wait = _RETRY_WAIT_S * attempt
                    logger.warning(
                        "Download attempt %d/%d failed for %s — retrying in %ds.",
                        attempt,
                        _DOWNLOAD_ATTEMPTS,
                        name,
                        wait,
                    )
                    time.sleep(wait)
            if not ok:
                logger.warning("Download failed for %s — skipped.", name)
                failed += 1
                continue

        if extract_dir is not None:
            result = extract_thermal_channels(sen3_dir, extract_dir)
            if result is None:
                logger.warning("Extraction failed for %s — raw kept.", name)
                failed += 1
                continue
            extracted.append(_extraction_record(sen3_dir, result))
            if args.delete_raw and _is_extracted(extract_dir, name):
                shutil.rmtree(sen3_dir)
                logger.info("Deleted raw %s", sen3_dir.name)

        downloaded.append(record)
        _append_inventory(output_dir, name)
        # Written per granule so an interrupted run loses nothing.
        _write_manifest(output_dir, [record], args)
        if extract_dir is not None:
            _write_extraction_manifest(extract_dir, extracted[-1:], output_dir)

    _write_manifest(output_dir, [], args)
    logger.info(
        "Acquisition complete: %d/%d products downloaded (%d failed).  "
        "Manifest → %s/acquisition_manifest.json",
        len(downloaded),
        len(products),
        failed,
        output_dir,
    )
    return log_run_footer(start_time, exit_code=1 if failed else 0)


def _sensing_start(product: dict) -> str:
    return product.get("ContentDate", {}).get("Start", "")


def _sen3_dirname(name: str) -> str:
    """CDSE names already end in ``.SEN3``; older code appended it twice."""
    return name if name.endswith(".SEN3") else f"{name}.SEN3"


def _remove_partial(output_dir: Path, name: str) -> None:
    """Delete a failed download's leftovers so the next attempt starts clean."""
    (output_dir / f"{name}.zip").unlink(missing_ok=True)
    sen3_dir = output_dir / _sen3_dirname(name)
    if sen3_dir.is_dir():
        shutil.rmtree(sen3_dir)


def _is_extracted(extract_dir: Path, name: str) -> bool:
    prefix = granule_key(name)
    return all(
        (extract_dir / f"{prefix}_{suffix}.npy").is_file()
        for suffix in _EXTRACTED_SUFFIXES
    )


def _read_inventory(output_dir: Path) -> set:
    path = output_dir / INVENTORY_FILE
    if not path.is_file():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {granule_key(line) for line in fh if line.strip()}


def _append_inventory(output_dir: Path, name: str) -> None:
    """Add *name* to ``granule_inventory.txt`` (sorted, unique, ``.SEN3`` names)."""
    keys = _read_inventory(output_dir) | {granule_key(name)}
    path = output_dir / INVENTORY_FILE
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(f"{k}.SEN3\n" for k in sorted(keys))


def _extraction_record(sen3_dir: Path, result: tuple) -> dict:
    p_f2, p_s8, p_lat, p_lon = result
    return {
        "granule_id": sen3_dir.stem,
        "prefix": p_f2.stem[: -len("_F2_BT")],
        "sen3_dir": str(sen3_dir),
        "npy_files": {
            "F2_BT": str(p_f2),
            "S8_BT": str(p_s8),
            "lat": str(p_lat),
            "lon": str(p_lon),
        },
    }


def _write_extraction_manifest(
    extract_dir: Path, records: list, input_dir: Path
) -> None:
    """Merge *records* into ``extraction_manifest.json`` (run_extraction format)."""
    path = extract_dir / "extraction_manifest.json"
    existing = []
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh).get("granules", [])
    merged = {r["prefix"]: r for r in existing}
    merged.update({r["prefix"]: r for r in records})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(extract_dir),
            "streamed_by": "run_acquisition",
        },
        "granules": sorted(merged.values(), key=lambda r: r["prefix"]),
        "total_extracted": len(merged),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _write_manifest(output_dir: Path, records: list, args) -> None:
    """Merge *records* into ``acquisition_manifest.json`` (keyed by product name)."""
    path = output_dir / "acquisition_manifest.json"
    existing = []
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh).get("granules", [])
    merged = {r["product_name"]: r for r in existing}
    merged.update({r["product_name"]: r for r in records})
    records = sorted(merged.values(), key=lambda r: r["sensing_start"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "roi": args.roi,
            "start": args.start,
            "end": args.end,
            "product_type": args.product_type,
            "max_products": args.max_products,
            "sampling": args.sampling,
            "skip_inventory": args.skip_inventory,
            "extract_dir": args.extract_dir,
            "delete_raw": args.delete_raw,
        },
        "granules": records,
        "total_downloaded": len(records),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.debug("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
