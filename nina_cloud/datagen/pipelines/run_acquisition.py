"""Pipeline stage 1: Acquire Sentinel-2 L2A scenes from CDSE.

Searches for both clear-sky (cloudCover ≤ 30%) and cloudy (cloudCover ≥ 60%)
scenes to ensure balanced label representation in the downstream dataset.

Scene deduplication for multi-operator / incremental runs
---------------------------------------------------------
Before downloading, the scene registry is checked.  Scenes already
registered by another operator are skipped — ``git pull`` first so you have
the latest registry (it is shared as ``data/manifests/scene_registry.json``).  After a successful download (or if the
.SAFE directory already exists locally) the scene is registered with a
deterministic split assignment (train / val / test) that never changes.

Outputs: 01_raw/ directory with downloaded .SAFE directories,
         acquisition_manifest.json, and updated scene_registry.json.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from shared.cdse_auth import get_access_token
from shared.cdse_download import download_and_extract
from shared.pipeline_logging import configure_logging, log_run_footer
from shared.scene_registry import (
    is_registered,
    load_registry,
    operator_name,
    register_scene,
    save_registry,
)

from nina_cloud.datagen import config
from nina_cloud.datagen.acquisition.cdse_search import search_s2_products

logger = logging.getLogger(__name__)


def _load_roi_wkt(roi_path: str) -> str:
    """Load a GeoJSON AOI file and return its WKT geometry string."""
    try:
        from sentinelsat import geojson_to_wkt, read_geojson  # type: ignore

        return geojson_to_wkt(read_geojson(roi_path))
    except ImportError:
        import json

        with open(roi_path) as f:
            gj = json.load(f)
        coords = gj["features"][0]["geometry"]["coordinates"][0]
        pts = " ".join(f"{lon} {lat}" for lon, lat in coords)
        return f"POLYGON(({pts}))"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 1: acquire Sentinel-2 L2A scenes from CDSE"
    )
    parser.add_argument("--roi", required=True, help="AOI GeoJSON file")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--output_dir", required=True, help="Output directory (01_raw)")
    parser.add_argument(
        "--cloud_mode",
        default="both",
        choices=["clear", "cloudy", "both"],
        help="Which cloud-cover tier to acquire (default: both)",
    )
    parser.add_argument(
        "--max_products",
        type=int,
        default=50,
        help="Max products per cloud mode (default: 50)",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help=(
            "Path to scene_registry.json.  "
            "Defaults to <output_dir>/../scene_registry.json "
            f"(i.e. data/cloud/{config.REGISTRY_FILENAME})."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Operator identifier recorded in the registry.  "
            "Defaults to $NINA_OPERATOR or the OS login name."
        ),
    )
    parser.add_argument(
        "--log_dir", default="data/logs", help="Directory for log files"
    )
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_acquisition",
        log_dir=args.log_dir,
        extra_id=f"{args.start}-{args.end}",
        argv=sys.argv[1:],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve registry path and operator
    registry_path = (
        Path(args.registry)
        if args.registry
        else output_dir.parent / config.REGISTRY_FILENAME
    )
    op = args.operator or operator_name()
    registry = load_registry(registry_path)
    logger.info(
        "Registry: %d scenes loaded from %s  operator=%s",
        len(registry),
        registry_path,
        op,
    )

    username = os.environ.get("CDSE_USER")
    password = os.environ.get("CDSE_PASSWORD")
    if not username or not password:
        logger.error("CDSE_USER and CDSE_PASSWORD environment variables must be set")
        return log_run_footer(start_time, 1)

    access_token = get_access_token(username, password)
    if not access_token:
        logger.error("Failed to obtain CDSE access token")
        return log_run_footer(start_time, 1)

    footprint_wkt = _load_roi_wkt(args.roi)
    logger.info("ROI: %s", args.roi)

    modes = ["clear", "cloudy"] if args.cloud_mode == "both" else [args.cloud_mode]
    manifest_records = []

    for mode in modes:
        logger.info(
            "Searching %s scenes (mode=%s, max=%d)",
            config.S2_PRODUCT_TYPE,
            mode,
            args.max_products,
        )
        products = search_s2_products(
            access_token=access_token,
            footprint_wkt=footprint_wkt,
            start_date=args.start,
            end_date=args.end,
            cloud_mode=mode,
            max_results=args.max_products,
        )
        logger.info("Found %d products for mode=%s", len(products), mode)

        for product in products:
            pid = product["Id"]
            name = product["Name"]  # scene_id == product Name

            # Multi-operator dedup: skip if another operator already registered this.
            if is_registered(registry, name):
                logger.info(
                    "Skipped (registered by %s): %s", registry[name]["operator"], name
                )
                status = "skipped_registered"
                manifest_records.append(
                    {
                        "product_id": pid,
                        "product_name": name,
                        "cloud_mode": mode,
                        "status": status,
                    }
                )
                continue

            safe_path = output_dir / f"{name}.SAFE"
            if safe_path.exists():
                # SAFE dir present but not registered (pre-registry run).
                logger.info("Already downloaded (catch-up register): %s", name)
                status = "skipped"
                register_scene(
                    registry, name, cloud_mode=mode, operator=op, status="acquired"
                )
            else:
                ok = download_and_extract(access_token, pid, name, output_dir)
                status = "downloaded" if ok else "failed"
                if ok:
                    register_scene(
                        registry, name, cloud_mode=mode, operator=op, status="acquired"
                    )

            manifest_records.append(
                {
                    "product_id": pid,
                    "product_name": name,
                    "cloud_mode": mode,
                    "status": status,
                }
            )

    # Persist registry after all downloads so a partial run doesn't mark
    # scenes as acquired before their .SAFE dirs are complete.
    save_registry(registry_path, registry)
    logger.info("Registry saved: %d scenes  path=%s", len(registry), registry_path)

    manifest_path = output_dir / "acquisition_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "stage": "run_acquisition",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "args": vars(args),
                "products": manifest_records,
            },
            f,
            indent=2,
        )
    logger.info("Manifest written: %s", manifest_path)
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
