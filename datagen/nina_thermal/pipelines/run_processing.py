"""Pipeline Pass 1: load → spectral correction → aggregate → tile patches.

Usage::

    python -m nina_thermal.pipelines.run_processing \\
        --input_dir  data/02_extracted \\
        --output_dir data/03_patches

For each granule found via ``.npy`` files in ``input_dir``:

1. :func:`load_slstr_granule` — fast ``.npy`` path or NetCDF fallback
2. :func:`apply_broadband_correction` on ``F2_BT``
3. :func:`aggregate_bt_to_lepton_gsd` — radiance-consistent spatial averaging
4. :func:`tile_granule_to_lepton_patches` — with NaN handling and rejection

Each valid patch is saved as::

    {granule_id}_p{patch_index:04d}_bt.npy

Writes ``patches_manifest.json`` to ``output_dir`` with one record per
valid patch.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from nina_thermal.ingestion.load_granule import load_slstr_granule
from nina_thermal.ingestion.validate_granule import validate_granule
from nina_thermal.processing.spectral import apply_broadband_correction
from nina_thermal.processing.spatial import tile_granule_to_lepton_patches
from nina_thermal import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_processing",
        description=(
            "Pass 1: load extracted .npy granules, apply broadband correction, "
            "aggregate to Lepton GSD, and tile into patches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        metavar="DIR",
        help="Directory with extracted .npy files (output of run_extraction).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory for BT patch .npy files and patches_manifest.json.",
    )
    return p


def main(argv=None) -> int:
    """Run the processing pipeline (Pass 1).

    Returns:
        Exit code: 0 on success, 1 if no granules were found.
    """
    args = build_parser().parse_args(argv)
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover unique granule prefixes from F2_BT .npy files
    f2_files = sorted(input_dir.glob("*_F2_BT.npy"))
    if not f2_files:
        logger.warning("No *_F2_BT.npy files found in %s.", input_dir)
        _write_manifest(output_dir, [], args)
        return 1

    # One granule dir entry per unique F2_BT file (they live in input_dir)
    granule_dirs = sorted({f.parent for f in f2_files})
    logger.info(
        "Found %d granule(s) (%d F2_BT files).",
        len(granule_dirs), len(f2_files),
    )

    all_patch_records: List[dict] = []
    total_rejected = 0

    for granule_dir in granule_dirs:
        # There may be multiple granules in the same directory; iterate per F2 file
        for f2_file in sorted(granule_dir.glob("*_F2_BT.npy")):
            prefix     = f2_file.stem[: -len("_F2_BT")]
            granule_id = prefix

            logger.info("Processing granule %s …", granule_id)

            # ── Load ──────────────────────────────────────────────────────────
            granule = load_slstr_granule(granule_dir)
            if granule is None:
                logger.warning("Could not load %s — skipped.", granule_id)
                continue
            if not validate_granule(granule):
                logger.warning("Granule %s failed validation — skipped.", granule_id)
                continue

            f2_bt = granule["F2_BT"]
            lat   = granule["lat"]
            lon   = granule["lon"]

            # ── Spectral correction ───────────────────────────────────────────
            f2_bt_corrected = apply_broadband_correction(f2_bt).astype(np.float32)

            # ── Tile into patches (aggregation is done inside) ───────────────
            # tile_granule_to_lepton_patches calls aggregate_bt_to_lepton_gsd
            # internally; pass the corrected (non-aggregated) array directly.
            patches = tile_granule_to_lepton_patches(f2_bt_corrected)

            # Aggregate lat/lon once for centroid computation
            agg = config.SLSTR_AGG_FACTOR
            h_crop = (f2_bt_corrected.shape[0] // agg) * agg
            w_crop = (f2_bt_corrected.shape[1] // agg) * agg
            lat_agg = (
                lat[:h_crop, :w_crop]
                .reshape(h_crop // agg, agg, w_crop // agg, agg)
                .mean(axis=(1, 3))
                .astype(np.float32)
            )
            lon_agg = (
                lon[:h_crop, :w_crop]
                .reshape(h_crop // agg, agg, w_crop // agg, agg)
                .mean(axis=(1, 3))
                .astype(np.float32)
            )

            w_px, h_px = config.LEPTON_RESOLUTION  # (160, 120)
            n_cols = lat_agg.shape[1] // w_px
            n_rows_rejected_before = total_rejected

            for patch_idx, patch in enumerate(patches):
                row = patch_idx // n_cols
                col = patch_idx  % n_cols

                # Centroid from aggregated coordinate arrays (best-effort)
                r0, c0 = row * h_px, col * w_px
                r1, c1 = r0 + h_px, c0 + w_px

                lat_patch = lat_agg[r0:r1, c0:c1] if lat_agg.shape[0] >= r1 and lat_agg.shape[1] >= c1 else None
                lon_patch = lon_agg[r0:r1, c0:c1] if lon_agg.shape[0] >= r1 and lon_agg.shape[1] >= c1 else None

                centroid_lat = float(np.nanmean(lat_patch)) if lat_patch is not None else float("nan")
                centroid_lon = float(np.nanmean(lon_patch)) if lon_patch is not None else float("nan")

                nan_fraction = float(np.isnan(patch).mean())
                finite = patch[np.isfinite(patch)]
                bt_min = float(finite.min()) if finite.size else float("nan")
                bt_max = float(finite.max()) if finite.size else float("nan")

                # Save patch
                patch_filename = f"{granule_id}_p{patch_idx:04d}_bt.npy"
                np.save(output_dir / patch_filename, patch)

                all_patch_records.append({
                    "granule_id":    granule_id,
                    "patch_index":   patch_idx,
                    "patch_file":    patch_filename,
                    "nan_fraction":  nan_fraction,
                    "centroid_lat":  centroid_lat,
                    "centroid_lon":  centroid_lon,
                    "bt_min":        bt_min,
                    "bt_max":        bt_max,
                })

            n_patches = len(patches)
            logger.info(
                "  %s: %d patch(es) saved.", granule_id, n_patches,
            )

    _write_manifest(output_dir, all_patch_records, args)
    logger.info(
        "Processing complete: %d patch(es) across %d granule(s).  "
        "Manifest → %s/patches_manifest.json",
        len(all_patch_records),
        len(granule_dirs),
        output_dir,
    )
    return 0


def _write_manifest(output_dir: Path, records: list, args) -> None:
    manifest = {
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir":  args.input_dir,
            "output_dir": args.output_dir,
        },
        "patches":      records,
        "total_patches": len(records),
    }
    path = output_dir / "patches_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
