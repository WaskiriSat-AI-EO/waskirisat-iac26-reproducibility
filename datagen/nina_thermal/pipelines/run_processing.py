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

from nina_thermal.ingestion.validate_granule import validate_granule
from nina_thermal.processing.spectral import apply_broadband_correction
from nina_thermal.processing.spatial import tile_granule_to_lepton_patches
from nina_thermal import config
from nina_thermal.pipelines._logging import configure_logging, log_run_footer

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
    p.add_argument(
        "--log_dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for log files.  Defaults to "
            f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}."
        ),
    )
    return p


def main(argv=None) -> int:
    """Run the processing pipeline (Pass 1).

    Returns:
        Exit code: 0 on success, 1 if no granules were found.
    """
    args = build_parser().parse_args(argv)
    log_dir = args.log_dir or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    _, start_time = configure_logging("run_processing", log_dir, argv=sys.argv[1:])
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover granules — one per *_F2_BT.npy file
    f2_files = sorted(input_dir.glob("*_F2_BT.npy"))
    if not f2_files:
        logger.warning("No *_F2_BT.npy files found in %s.", input_dir)
        _write_manifest(output_dir, [], args)
        return log_run_footer(start_time, exit_code=1)

    logger.info("Found %d granule(s).", len(f2_files))

    all_patch_records: List[dict] = []
    total_rejected = 0

    for f2_file in f2_files:
        prefix      = f2_file.stem[: -len("_F2_BT")]
        granule_id  = prefix
        granule_dir = f2_file.parent

        logger.info("Processing granule %s …", granule_id)

        # ── Load ──────────────────────────────────────────────────────────
        try:
            f2_bt_raw = np.load(granule_dir / f"{prefix}_F2_BT.npy").astype(np.float32)
            s8_bt_raw = np.load(granule_dir / f"{prefix}_S8_BT.npy").astype(np.float32)
            lat       = np.load(granule_dir / f"{prefix}_lat.npy").astype(np.float32)
            lon       = np.load(granule_dir / f"{prefix}_lon.npy").astype(np.float32)
        except Exception as exc:
            logger.warning("Could not load %s — skipped: %s", granule_id, exc)
            continue

        granule = {
            "F2_BT":   f2_bt_raw,
            "S8_BT":   s8_bt_raw,
            "delta_T": (f2_bt_raw - s8_bt_raw).astype(np.float32),
            "lat":     lat,
            "lon":     lon,
        }
        if not validate_granule(granule):
            logger.warning("Granule %s failed validation — skipped.", granule_id)
            continue

        f2_bt = granule["F2_BT"]
        lat   = granule["lat"]
        lon   = granule["lon"]

        # ── Spectral correction ───────────────────────────────────────────
        f2_bt_corrected = apply_broadband_correction(f2_bt).astype(np.float32)

        # ── Tile into patches (aggregation is done inside) ───────────────
        # Co-tile delta_T alongside BT so run_simulation can find _delta_t.npy.
        # delta_T uses simple nanmean aggregation (not Planck-domain) because
        # it is a temperature difference, not an absolute radiance.
        delta_T = granule["delta_T"]
        patches, extra = tile_granule_to_lepton_patches(
            f2_bt_corrected,
            stride_fraction=0.5,
            extra_arrays={"delta_T": delta_T},
        )
        delta_t_patches = extra["delta_T"]

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

            # Save BT patch and co-tiled delta_T patch
            patch_filename = f"{granule_id}_p{patch_idx:04d}_bt.npy"
            np.save(output_dir / patch_filename, patch)
            np.save(
                output_dir / f"{granule_id}_p{patch_idx:04d}_delta_t.npy",
                delta_t_patches[patch_idx],
            )

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
        len(f2_files),
        output_dir,
    )
    return log_run_footer(start_time, exit_code=0)


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
