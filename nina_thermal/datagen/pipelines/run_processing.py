"""Pipeline Pass 1: load → aggregate → tile patches → spectral correction.

Usage::

    python -m nina_thermal.pipelines.run_processing \\
        --input_dir  data/02_extracted \\
        --output_dir data/03_patches

For each granule found via ``.npy`` files in ``input_dir``:

1. :func:`load_slstr_granule` — fast ``.npy`` path or NetCDF fallback
2. :func:`aggregate_bt_to_lepton_gsd` — radiance-consistent spatial averaging
3. :func:`tile_granule_to_lepton_patches` — with NaN handling and rejection
4. :func:`apply_broadband_correction` on each 4 km BT patch (post-aggregation)

The spectral correction is applied AFTER spatial aggregation so that it
operates on the Lepton-equivalent 4 km signal.  Applying it before
aggregation would correct individual 1 km SLSTR pixels, which is not what
the Lepton sensor physically integrates over its footprint.

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

from nina_thermal.datagen.ingestion.validate_granule import validate_granule
from nina_thermal.datagen.ingestion.firms_labels import (
    extract_sensing_date,
    load_firms_for_granule,
)
from nina_thermal.datagen.ingestion.roi_mask import (
    load_roi_polygon,
    compute_roi_pixel_mask,
)
from nina_thermal.datagen.processing.spectral import apply_broadband_correction
from nina_thermal.datagen.processing.spatial import tile_granule_to_lepton_patches
from nina_thermal.datagen import config
from shared.pipeline_logging import configure_logging, log_run_footer

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
    p.add_argument(
        "--roi",
        default=None,
        metavar="GEOJSON",
        help=(
            "Optional GeoJSON file defining the area of interest. "
            "When provided, patches with no pixels inside the polygon are dropped. "
            "Matches the ROI used in run_acquisition."
        ),
    )
    p.add_argument(
        "--firms_dir",
        default=None,
        metavar="DIR",
        help=(
            "Optional directory containing FIRMS active-fire CSV files. "
            "When provided, FIRMS detections are OR-combined with the BT-based "
            "fire mask for improved recall."
        ),
    )
    return p


def main(argv=None) -> int:
    """Run the processing pipeline (Pass 1).

    Returns:
        Exit code: 0 on success, 1 if no granules were found.
    """
    args = build_parser().parse_args(argv)
    log_dir = (
        args.log_dir
        or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    )
    _, start_time = configure_logging("run_processing", log_dir, argv=sys.argv[1:])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ROI polygon once — used to reject patches outside the study area
    roi_polygon = None
    if args.roi:
        try:
            roi_polygon = load_roi_polygon(Path(args.roi))
            logger.info("ROI loaded from %s (bounds: %s)", args.roi, roi_polygon.bounds)
        except Exception as exc:
            logger.error(
                "Failed to load ROI %s: %s — proceeding without spatial filter.",
                args.roi,
                exc,
            )

    # Discover granules — one per *_F2_BT.npy file
    f2_files = sorted(input_dir.glob("*_F2_BT.npy"))
    if not f2_files:
        logger.warning("No *_F2_BT.npy files found in %s.", input_dir)
        _write_manifest(output_dir, [], args)
        return log_run_footer(start_time, exit_code=1)

    logger.info("Found %d granule(s).", len(f2_files))

    all_patch_records: List[dict] = []

    for f2_file in f2_files:
        prefix = f2_file.stem[: -len("_F2_BT")]
        granule_id = prefix
        granule_dir = f2_file.parent

        logger.info("Processing granule %s …", granule_id)

        # ── Load ──────────────────────────────────────────────────────────
        try:
            f2_bt_raw = np.load(granule_dir / f"{prefix}_F2_BT.npy").astype(np.float32)
            s8_bt_raw = np.load(granule_dir / f"{prefix}_S8_BT.npy").astype(np.float32)
            lat = np.load(granule_dir / f"{prefix}_lat.npy").astype(np.float32)
            lon = np.load(granule_dir / f"{prefix}_lon.npy").astype(np.float32)
        except Exception as exc:
            logger.warning("Could not load %s — skipped: %s", granule_id, exc)
            continue

        granule = {
            "F2_BT": f2_bt_raw,
            "S8_BT": s8_bt_raw,
            "delta_T": (f2_bt_raw - s8_bt_raw).astype(np.float32),
            "lat": lat,
            "lon": lon,
        }
        if not validate_granule(granule):
            logger.warning("Granule %s failed validation — skipped.", granule_id)
            continue

        f2_bt = granule["F2_BT"]
        lat = granule["lat"]
        lon = granule["lon"]

        # ── Fire labels at 1 km (before aggregation) ─────────────────────
        # Thresholds in config are calibrated for 1 km SLSTR pixels.
        # After 4×4 aggregation, fire signals dilute to ~1/16 of their 1 km
        # magnitude, making post-aggregation detection unreliable.
        # Strategy: build a binary fire-pixel mask at 1 km, aggregate it
        # alongside BT (nanmean → fire fraction per 4 km pixel), then label
        # each patch from the aggregated fire fraction.
        delta_T = granule["delta_T"]
        bt_fire_mask = (
            (f2_bt > config.FIRE_F2_ABS_THRESHOLD_K)
            & (delta_T > config.FIRE_DELTA_T_THRESHOLD_K)
            & np.isfinite(f2_bt)
            & np.isfinite(delta_T)
        )
        n_bt_fire = int(bt_fire_mask.sum())

        firms_fire_mask = np.zeros_like(bt_fire_mask)
        if args.firms_dir:
            sensing_date = extract_sensing_date(granule_id)
            if sensing_date:
                firms_result = load_firms_for_granule(
                    Path(args.firms_dir), lat, lon, sensing_date
                )
                if firms_result is not None:
                    firms_fire_mask = firms_result
                    logger.info(
                        "  %s: FIRMS fire pixels = %d",
                        granule_id,
                        int(firms_fire_mask.sum()),
                    )
            else:
                logger.warning(
                    "  %s: could not parse sensing date — FIRMS skipped.", granule_id
                )

        fire_px_1km = (bt_fire_mask | firms_fire_mask).astype(np.float32)
        n_fire_1km = int(fire_px_1km.sum())
        logger.info(
            "  %s: fire pixels at 1 km = %d (BT-only=%d) / %d (%.4f%%)",
            granule_id,
            n_fire_1km,
            n_bt_fire,
            fire_px_1km.size,
            100.0 * n_fire_1km / fire_px_1km.size,
        )

        # ── ROI pixel mask at 1 km (co-tiled to get per-patch ROI coverage) ─
        roi_mask_1km = None
        if roi_polygon is not None:
            roi_mask_1km = compute_roi_pixel_mask(lat, lon, roi_polygon).astype(
                np.float32
            )

        # ── Tile into patches (aggregation is done inside) ───────────────
        # BT is passed uncorrected here; the broadband spectral correction is
        # applied AFTER aggregation (per patch) so that it operates on the
        # Lepton-equivalent 4 km signal rather than individual 1 km SLSTR
        # pixels. This is the physically correct order: the Lepton sensor
        # integrates broadband radiance over its whole footprint; the spectral
        # correction models its SRF difference from SLSTR's narrowband response
        # for that spatially-integrated signal.
        #
        # delta_T and fire_frac use simple nanmean (not Planck-domain)
        # because they are not absolute radiances.
        extra_in = {"delta_T": delta_T, "fire_frac": fire_px_1km}
        if roi_mask_1km is not None:
            extra_in["roi_frac"] = roi_mask_1km

        patches_raw, grid_positions, extra = tile_granule_to_lepton_patches(
            f2_bt,  # uncorrected — broadband correction applied per-patch below
            stride_fraction=0.5,
            extra_arrays=extra_in,
        )

        # Apply broadband spectral correction to each aggregated patch.
        patches = [apply_broadband_correction(p).astype(np.float32) for p in patches_raw]

        delta_t_patches = extra["delta_T"]
        fire_frac_patches = extra["fire_frac"]
        roi_frac_patches = extra.get("roi_frac")

        # Aggregate lat/lon once for centroid computation
        agg = config.SLSTR_AGG_FACTOR
        h_crop = (f2_bt.shape[0] // agg) * agg
        w_crop = (f2_bt.shape[1] // agg) * agg
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
        n_roi_rejected = 0

        for patch_idx, patch in enumerate(patches):
            # ── ROI filter: skip patches with no pixels inside the study area ─
            if roi_frac_patches is not None:
                roi_patch = roi_frac_patches[patch_idx]
                if float(roi_patch.mean()) == 0.0:
                    n_roi_rejected += 1
                    continue

            # Use the recorded grid position (top-left of patch in aggregated
            # array) rather than reconstructing from list-index arithmetic.
            # List-index arithmetic breaks when NaN-rejected patches leave holes
            # in the grid — grid_positions is always aligned with the patch list.
            r0, c0 = grid_positions[patch_idx]
            r1, c1 = r0 + h_px, c0 + w_px

            lat_patch = (
                lat_agg[r0:r1, c0:c1]
                if lat_agg.shape[0] >= r1 and lat_agg.shape[1] >= c1
                else None
            )
            lon_patch = (
                lon_agg[r0:r1, c0:c1]
                if lon_agg.shape[0] >= r1 and lon_agg.shape[1] >= c1
                else None
            )

            centroid_lat = (
                float(np.nanmean(lat_patch)) if lat_patch is not None else float("nan")
            )
            centroid_lon = (
                float(np.nanmean(lon_patch)) if lon_patch is not None else float("nan")
            )

            nan_fraction = float(np.isnan(patch).mean())
            finite = patch[np.isfinite(patch)]
            bt_min = float(finite.min()) if finite.size else float("nan")
            bt_max = float(finite.max()) if finite.size else float("nan")

            # ── Fire label from 1km mask (aggregated to patch) ────────────
            fire_frac_patch = fire_frac_patches[patch_idx]
            finite_frac = fire_frac_patch[np.isfinite(fire_frac_patch)]
            fire_fraction = float(finite_frac.mean()) if finite_frac.size else 0.0
            label = 1 if fire_fraction >= config.MIN_FIRE_FRACTION else 0

            # Save BT patch and co-tiled delta_T patch
            patch_filename = f"{granule_id}_p{patch_idx:04d}_bt.npy"
            np.save(output_dir / patch_filename, patch)
            np.save(
                output_dir / f"{granule_id}_p{patch_idx:04d}_delta_t.npy",
                delta_t_patches[patch_idx],
            )

            all_patch_records.append(
                {
                    "granule_id": granule_id,
                    "patch_index": patch_idx,
                    "patch_file": patch_filename,
                    "nan_fraction": nan_fraction,
                    "centroid_lat": centroid_lat,
                    "centroid_lon": centroid_lon,
                    "bt_min": bt_min,
                    "bt_max": bt_max,
                    "label": label,
                    "fire_fraction": fire_fraction,
                }
            )

        n_patches = len(patches) - n_roi_rejected
        logger.info(
            "  %s: %d patch(es) saved  (%d outside ROI — dropped).",
            granule_id,
            n_patches,
            n_roi_rejected,
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
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
        },
        "patches": records,
        "total_patches": len(records),
    }
    path = output_dir / "patches_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
