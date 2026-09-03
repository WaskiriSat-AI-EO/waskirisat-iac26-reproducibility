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

**Fire-label provenance:** positive fire labels are derived SOLELY from NASA
FIRMS active-fire detections supplied via ``--firms_dir``.  The single-band
BT threshold (10.85 µm LWIR) cannot discriminate fire from warm bare rock or
sand on the FLIR Lepton 3.5 (broadband 8–14 µm, no MWIR channel); it is
therefore retained only as a per-granule QA cross-check metric — it is NEVER
used to assign label = 1.  If ``--firms_dir`` is not provided, all patches
receive label 0 and a warning is emitted at startup.
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
            "Directory containing FIRMS active-fire CSV files (VIIRS 375 m or "
            "MODIS 1 km NRT/standard products).  FIRMS is the SOLE authoritative "
            "source for positive fire labels.  If omitted, all patches receive "
            "label 0 and a warning is logged."
        ),
    )
    p.add_argument(
        "--firms_min_frp",
        type=float,
        default=config.FIRMS_MIN_FRP_MW,
        metavar="MW",
        help=(
            "Minimum Fire Radiative Power [MW] for a FIRMS detection to be "
            "accepted as a positive label.  Detections below this floor are "
            "too weak to produce a detectable LWIR thermal anomaly at the "
            f"{config.NINA_LEPTON_GSD_M:.0f} m Lepton GSD and would introduce "
            "contradictory (unlearnable) training samples.  "
            f"Default: {config.FIRMS_MIN_FRP_MW} MW."
        ),
    )
    p.add_argument(
        "--firms_min_confidence",
        choices=["low", "nominal", "high"],
        default=config.FIRMS_MIN_CONFIDENCE,
        help=(
            "Minimum FIRMS confidence tier for a detection to be accepted as "
            "a positive label.  "
            f"Default: {config.FIRMS_MIN_CONFIDENCE!r}."
        ),
    )
    p.add_argument(
        "--time_window_days",
        type=int,
        default=0,
        help="Number of calendar days around sensing_date to include FIRMS alerts (default: 0 = same day only).",
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

    # Warn loudly when FIRMS is absent — without it all labels are 0.
    if not args.firms_dir:
        logger.warning(
            "--firms_dir not provided.  FIRMS is the sole authoritative source "
            "for positive fire labels.  All patches in this run will receive "
            "label = 0.  Supply --firms_dir to produce a labelled dataset."
        )

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

        # ── FIRMS fire mask at 1 km (authoritative label source) ─────────
        # FIRMS (VIIRS 375 m / MODIS 1 km) is the SOLE source of positive
        # fire labels.  The binary mask is built at native SLSTR 1 km resolution
        # here, before 4×4 aggregation; after aggregation it becomes a fire
        # fraction per 4 km pixel (via nanmean), and a patch is labeled 1 when
        # that fraction exceeds MIN_FIRE_FRACTION.
        #
        # The BT-threshold mask is computed in parallel as a QA cross-check
        # only — it is never used to set label = 1.  Its values are logged and
        # written to the manifest so the paper can quantify the warm-surface
        # false-positive rate of the deprecated single-band algorithm.
        delta_T = granule["delta_T"]

        # QA cross-check: BT anomaly mask (NOT a label source).
        bt_fire_mask = (
            (f2_bt > config.FIRE_F2_ABS_THRESHOLD_K)
            & (delta_T > config.FIRE_DELTA_T_THRESHOLD_K)
            & np.isfinite(f2_bt)
            & np.isfinite(delta_T)
        )

        firms_fire_mask = np.zeros(f2_bt.shape, dtype=bool)
        if args.firms_dir:
            sensing_date = extract_sensing_date(granule_id)
            if sensing_date:
                firms_result = load_firms_for_granule(
                    Path(args.firms_dir),
                    lat,
                    lon,
                    sensing_date,
                    time_window_days=args.time_window_days,
                    min_confidence=args.firms_min_confidence,
                    min_frp_mw=args.firms_min_frp,
                )
                if firms_result is not None:
                    firms_fire_mask = firms_result
            else:
                logger.warning(
                    "  %s: could not parse sensing date — FIRMS skipped.", granule_id
                )

        # FIRMS is the label source — BT mask is QA only.
        fire_px_1km = firms_fire_mask.astype(np.float32)

        # QA cross-check metrics logged per granule and written to manifest.
        n_firms = int(firms_fire_mask.sum())
        n_bt_only = int((bt_fire_mask & ~firms_fire_mask).sum())
        n_firms_only = int((firms_fire_mask & ~bt_fire_mask).sum())
        n_agreement = int((bt_fire_mask & firms_fire_mask).sum())
        logger.info(
            "  %s: fire px (FIRMS)=%d  BT-only=%d  FIRMS-only=%d  agreement=%d"
            "  / %d total (%.4f%% labeled fire)",
            granule_id,
            n_firms,
            n_bt_only,
            n_firms_only,
            n_agreement,
            fire_px_1km.size,
            100.0 * n_firms / fire_px_1km.size,
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
        # delta_T, fire_frac, and bt_fire_frac use simple nanmean (not
        # Planck-domain) because they are not absolute radiances.
        # bt_fire_frac is the BT-threshold QA cross-check; it is co-tiled so
        # each patch record can carry both FIRMS fraction (label source) and BT
        # fraction (QA metric) for paper analysis.
        extra_in = {
            "delta_T": delta_T,
            "fire_frac": fire_px_1km,  # FIRMS label source
            "bt_fire_frac": bt_fire_mask.astype(np.float32),  # QA cross-check
        }
        if roi_mask_1km is not None:
            extra_in["roi_frac"] = roi_mask_1km

        patches_raw, grid_positions, extra = tile_granule_to_lepton_patches(
            f2_bt,  # uncorrected — broadband correction applied per-patch below
            stride_fraction=0.5,
            extra_arrays=extra_in,
        )

        # Apply broadband spectral correction to all aggregated patches at once.
        # apply_broadband_correction is purely elementwise (no cross-pixel
        # dependency), so one vectorized call over the stacked patches is
        # equivalent to — and much cheaper than — one call per patch.
        if patches_raw:
            patches = list(
                apply_broadband_correction(np.stack(patches_raw)).astype(np.float32)
            )
        else:
            patches = []

        delta_t_patches = extra["delta_T"]
        fire_frac_patches = extra["fire_frac"]
        bt_fire_frac_patches = extra["bt_fire_frac"]
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

            # ── Fire label from FIRMS 1 km mask (aggregated to patch) ────
            # fire_fraction: FIRMS-based — the authoritative label source.
            # bt_fire_fraction: BT-threshold-based — QA cross-check only,
            #   never used to set label = 1.  Enables the paper to quantify
            #   the warm-surface false-positive rate of the deprecated algorithm.
            fire_frac_patch = fire_frac_patches[patch_idx]
            finite_frac = fire_frac_patch[np.isfinite(fire_frac_patch)]
            fire_fraction = float(finite_frac.mean()) if finite_frac.size else 0.0
            label = 1 if fire_fraction >= config.MIN_FIRE_FRACTION else 0

            bt_frac_patch = bt_fire_frac_patches[patch_idx]
            bt_finite = bt_frac_patch[np.isfinite(bt_frac_patch)]
            bt_fire_fraction = float(bt_finite.mean()) if bt_finite.size else 0.0

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
                    "fire_fraction": fire_fraction,  # FIRMS-based (label source)
                    "bt_fire_fraction": bt_fire_fraction,  # BT-threshold (QA only)
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
