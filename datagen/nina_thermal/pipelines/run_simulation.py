"""Pipeline Pass 2: simulate FLIR Lepton images, compute labels, export dataset.

Usage::

    python -m nina_thermal.pipelines.run_simulation \\
        --input_dir  data/03_patches \\
        --output_dir data/04_dataset \\
        --profiles   A,B,C \\
        --agc_modes  fixed,global \\
        --seed       42

For each patch record in ``patches_manifest.json``:

1. Load the BT patch ``.npy`` (and delta-T patch if present).
2. :func:`compute_patch_label` and :func:`classify_patch_surface_type`.
3. For each AGC mode × degradation profile:

   a. :func:`apply_agc` to normalise BT.
   b. :func:`simulate_lepton` to apply the full degradation chain.
   c. Save grayscale PNG to ``images/`` and JSON metadata to ``metadata/``.
   d. Optionally save false-colour PNG to ``viz/`` (requires ``--validation``).

Writes ``label_manifest.json`` and ``simulation_manifest.json`` to
``output_dir``.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nina_thermal import config
from nina_thermal.dataset.export import (
    build_image_metadata,
    save_label_manifest,
    save_patch_metadata,
)
from nina_thermal.labeling.fire_label import (
    classify_patch_surface_type,
    compute_patch_label,
)
from nina_thermal.processing.agc import apply_agc, compute_global_percentiles
from nina_thermal.simulation.noise import generate_lepton_fpn
from nina_thermal.simulation.sensor import simulate_lepton
from nina_thermal.pipelines._logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)

# Manifests and subdirectories written by this stage
SKIP_NAMES = {
    "label_manifest.json",
    "simulation_manifest.json",
    "dataset_summary.json",
}

# Deterministic profile -> FPN seed offset (avoids hash() randomness)
_PROFILE_SEED_OFFSET = {"A": 0, "B": 1, "C": 2}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_simulation",
        description=(
            "Pass 2: simulate FLIR Lepton 3.5 images, compute fire labels, "
            "and export the dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        metavar="DIR",
        help="Directory with patch .npy files and patches_manifest.json.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Root dataset directory for images, metadata, and manifests.",
    )
    p.add_argument(
        "--profiles",
        default="A,B,C",
        help="Comma-separated degradation profile keys.",
    )
    p.add_argument(
        "--agc_modes",
        default="fixed,global",
        help="Comma-separated AGC modes: fixed, global, auto.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--validation",
        action="store_true",
        help="Also save false-colour PNG visualisations in viz/ subdirectory.",
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
    """Run the simulation pipeline (Pass 2).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = build_parser().parse_args(argv)
    log_dir = args.log_dir or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    _, start_time = configure_logging("run_simulation", log_dir, argv=sys.argv[1:])
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    profiles  = [p.strip() for p in args.profiles.split(",")]
    agc_modes = [m.strip() for m in args.agc_modes.split(",")]
    seed      = args.seed

    # ── Validate profiles ─────────────────────────────────────────────────────
    for prof in profiles:
        if prof not in config.DEGRADATION_PROFILES:
            logger.error("Unknown profile '%s'. Valid: %s", prof,
                         list(config.DEGRADATION_PROFILES))
            return log_run_footer(start_time, exit_code=1)

    # ── Output directories ────────────────────────────────────────────────────
    img_dir  = output_dir / "images"
    meta_dir = output_dir / "metadata"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.validation:
        (output_dir / "viz").mkdir(parents=True, exist_ok=True)

    # ── Load patches manifest ─────────────────────────────────────────────────
    manifest_path = input_dir / "patches_manifest.json"
    if not manifest_path.exists():
        logger.error("patches_manifest.json not found in %s", input_dir)
        return log_run_footer(start_time, exit_code=1)

    with open(manifest_path, encoding="utf-8") as fh:
        manifest_data = json.load(fh)
    patch_records: List[Dict] = manifest_data.get("patches", [])

    if not patch_records:
        logger.warning("No patches found in manifest — exiting.")
        return log_run_footer(start_time, exit_code=1)
    logger.info("Loaded %d patch records from manifest.", len(patch_records))

    # ── Load all BT patches for global percentile computation ─────────────────
    all_bt_patches: List[np.ndarray] = []
    for rec in patch_records:
        p = input_dir / rec["patch_file"]
        if p.exists():
            all_bt_patches.append(np.load(p))

    # ── Compute global AGC percentiles ────────────────────────────────────────
    global_t_low: Optional[float] = None
    global_t_high: Optional[float] = None
    if "global" in agc_modes and all_bt_patches:
        logger.info("Computing global BT percentiles across %d patches …",
                    len(all_bt_patches))
        global_t_low, global_t_high = compute_global_percentiles(
            all_bt_patches, seed=seed
        )
        logger.info("  Global AGC bounds: %.2f K – %.2f K",
                    global_t_low, global_t_high)

    # ── Generate FPN maps per profile (deterministic) ─────────────────────────
    fpn_maps: Dict[str, Dict] = {}
    for prof in profiles:
        prof_cfg = config.DEGRADATION_PROFILES[prof]
        fpn_maps[prof] = generate_lepton_fpn(
            column_strength=prof_cfg["fpn_column_strength"],
            pixel_strength=prof_cfg["fpn_pixel_strength"],
            seed=seed + _PROFILE_SEED_OFFSET.get(prof, 0),
        )

    rng = np.random.default_rng(seed)

    # ── Process each patch ────────────────────────────────────────────────────
    label_records:  List[Dict] = []
    sim_records:    List[Dict] = []
    n_fire   = 0
    n_nofire = 0

    n_total_patches = len(patch_records)
    for patch_num, rec in enumerate(patch_records, 1):
        granule_id   = rec["granule_id"]
        patch_index  = rec["patch_index"]
        patch_file   = input_dir / rec["patch_file"]
        nan_frac     = rec.get("nan_fraction", 0.0)
        centroid_lat = rec.get("centroid_lat", float("nan"))
        centroid_lon = rec.get("centroid_lon", float("nan"))

        if patch_num % 50 == 0 or patch_num == n_total_patches:
            logger.info("[%d/%d] simulating patches …", patch_num, n_total_patches)

        if not patch_file.exists():
            logger.warning("Patch file missing: %s — skipped.", patch_file)
            continue

        bt_patch = np.load(patch_file)

        # ── Load delta_T if saved alongside the BT patch ──────────────────────
        delta_t_file = patch_file.parent / patch_file.name.replace(
            "_bt.npy", "_delta_t.npy"
        )
        if delta_t_file.exists():
            delta_t = np.load(delta_t_file)
        else:
            delta_t = np.zeros_like(bt_patch)
            logger.warning(
                "delta_T missing for %s_p%04d — zeros used.",
                granule_id, patch_index,
            )

        # ── Label ─────────────────────────────────────────────────────────────
        # Prefer the pre-computed label stored in the manifest by run_processing.
        # That label was derived from 1km SLSTR pixels before 4×4 aggregation,
        # where fire signals are 16× stronger and thresholds are meaningful.
        # Fall back to post-aggregation detection only for old manifests that
        # pre-date this change.
        stored_label = rec.get("label", None)
        if stored_label is not None:
            label = int(stored_label)
        else:
            label = compute_patch_label(bt_patch, delta_t)
            logger.debug(
                "Fallback label for %s_p%04d: no pre-computed label in manifest.",
                granule_id, patch_index,
            )
        surface_type = classify_patch_surface_type(bt_patch, delta_t, label)
        logger.debug(
            "  patch %s_p%04d: label=%d  surface=%s  nan_frac=%.3f",
            granule_id, patch_index, label, surface_type, nan_frac,
        )

        if label == 1:
            n_fire += 1
        else:
            n_nofire += 1

        base_key = f"{granule_id}_p{patch_index:04d}"

        # One label-manifest entry per physical patch (independent of config)
        label_records.append({
            "patch":        base_key,
            "label":        label,
            "granule_id":   granule_id,
            "surface_type": surface_type,
            "split":        "",  # populated by run_split
        })

        # ── Simulate per AGC mode × profile ───────────────────────────────────
        for agc_mode in agc_modes:
            if agc_mode == "fixed":
                t_min = config.FIXED_TEMP_MIN_K
                t_max = config.FIXED_TEMP_MAX_K
            elif agc_mode == "global":
                if global_t_low is None or global_t_high is None:
                    logger.warning(
                        "Global percentiles unavailable — skipping global mode."
                    )
                    continue
                t_min, t_max = global_t_low, global_t_high
            else:  # "auto": per-patch percentiles
                finite = bt_patch[np.isfinite(bt_patch)]
                if finite.size == 0:
                    continue
                t_min = float(np.percentile(finite, 2.0))
                t_max = float(np.percentile(finite, 98.0))
                if t_max <= t_min:
                    t_max = t_min + 1.0  # avoid zero-range

            agc_range_k   = t_max - t_min
            bt_normalized = apply_agc(bt_patch, t_min, t_max)

            for prof in profiles:
                img_name  = f"{base_key}_{agc_mode}_{prof}"
                img_path  = img_dir / f"{img_name}.png"
                meta_path = meta_dir / f"{img_name}.json"

                # Simulate
                img_u8 = simulate_lepton(
                    bt_normalized,
                    profile=prof,
                    agc_range_k=agc_range_k,
                    fpn=fpn_maps[prof],
                    rng=rng,
                )

                # Save grayscale PNG
                _save_grayscale_png(img_u8, img_path)

                # Save metadata
                meta = build_image_metadata(
                    granule_id=granule_id,
                    patch_index=patch_index,
                    label=label,
                    surface_type=surface_type,
                    agc_mode=agc_mode,
                    profile=prof,
                    t_min=t_min,
                    t_max=t_max,
                    agc_range_k=agc_range_k,
                    nan_fraction=nan_frac,
                    centroid_lat=centroid_lat,
                    centroid_lon=centroid_lon,
                    split="",  # populated by run_split
                    seed=seed,
                )
                save_patch_metadata(meta, meta_path)

                # Optional false-colour visualisation
                if args.validation:
                    from nina_thermal.visualization.colormap import apply_colormap

                    rgb = apply_colormap(img_u8)
                    _save_rgb_png(rgb, output_dir / "viz" / f"{img_name}_viz.png")

                sim_records.append({
                    "image":       f"images/{img_name}.png",
                    "metadata":    f"metadata/{img_name}.json",
                    "granule_id":  granule_id,
                    "patch_index": patch_index,
                    "agc_mode":    agc_mode,
                    "profile":     prof,
                    "label":       label,
                })

    # ── Save label manifest ───────────────────────────────────────────────────
    save_label_manifest(label_records, output_dir / "label_manifest.json")
    logger.info("Wrote label_manifest.json (%d patch records).",
                len(label_records))

    # ── Write simulation manifest ─────────────────────────────────────────────
    sim_manifest = {
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir":  str(args.input_dir),
            "output_dir": str(args.output_dir),
            "profiles":   profiles,
            "agc_modes":  agc_modes,
            "seed":       seed,
        },
        "total_patches": len(label_records),
        "total_images":  len(sim_records),
        "fire_patches":  n_fire,
        "nofire_patches": n_nofire,
        "images": sim_records,
    }
    sim_path = output_dir / "simulation_manifest.json"
    with open(sim_path, "w", encoding="utf-8") as fh:
        json.dump(sim_manifest, fh, indent=2)

    logger.info(
        "Simulation complete: %d patches × %d config(s) = %d images "
        "(fire=%d, nofire=%d). Manifest → %s",
        len(label_records),
        len(profiles) * len(agc_modes),
        len(sim_records),
        n_fire,
        n_nofire,
        sim_path,
    )
    return log_run_footer(start_time, exit_code=0)


# ── Private helpers ───────────────────────────────────────────────────────────

def _save_grayscale_png(img_u8: np.ndarray, path: Path) -> None:
    """Save (H, W) uint8 array as a grayscale PNG."""
    from PIL import Image  # type: ignore
    Image.fromarray(img_u8, mode="L").save(path)


def _save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    """Save (H, W, 3) uint8 array as an RGB PNG."""
    from PIL import Image  # type: ignore
    Image.fromarray(rgb, mode="RGB").save(path)


if __name__ == "__main__":
    sys.exit(main())
