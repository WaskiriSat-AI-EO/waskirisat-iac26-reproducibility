"""Pipeline Pass 2: simulate FLIR Lepton images, compute labels, export dataset.

Usage::

    python -m nina_thermal.pipelines.run_simulation \\
        --input_dir  data/03_patches \\
        --output_dir data/04_dataset \\
        --seed       42

Per-split profile / AGC policy (enforced automatically when ``--profiles`` /
``--agc_modes`` are omitted):

    split   profiles   AGC modes
    ------  ---------  ------------------
    train   B only     fixed, global
    val     B only     fixed
    test    A, B, C    fixed

Rationale:
- Profiles A/B/C are fixed-seed deterministic noise variants of the same BT
  patch.  Training on all three duplicates spatial content with memorisable
  noise patterns — not genuine augmentation.  A & C belong in the test split
  where they quantify cross-degradation robustness for the IAC paper table.
- ``global`` AGC is a train-only contrast augmentation.  ``fixed`` is the
  confirmed flight inference configuration and the only AGC mode val/test may
  use, ensuring reported metrics correspond to real deployment.

Pass ``--profiles A,B,C --agc_modes fixed,global`` to override the policy for
all splits (useful for debugging / ad-hoc full generation).

This stage also writes ``split_manifest.json`` (the split authority) by running
the same ``granule_level_split`` used by ``run_split``.  ``run_split`` is then
idempotent: if ``split_manifest.json`` already exists, it reuses it.

For each patch record in ``patches_manifest.json``:

1. Determine the patch's split (train / val / test) from the granule-level split
   computed in the pre-pass.
2. Apply the per-split profile / AGC policy (or CLI override).
3. For each selected AGC mode × profile:

   a. :func:`apply_agc` to normalise BT.
   b. :func:`simulate_lepton` to apply the full degradation chain.
   c. Save grayscale PNG to ``images/`` and JSON metadata to ``metadata/``.
   d. Optionally save false-colour PNG to ``viz/`` (requires ``--validation``).

Writes ``label_manifest.json``, ``simulation_manifest.json``, and
``split_manifest.json`` to ``output_dir``.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.datagen.dataset.export import (
    build_image_metadata,
    save_label_manifest,
    save_patch_metadata,
)
from nina_thermal.datagen.labeling.fire_label import (
    classify_patch_surface_type,
    compute_patch_label,
)
from nina_thermal.datagen.labeling.split import (
    granule_level_split,
    invert_splits,
    log_split_summary,
)
from nina_thermal.datagen.processing.agc import apply_agc, compute_global_percentiles
from nina_thermal.datagen.simulation.noise import generate_lepton_fpn
from nina_thermal.datagen.simulation.sensor import simulate_lepton
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)

# Manifests and subdirectories written by this stage
SKIP_NAMES = {
    "label_manifest.json",
    "simulation_manifest.json",
    "split_manifest.json",
    "dataset_summary.json",
}

# Deterministic profile -> FPN seed offset (avoids hash() randomness)
_PROFILE_SEED_OFFSET = {"A": 0, "B": 1, "C": 2}


def _policy_for_split(
    split: str,
    profile_override: Optional[List[str]] = None,
    agc_override: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Return (profiles, agc_modes) for *split* per the IAC paper methodology,
    or the CLI override values when given (applied uniformly across splits).

    Policy:
        train → profile B,      AGC {fixed, global}
        val   → profile B,      AGC {fixed}
        test  → profiles A,B,C, AGC {fixed}
    """
    if split == "test":
        base_profiles, base_agc_modes = config.TEST_PROFILES, config.EVAL_AGC_MODES
    elif split == "val":
        base_profiles, base_agc_modes = config.TRAIN_PROFILES, config.EVAL_AGC_MODES
    else:  # train
        base_profiles, base_agc_modes = config.TRAIN_PROFILES, config.TRAIN_AGC_MODES
    return (
        profile_override if profile_override is not None else base_profiles,
        agc_override if agc_override is not None else base_agc_modes,
    )


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
        default=None,
        metavar="PROFILES",
        help=(
            "Override degradation profiles for ALL splits (comma-separated, "
            "e.g. 'A,B,C').  When omitted, the per-split policy is used: "
            "train/val → B, test → A,B,C."
        ),
    )
    p.add_argument(
        "--agc_modes",
        default=None,
        metavar="AGC_MODES",
        help=(
            "Override AGC modes for ALL splits (comma-separated, e.g. "
            "'fixed,global').  When omitted, the per-split policy is used: "
            "train → fixed,global; val/test → fixed."
        ),
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
    log_dir = (
        args.log_dir
        or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    )
    _, start_time = configure_logging("run_simulation", log_dir, argv=sys.argv[1:])
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    seed = args.seed

    # ── CLI override vs per-split policy ─────────────────────────────────────
    profile_override: Optional[List[str]] = None
    agc_override: Optional[List[str]] = None

    if args.profiles is not None:
        profile_override = [p.strip() for p in args.profiles.split(",")]
        for prof in profile_override:
            if prof not in config.DEGRADATION_PROFILES:
                logger.error(
                    "Unknown profile '%s'. Valid: %s",
                    prof,
                    list(config.DEGRADATION_PROFILES),
                )
                return log_run_footer(start_time, exit_code=1)
        logger.info(
            "Profile override: %s (applied to all splits — per-split policy disabled)",
            profile_override,
        )

    if args.agc_modes is not None:
        agc_override = [m.strip() for m in args.agc_modes.split(",")]
        for mode in agc_override:
            if mode not in config.VALID_AGC_MODES:
                logger.error(
                    "Unknown AGC mode '%s'. Valid: %s",
                    mode,
                    config.VALID_AGC_MODES,
                )
                return log_run_footer(start_time, exit_code=1)
        logger.info(
            "AGC mode override: %s (applied to all splits — per-split policy disabled)",
            agc_override,
        )

    if profile_override is None and agc_override is None:
        logger.info(
            "Per-split policy active: "
            "train→(B, fixed+global)  val→(B, fixed)  test→(A,B,C, fixed)"
        )

    # ── Output directories ────────────────────────────────────────────────────
    img_dir = output_dir / "images"
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

    # ── Resolve the effective label for every patch, once ──────────────────────
    # Prefer the pre-computed label stored in the manifest by run_processing.
    # Fall back to compute_patch_label for records that pre-date it (older
    # manifests). This is the SAME resolution the per-patch loop below uses —
    # computed once here so the granule-level split's fire-fraction
    # stratification is stratified on the same labels the loop later stamps
    # into label_manifest.json / metadata, instead of silently defaulting
    # unlabelled patches to 0 for stratification purposes only.
    patch_labels: List[int] = []
    for rec in patch_records:
        stored_label = rec.get("label", None)
        if stored_label is not None:
            patch_labels.append(int(stored_label))
            continue
        patch_file = input_dir / rec["patch_file"]
        if not patch_file.exists():
            patch_labels.append(0)
            continue
        bt_patch = np.load(patch_file)
        delta_t_file = patch_file.parent / patch_file.name.replace(
            "_bt.npy", "_delta_t.npy"
        )
        delta_t = (
            np.load(delta_t_file) if delta_t_file.exists() else np.zeros_like(bt_patch)
        )
        patch_labels.append(compute_patch_label(bt_patch, delta_t))

    # ── Granule-level split (run_simulation is the split authority) ───────────
    granule_labels: Dict[str, List[int]] = {}
    for rec, label in zip(patch_records, patch_labels):
        granule_labels.setdefault(rec["granule_id"], []).append(label)

    splits = granule_level_split(granule_labels, seed=seed)
    granule_to_split: Dict[str, str] = invert_splits(splits)

    # Log per-split summary
    log_split_summary(splits, granule_labels, logger)

    # Write split_manifest.json (authority consumed by run_split)
    split_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "seed": seed,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    split_manifest_path = output_dir / "split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)
    logger.info(
        "Wrote split_manifest.json: train=%d, val=%d, test=%d granules.",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    # ── Global AGC percentiles from TRAIN patches only ────────────────────────
    # 'global' AGC is a train-only augmentation; deriving its bounds from train
    # data avoids leaking test BT statistics into a training-time transform.
    train_granules = set(splits["train"])
    train_patches = [
        np.load(input_dir / rec["patch_file"])
        for rec in patch_records
        if rec["granule_id"] in train_granules
        and (input_dir / rec["patch_file"]).exists()
    ]

    global_t_low: Optional[float] = None
    global_t_high: Optional[float] = None

    # Determine if global AGC will be used at all under the effective policy
    uses_global = (agc_override is not None and "global" in agc_override) or (
        agc_override is None and "global" in config.TRAIN_AGC_MODES
    )
    if uses_global and train_patches:
        logger.info(
            "Computing global BT percentiles from %d train patch(es) …",
            len(train_patches),
        )
        global_t_low, global_t_high = compute_global_percentiles(
            train_patches, seed=seed
        )
        logger.info(
            "  Global AGC bounds (train): %.2f K – %.2f K",
            global_t_low,
            global_t_high,
        )

    # ── Pre-compute all profiles that appear across any split ─────────────────
    all_profiles: List[str] = (
        profile_override
        if profile_override is not None
        else list(config.DEGRADATION_PROFILES.keys())  # union of A, B, C
    )

    fpn_maps: Dict[str, Dict] = {}
    for prof in all_profiles:
        prof_cfg = config.DEGRADATION_PROFILES[prof]
        fpn_maps[prof] = generate_lepton_fpn(
            column_strength=prof_cfg["fpn_column_strength"],
            pixel_strength=prof_cfg["fpn_pixel_strength"],
            seed=seed + _PROFILE_SEED_OFFSET.get(prof, 0),
        )

    rng = np.random.default_rng(seed)

    # ── Process each patch ────────────────────────────────────────────────────
    label_records: List[Dict] = []
    sim_records: List[Dict] = []
    n_fire = 0
    n_nofire = 0

    # Per-split image counts for the manifest summary
    split_image_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    n_total_patches = len(patch_records)
    for patch_num, (rec, label) in enumerate(zip(patch_records, patch_labels), 1):
        granule_id = rec["granule_id"]
        patch_index = rec["patch_index"]
        patch_file = input_dir / rec["patch_file"]
        nan_frac = rec.get("nan_fraction", 0.0)
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
                granule_id,
                patch_index,
            )

        # ── Label (resolved once, up-front, in the patch_labels pre-pass) ──────
        surface_type = classify_patch_surface_type(bt_patch, delta_t, label)
        logger.debug(
            "  patch %s_p%04d: label=%d  surface=%s  nan_frac=%.3f",
            granule_id,
            patch_index,
            label,
            surface_type,
            nan_frac,
        )

        if label == 1:
            n_fire += 1
        else:
            n_nofire += 1

        # ── Resolve split and effective policy ────────────────────────────────
        split = granule_to_split.get(granule_id, "train")
        profiles_eff, agc_modes_eff = _policy_for_split(
            split, profile_override, agc_override
        )

        base_key = f"{granule_id}_p{patch_index:04d}"

        # One label-manifest entry per physical patch (independent of config)
        label_records.append(
            {
                "patch": base_key,
                "label": label,
                "granule_id": granule_id,
                "surface_type": surface_type,
                "split": split,  # populated here — run_split will reuse this
            }
        )

        # ── Simulate per AGC mode × profile ───────────────────────────────────
        for agc_mode in agc_modes_eff:
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
                    t_max = t_min + 1.0

            agc_range_k = t_max - t_min
            bt_normalized = apply_agc(bt_patch, t_min, t_max)

            for prof in profiles_eff:
                img_name = f"{base_key}_{agc_mode}_{prof}"
                img_path = img_dir / f"{img_name}.png"
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
                    split=split,
                    seed=seed,
                )
                save_patch_metadata(meta, meta_path)

                # Optional false-colour visualisation
                if args.validation:
                    from nina_thermal.datagen.visualization.colormap import (
                        apply_colormap,
                    )

                    rgb = apply_colormap(img_u8)
                    _save_rgb_png(rgb, output_dir / "viz" / f"{img_name}_viz.png")

                sim_records.append(
                    {
                        "image": f"images/{img_name}.png",
                        "metadata": f"metadata/{img_name}.json",
                        "granule_id": granule_id,
                        "patch_index": patch_index,
                        "agc_mode": agc_mode,
                        "profile": prof,
                        "label": label,
                        "split": split,
                    }
                )
                split_image_counts[split] = split_image_counts.get(split, 0) + 1

    # ── Save label manifest ───────────────────────────────────────────────────
    save_label_manifest(label_records, output_dir / "label_manifest.json")
    logger.info("Wrote label_manifest.json (%d patch records).", len(label_records))

    # ── Write simulation manifest ─────────────────────────────────────────────
    policy_description = (
        {
            "train": {"profiles": config.TRAIN_PROFILES, "agc_modes": config.TRAIN_AGC_MODES},
            "val":   {"profiles": config.TRAIN_PROFILES, "agc_modes": config.EVAL_AGC_MODES},
            "test":  {"profiles": config.TEST_PROFILES,  "agc_modes": config.EVAL_AGC_MODES},
        }
        if (profile_override is None and agc_override is None)
        else "override"
    )
    sim_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "profiles": profile_override or "per-split-policy",
            "agc_modes": agc_override or "per-split-policy",
            "policy": policy_description,
            "seed": seed,
        },
        "total_patches": len(label_records),
        "total_images": len(sim_records),
        "images_by_split": split_image_counts,
        "fire_patches": n_fire,
        "nofire_patches": n_nofire,
        "images": sim_records,
    }
    sim_path = output_dir / "simulation_manifest.json"
    with open(sim_path, "w", encoding="utf-8") as fh:
        json.dump(sim_manifest, fh, indent=2)

    logger.info(
        "Simulation complete: %d patches → %d images "
        "(train=%d, val=%d, test=%d | fire_patches=%d, nofire_patches=%d). "
        "Manifest → %s",
        len(label_records),
        len(sim_records),
        split_image_counts.get("train", 0),
        split_image_counts.get("val", 0),
        split_image_counts.get("test", 0),
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
