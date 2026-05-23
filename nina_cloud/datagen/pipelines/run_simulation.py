"""Pipeline stage 4: Simulate PAG7936 images from S2 tiles.

For each tile × profile, applies the full degradation chain (PSF blur →
motion blur → shot/read noise → uint8 → FPN → JPEG) and saves the result
as a PNG plus per-image metadata JSON.

Outputs: 04_dataset/images/, 04_dataset/metadata/,
         simulation_manifest.json, label_manifest.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from shared.pipeline_logging import configure_logging, log_run_footer
from shared.manifest import expected_scene_ids, assert_chain, load_records
from shared.seeding import derive_seed

from nina_cloud.datagen import config
from nina_cloud.datagen.dataset.export import (
    build_image_metadata,
    save_label_manifest,
    save_tile_image,
    save_tile_metadata,
)
from nina_cloud.datagen.labeling.cloud_label import classify_tile_type
from nina_cloud.datagen.simulation.noise import generate_pag7936_fpn
from nina_cloud.datagen.simulation.sensor import simulate_pag7936
from nina_cloud.datagen.validation.metrics import compute_entropy, compute_snr_laplacian

logger = logging.getLogger(__name__)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 4: simulate PAG7936 images from tiles"
    )
    parser.add_argument("--input_dir",  required=True, help="03_tiles/ directory")
    parser.add_argument("--output_dir", required=True, help="04_dataset/ directory")
    parser.add_argument("--profiles",   default="B", help="Degradation profiles (comma-separated)")
    parser.add_argument("--seed",       type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--validation", action="store_true", help="Run image quality checks")
    parser.add_argument("--log_dir",    default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_simulation", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_dir   = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    profiles = [p.strip() for p in args.profiles.split(",")]
    logger.info("Profiles: %s", profiles)

    tiles_manifest_path = input_dir / "tiles_manifest.json"
    if not tiles_manifest_path.exists():
        logger.error("tiles_manifest.json not found in %s", input_dir)
        return log_run_footer(start_time, 1)

    with open(tiles_manifest_path) as f:
        tiles_manifest = json.load(f)

    tile_records = tiles_manifest["tiles"]
    logger.info("Simulating %d tiles × %d profiles", len(tile_records), len(profiles))

    label_manifest = []
    upstream_ids = expected_scene_ids(tile_records)  # no status field in tiles manifest

    for meta in tile_records:
        scene_id  = meta["scene_id"]
        tile_idx  = meta["tile_idx"]
        label     = meta["label"]
        tile_type = meta["tile_type"]

        stem = f"{scene_id}_tile{tile_idx:04d}"
        rgb_path = input_dir / f"{stem}_rgb.npy"
        if not rgb_path.exists():
            logger.warning("Missing tile file: %s", rgb_path)
            continue

        rgb_float = np.load(rgb_path).astype(np.float32)

        for profile in profiles:
            prof_cfg = config.DEGRADATION_PROFILES[profile]
            fpn = generate_pag7936_fpn(
                shape=rgb_float.shape[:2],
                column_strength=prof_cfg["fpn_column_strength"],
                pixel_strength=prof_cfg["fpn_pixel_strength"],
                seed=derive_seed(args.seed, scene_id, tile_idx, profile, "fpn"),
            )
            noise_seed = derive_seed(args.seed, scene_id, tile_idx, profile, "noise")
            rng_tile = np.random.default_rng(noise_seed)

            img_u8 = simulate_pag7936(
                rgb_float, profile=profile, fpn=fpn, rng=rng_tile
            )

            img_name = f"{stem}_profile{profile}.png"
            save_tile_image(img_u8, images_dir / img_name)

            img_meta = build_image_metadata(
                scene_id=scene_id,
                tile_index=tile_idx,
                label=label,
                tile_type=tile_type,
                cloud_frac=meta["cloud_frac"],
                nodata_frac=meta["nodata_frac"],
                profile=profile,
                origin_row=meta["origin_row"],
                origin_col=meta["origin_col"],
                split="",  # filled in by run_split
                seed=noise_seed,
            )
            if args.validation:
                img_meta["validation"] = {
                    "snr_laplacian": compute_snr_laplacian(img_u8),
                    "entropy":       compute_entropy(img_u8),
                }

            meta_name = f"{stem}_profile{profile}_meta.json"
            save_tile_metadata(img_meta, metadata_dir / meta_name)

            label_manifest.append({
                "image":      img_name,
                "metadata":   meta_name,
                "scene_id":   scene_id,
                "tile_idx":   tile_idx,
                "label":      label,
                "tile_type":  tile_type,
                "cloud_frac": meta["cloud_frac"],
                "profile":    profile,
                "split":      "",
            })

    produced_ids = {r["scene_id"] for r in label_manifest}
    if not assert_chain(upstream_ids, produced_ids, logger):
        return log_run_footer(start_time, 1)

    save_label_manifest(label_manifest, output_dir / "label_manifest.json")
    logger.info(
        "Simulation complete: %d images  profiles=%s",
        len(label_manifest), profiles,
    )
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
