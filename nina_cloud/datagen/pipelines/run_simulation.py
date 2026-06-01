"""Pipeline stage 4: Simulate PAG7936 images from S2 tiles.

For each tile × profile, applies the full degradation chain (PSF blur →
motion blur → shot/read noise → uint8 → FPN → JPEG) and saves the result
as a PNG plus per-image metadata JSON.

Profile selection policy (applied when ``--profiles`` is NOT passed):
  - train / val scenes → profile B only     (saves ~50 % storage)
  - test scenes        → profiles A, B, C   (cross-degradation robustness)

Pass ``--profiles A,B,C`` to override this policy for all scenes.

Outputs: 04_dataset/images/, 04_dataset/metadata/,
         simulation_manifest.json, label_manifest.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from shared.pipeline_logging import configure_logging, log_run_footer
from shared.manifest import expected_scene_ids, assert_chain
from shared.scene_registry import load_registry, split_for
from shared.seeding import derive_seed

from nina_cloud.datagen import config
from nina_cloud.datagen.dataset.export import (
    build_image_metadata,
    save_label_manifest,
    save_tile_image,
    save_tile_metadata,
)
from nina_cloud.datagen.simulation.noise import generate_pag7936_fpn
from nina_cloud.datagen.simulation.sensor import simulate_pag7936
from nina_cloud.datagen.validation.metrics import compute_entropy, compute_snr_laplacian

logger = logging.getLogger(__name__)


def _profiles_for_scene(scene_id: str, registry: dict) -> List[str]:
    """Return the degradation profiles to generate for *scene_id*.

    Policy (config.TRAIN_PROFILES / config.TEST_PROFILES):
      train / val  →  ["B"]
      test         →  ["A", "B", "C"]
    """
    split = split_for(registry, scene_id)
    if split == "test":
        return config.TEST_PROFILES
    return config.TRAIN_PROFILES


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 4: simulate PAG7936 images from tiles"
    )
    parser.add_argument("--input_dir", required=True, help="03_tiles/ directory")
    parser.add_argument("--output_dir", required=True, help="04_dataset/ directory")
    parser.add_argument(
        "--profiles",
        default=None,
        help=(
            "Override degradation profiles for ALL scenes (comma-separated, "
            "e.g. 'A,B,C').  When omitted, profiles are selected per scene "
            "from the registry: train/val → B, test → A,B,C."
        ),
    )
    parser.add_argument(
        "--registry",
        default=None,
        help=(
            "Path to scene_registry.json.  "
            "Defaults to <input_dir>/../../scene_registry.json "
            f"(i.e. data/cloud/{config.REGISTRY_FILENAME}).  "
            "Only needed when --profiles is not passed."
        ),
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--validation", action="store_true", help="Run image quality checks"
    )
    parser.add_argument("--log_dir", default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_simulation", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Resolve profile override vs per-scene policy
    profile_override: Optional[List[str]] = None
    if args.profiles is not None:
        profile_override = [p.strip() for p in args.profiles.split(",")]
        logger.info("Profile override: %s (applied to all scenes)", profile_override)
    else:
        # Per-scene selection — load registry
        if args.registry is not None:
            registry_path = Path(args.registry)
        else:
            registry_path = input_dir.parent / config.REGISTRY_FILENAME
        registry = load_registry(registry_path)
        if not registry:
            logger.warning(
                "Registry not found at %s — all scenes will use train policy (%s)",
                registry_path,
                config.TRAIN_PROFILES,
            )
        else:
            logger.info(
                "Loaded registry: %d scenes — using per-scene profile policy",
                len(registry),
            )

    tiles_manifest_path = input_dir / "tiles_manifest.json"
    if not tiles_manifest_path.exists():
        logger.error("tiles_manifest.json not found in %s", input_dir)
        return log_run_footer(start_time, 1)

    with open(tiles_manifest_path) as f:
        tiles_manifest = json.load(f)

    tile_records = tiles_manifest["tiles"]

    # Compute per-scene profiles upfront for logging
    if profile_override is not None:
        scene_profiles = {rec["scene_id"]: profile_override for rec in tile_records}
    else:
        seen: dict = {}
        for rec in tile_records:
            sid = rec["scene_id"]
            if sid not in seen:
                seen[sid] = _profiles_for_scene(sid, registry)
        scene_profiles = seen

    unique_profile_sets = {tuple(v) for v in scene_profiles.values()}
    logger.info(
        "Simulating %d tiles across %d scenes; profile sets in use: %s",
        len(tile_records),
        len(scene_profiles),
        sorted(unique_profile_sets),
    )

    label_manifest = []
    upstream_ids = expected_scene_ids(tile_records)

    for meta in tile_records:
        scene_id = meta["scene_id"]
        tile_idx = meta["tile_idx"]
        label = meta["label"]
        tile_type = meta["tile_type"]

        stem = f"{scene_id}_tile{tile_idx:04d}"
        rgb_path = input_dir / f"{stem}_rgb.npy"
        if not rgb_path.exists():
            logger.warning("Missing tile file: %s", rgb_path)
            continue

        rgb_float = np.load(rgb_path).astype(np.float32)
        profiles = scene_profiles[scene_id]

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

            img_u8 = simulate_pag7936(rgb_float, profile=profile, fpn=fpn, rng=rng_tile)

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
                    "entropy": compute_entropy(img_u8),
                }

            meta_name = f"{stem}_profile{profile}_meta.json"
            save_tile_metadata(img_meta, metadata_dir / meta_name)

            label_manifest.append(
                {
                    "image": img_name,
                    "metadata": meta_name,
                    "scene_id": scene_id,
                    "tile_idx": tile_idx,
                    "label": label,
                    "tile_type": tile_type,
                    "cloud_frac": meta["cloud_frac"],
                    "profile": profile,
                    "split": "",
                }
            )

    produced_ids = {r["scene_id"] for r in label_manifest}
    if not assert_chain(upstream_ids, produced_ids, logger):
        return log_run_footer(start_time, 1)

    save_label_manifest(label_manifest, output_dir / "label_manifest.json")
    logger.info(
        "Simulation complete: %d images  profile_sets=%s",
        len(label_manifest),
        sorted(unique_profile_sets),
    )
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
