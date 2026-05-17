"""Pipeline stage 3: Tile S2 scenes into 1280×800 patches with cloud labels.

Reads .npy scene files from 02_extracted/, tiles each scene into overlapping
1280×800 windows, computes SCL-based cloud labels, rejects ambiguous tiles,
and saves accepted tiles plus a tiles_manifest.json.

Outputs: 03_tiles/ directory with {scene_id}_tile{idx}_rgb.npy,
         _scl.npy, _meta.json, and tiles_manifest.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from nina_thermal.pipelines._logging import configure_logging, log_run_footer

from nina_cloud import config
from nina_cloud.ingestion.load_scene import load_s2_scene
from nina_cloud.labeling.cloud_label import classify_tile_type, compute_tile_cloud_label
from nina_cloud.processing.spatial import tile_scene_to_patches

logger = logging.getLogger(__name__)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 3: tile scenes and assign cloud labels"
    )
    parser.add_argument("--input_dir",  required=True, help="02_extracted/ directory")
    parser.add_argument("--output_dir", required=True, help="03_tiles/ directory")
    parser.add_argument("--log_dir",    default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_processing", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "extraction_manifest.json"
    if not manifest_path.exists():
        logger.error("extraction_manifest.json not found in %s", input_dir)
        return log_run_footer(start_time, 1)

    with open(manifest_path) as f:
        extraction_manifest = json.load(f)

    scene_ids = [r["scene_id"] for r in extraction_manifest["scenes"] if r["status"] == "ok"]
    logger.info("Processing %d scenes", len(scene_ids))

    tile_records = []
    n_cloud = n_clear = n_ambig = n_nodata_rejected = 0

    for scene_id in scene_ids:
        scene = load_s2_scene(input_dir, scene_id=scene_id)
        if scene is None:
            logger.warning("Could not load scene: %s", scene_id)
            continue

        rgb_patches, scl_patches, origins = tile_scene_to_patches(
            scene["rgb"], scene["scl"]
        )

        for tile_idx, (rgb_tile, scl_tile, (row, col)) in enumerate(
            zip(rgb_patches, scl_patches, origins)
        ):
            label, cloud_frac = compute_tile_cloud_label(scl_tile)
            tile_type = classify_tile_type(label, cloud_frac)

            if label == -1:
                n_ambig += 1
                continue

            # Save tile files
            stem = f"{scene_id}_tile{tile_idx:04d}"
            np.save(output_dir / f"{stem}_rgb.npy", rgb_tile)
            np.save(output_dir / f"{stem}_scl.npy", scl_tile)

            nodata_frac = float(
                np.isin(scl_tile, list(config.SCL_INVALID_CLASSES)).mean()
            )

            meta = {
                "scene_id":   scene_id,
                "tile_idx":   tile_idx,
                "label":      label,
                "tile_type":  tile_type,
                "cloud_frac": round(cloud_frac, 4),
                "nodata_frac": round(nodata_frac, 4),
                "origin_row": row,
                "origin_col": col,
            }
            with open(output_dir / f"{stem}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            tile_records.append(meta)

            if label == 1:
                n_cloud += 1
            else:
                n_clear += 1

    tiles_manifest_path = output_dir / "tiles_manifest.json"
    with open(tiles_manifest_path, "w") as f:
        json.dump(
            {
                "stage": "run_processing",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_cloud": n_cloud, "n_clear": n_clear,
                "n_ambig_rejected": n_ambig,
                "tiles": tile_records,
            },
            f, indent=2,
        )

    logger.info(
        "Tiling complete: cloud=%d  clear=%d  ambig_rejected=%d  manifest=%s",
        n_cloud, n_clear, n_ambig, tiles_manifest_path,
    )
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
