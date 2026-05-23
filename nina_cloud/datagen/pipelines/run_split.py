"""Pipeline stage 5: Scene-level stratified train/val/test split.

Reads label_manifest.json, groups tiles by scene, performs stratified
split at the scene level, and writes split assignments back to the manifest.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_cloud.datagen import config
from nina_cloud.datagen.labeling.split import scene_level_split

logger = logging.getLogger(__name__)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 5: scene-level train/val/test split"
    )
    parser.add_argument("--dataset_dir", required=True, help="04_dataset/ directory")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--log_dir", default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_split", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "label_manifest.json"
    if not manifest_path.exists():
        logger.error("label_manifest.json not found in %s", dataset_dir)
        return log_run_footer(start_time, 1)

    with open(manifest_path) as f:
        records = json.load(f)

    # Build scene → labels mapping (use unique scene_id, not per-profile)
    scene_labels: dict = {}
    for rec in records:
        sid = rec["scene_id"]
        if sid not in scene_labels:
            scene_labels[sid] = []
        # Only count each tile once (skip duplicate profiles)
        if rec["profile"] == list(config.DEGRADATION_PROFILES.keys())[0]:
            scene_labels[sid].append(rec["label"])

    splits = scene_level_split(scene_labels, seed=args.seed)

    scene_to_split = {}
    for split_name, scene_ids in splits.items():
        for sid in scene_ids:
            scene_to_split[sid] = split_name

    for rec in records:
        rec["split"] = scene_to_split.get(rec["scene_id"], "unassigned")

    # Also update per-image metadata JSON files
    metadata_dir = dataset_dir / "metadata"
    for rec in records:
        meta_path = metadata_dir / rec["metadata"]
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["split"] = rec["split"]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)

    split_counts = {
        s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")
    }
    logger.info("Split complete: %s  manifest=%s", split_counts, manifest_path)
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
