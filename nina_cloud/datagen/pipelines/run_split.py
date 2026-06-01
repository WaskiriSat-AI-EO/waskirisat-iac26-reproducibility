"""Pipeline stage 5: Stamp train/val/test splits from the scene registry.

Reads label_manifest.json, assigns each record a split via the scene
registry (deterministic hash — no reshuffle across runs or operators),
and writes the split assignments back to the manifest and to each
per-image metadata JSON.

If the registry file is absent (first run or registry not yet adopted)
every scene falls back to the deterministic ``assign_split`` hash with a
warning — no crash, no data corruption.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from shared.pipeline_logging import configure_logging, log_run_footer
from shared.scene_registry import load_registry

from nina_cloud.datagen import config
from nina_cloud.datagen.labeling.split import assign_splits_from_registry

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 5: stamp train/val/test splits from registry"
    )
    parser.add_argument("--dataset_dir", required=True, help="04_dataset/ directory")
    parser.add_argument(
        "--registry",
        default=None,
        help=(
            "Path to scene_registry.json.  "
            "Defaults to <dataset_dir>/../../scene_registry.json "
            f"(i.e. data/cloud/{config.REGISTRY_FILENAME})."
        ),
    )
    # --seed is accepted but ignored; split is deterministic from scene_id.
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="(deprecated — split no longer uses RNG; kept for CLI compatibility)",
    )
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

    # Resolve registry path
    if args.registry is not None:
        registry_path = Path(args.registry)
    else:
        registry_path = dataset_dir.parent / config.REGISTRY_FILENAME

    registry = load_registry(registry_path)
    if not registry:
        logger.warning(
            "Registry not found at %s — all scenes use deterministic assign_split fallback",
            registry_path,
        )
    else:
        logger.info("Loaded registry: %d scenes from %s", len(registry), registry_path)

    with open(manifest_path) as f:
        records = json.load(f)

    assign_splits_from_registry(records, registry)

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
