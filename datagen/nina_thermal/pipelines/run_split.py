"""Pipeline: assign granule-level train/val/test splits.

Usage::

    python -m nina_thermal.pipelines.run_split \\
        --dataset_dir data/04_dataset \\
        --seed        42

Reads ``label_manifest.json`` from ``dataset_dir``, calls
:func:`granule_level_split`, then:

* Updates every per-image metadata JSON in ``metadata/`` with a ``"split"``
  field.
* Rewrites ``label_manifest.json`` with the ``"split"`` field populated.
* Writes ``split_manifest.json``:
  ``{"train": [...], "val": [...], "test": [...]}``

CRITICAL: splits are at granule level — never at patch level.  Patches from
the same granule share atmospheric conditions, sun angle, and land surface
state; patch-level splitting produces optimistically biased metrics due to
spatial autocorrelation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from nina_thermal import config
from nina_thermal.labeling.split import granule_level_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_split",
        description=(
            "Assign granule-level train/val/test splits and update metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset_dir",
        required=True,
        metavar="DIR",
        help="Root dataset directory (output of run_simulation).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return p


def main(argv=None) -> int:
    """Assign granule-level splits and update metadata.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args      = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    meta_dir    = dataset_dir / "metadata"

    # ── Load label manifest ───────────────────────────────────────────────────
    manifest_path = dataset_dir / "label_manifest.json"
    if not manifest_path.exists():
        logger.error("label_manifest.json not found in %s", dataset_dir)
        return 1

    with open(manifest_path, encoding="utf-8") as fh:
        label_records: List[Dict] = json.load(fh)

    if not label_records:
        logger.warning("label_manifest.json is empty — nothing to split.")
        return 1

    # ── Build granule → labels mapping ───────────────────────────────────────
    granule_labels: Dict[str, List[int]] = {}
    for rec in label_records:
        gid = rec["granule_id"]
        granule_labels.setdefault(gid, []).append(rec["label"])

    logger.info("Found %d unique granules.", len(granule_labels))

    # ── Perform granule-level split ───────────────────────────────────────────
    splits = granule_level_split(granule_labels, seed=args.seed)

    # Log per-split summary (granule count, patch count, fire fraction)
    for split_name, gids in splits.items():
        patch_labels = [
            lbl
            for gid in gids
            for lbl in granule_labels[gid]
        ]
        n_patches = len(patch_labels)
        fire_frac = (sum(patch_labels) / n_patches) if n_patches else 0.0
        logger.info(
            "  %s: %d granules / %d patches / fire_frac=%.3f",
            split_name,
            len(gids),
            n_patches,
            fire_frac,
        )
    logger.info("  total: %d granules", len(granule_labels))

    # Build reverse map: granule_id → split_name
    granule_to_split: Dict[str, str] = {}
    for split_name, gids in splits.items():
        for gid in gids:
            granule_to_split[gid] = split_name

    # ── Update label manifest with split field ────────────────────────────────
    for rec in label_records:
        rec["split"] = granule_to_split.get(rec["granule_id"], "train")

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(label_records, fh, indent=2)
    logger.info("Updated label_manifest.json with split assignments.")

    # ── Update per-image metadata JSONs ──────────────────────────────────────
    n_updated = 0
    if meta_dir.exists():
        for meta_file in meta_dir.glob("*.json"):
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    meta = json.load(fh)
                gid        = meta.get("granule_id", "")
                split_name = granule_to_split.get(gid, "train")
                meta["split"] = split_name
                with open(meta_file, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
                n_updated += 1
            except Exception as exc:
                logger.warning("Could not update %s: %s", meta_file.name, exc)

    logger.info("Updated %d per-image metadata JSON files.", n_updated)

    # ── Write split manifest ──────────────────────────────────────────────────
    split_manifest = {
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "seed":           args.seed,
        "train":          splits["train"],
        "val":            splits["val"],
        "test":           splits["test"],
    }
    split_path = dataset_dir / "split_manifest.json"
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)

    n_total = sum(len(v) for v in splits.values())
    logger.info(
        "Wrote split_manifest.json: train=%d, val=%d, test=%d granules "
        "(total=%d).",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        n_total,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
