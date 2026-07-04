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

from nina_thermal.datagen import config
from nina_thermal.datagen.labeling.split import (
    granule_level_split,
    invert_splits,
    log_split_summary,
)
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_split",
        description=("Assign granule-level train/val/test splits and update metadata."),
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
    """Assign granule-level splits and update metadata.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = build_parser().parse_args(argv)
    log_dir = (
        args.log_dir
        or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    )
    _, start_time = configure_logging("run_split", log_dir, argv=sys.argv[1:])
    dataset_dir = Path(args.dataset_dir)
    meta_dir = dataset_dir / "metadata"

    # ── Load label manifest ───────────────────────────────────────────────────
    manifest_path = dataset_dir / "label_manifest.json"
    if not manifest_path.exists():
        logger.error("label_manifest.json not found in %s", dataset_dir)
        return log_run_footer(start_time, exit_code=1)

    with open(manifest_path, encoding="utf-8") as fh:
        label_records: List[Dict] = json.load(fh)

    if not label_records:
        logger.warning("label_manifest.json is empty — nothing to split.")
        return log_run_footer(start_time, exit_code=1)

    # ── Build granule → labels mapping ───────────────────────────────────────
    granule_labels: Dict[str, List[int]] = {}
    for rec in label_records:
        gid = rec["granule_id"]
        granule_labels.setdefault(gid, []).append(rec["label"])

    logger.info("Found %d unique granules.", len(granule_labels))

    # ── Perform (or reuse) granule-level split ────────────────────────────────
    # run_simulation is the split authority and writes split_manifest.json
    # before run_split is called.  When that file exists, reuse it verbatim to
    # guarantee the split stamped into metadata matches the policy already baked
    # into which images were generated.  Fall back to recomputing if absent, or
    # if it's stale relative to the current label_manifest.json (its granule
    # set no longer matches — e.g. granules were added/removed since it was
    # written), rather than silently mislabelling new granules as "train" or
    # crashing on granules that no longer exist (backward compatibility with
    # older dataset directories is handled by the same fallback).
    existing_split_path = dataset_dir / "split_manifest.json"
    splits = None
    effective_seed = args.seed
    if existing_split_path.exists():
        with open(existing_split_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        candidate_splits = {
            "train": existing.get("train", []),
            "val": existing.get("val", []),
            "test": existing.get("test", []),
        }
        split_granules = {gid for gids in candidate_splits.values() for gid in gids}
        current_granules = set(granule_labels)
        is_disjoint = sum(len(gids) for gids in candidate_splits.values()) == len(
            split_granules
        )
        if split_granules == current_granules and is_disjoint:
            splits = candidate_splits
            existing_seed = existing.get("seed")
            if existing_seed is not None and existing_seed != args.seed:
                logger.warning(
                    "split_manifest.json was computed with seed=%s; "
                    "requested --seed=%s is ignored since the manifest is "
                    "being reused verbatim.",
                    existing_seed,
                    args.seed,
                )
                effective_seed = existing_seed
            logger.info(
                "Reusing split_manifest.json written by run_simulation "
                "(train=%d, val=%d, test=%d granules).",
                len(splits["train"]),
                len(splits["val"]),
                len(splits["test"]),
            )
        elif not is_disjoint:
            logger.warning(
                "split_manifest.json is corrupt — a granule appears in more "
                "than one split (%d entries across splits vs %d unique "
                "granules).  Recomputing the split from scratch instead of "
                "trusting it.",
                sum(len(gids) for gids in candidate_splits.values()),
                len(split_granules),
            )
        else:
            logger.warning(
                "split_manifest.json is stale — its granule set (%d) does not "
                "match label_manifest.json's current granule set (%d; "
                "missing=%d, extra=%d).  Recomputing the split from scratch "
                "instead of trusting it.",
                len(split_granules),
                len(current_granules),
                len(current_granules - split_granules),
                len(split_granules - current_granules),
            )

    if splits is None:
        splits = granule_level_split(granule_labels, seed=args.seed)
        effective_seed = args.seed
        logger.info(
            "split_manifest.json not found or stale — computed split from "
            "scratch (seed=%d).",
            args.seed,
        )

    # Log per-split summary (granule count, patch count, fire fraction)
    log_split_summary(splits, granule_labels, logger)
    logger.info("  total: %d granules", len(granule_labels))

    # Build reverse map: granule_id → split_name
    granule_to_split: Dict[str, str] = invert_splits(splits)

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
                gid = meta.get("granule_id", "")
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
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "seed": effective_seed,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    split_path = dataset_dir / "split_manifest.json"
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)

    n_total = sum(len(v) for v in splits.values())
    logger.info(
        "Wrote split_manifest.json: train=%d, val=%d, test=%d granules " "(total=%d).",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        n_total,
    )
    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    sys.exit(main())
