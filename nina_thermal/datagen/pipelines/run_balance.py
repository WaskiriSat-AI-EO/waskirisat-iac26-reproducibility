"""Pipeline: class-balance the dataset within each split.

Usage::

    python -m nina_thermal.pipelines.run_balance \\
        --dataset_dir data/04_dataset \\
        --fire_ratio  0.20 \\
        --seed        42

Reads ``split_manifest.json`` and ``label_manifest.json`` from
``dataset_dir``.  For each split independently:

1. Filter label-manifest records to the granules in that split.
2. :func:`build_balanced_dataset` — undersample no-fire to reach
   *fire_ratio*.  Fire patches are **never** removed.
3. Mark excluded per-image metadata JSONs in ``metadata/`` with
   ``"excluded_by_balance": true``.

Writes ``balanced_manifest.json`` (retained records only, all splits
concatenated).

NOTE: image files are never deleted — balance is a manifest-level filter.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from nina_thermal.datagen import config
from nina_thermal.datagen.dataset.balance import build_balanced_dataset
from nina_thermal.datagen.dataset.export import patch_key
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_balance",
        description=(
            "Class-balance the dataset within each split by undersampling "
            "no-fire patches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset_dir",
        required=True,
        metavar="DIR",
        help="Root dataset directory (output of run_split).",
    )
    p.add_argument(
        "--fire_ratio",
        type=float,
        default=config.FIRE_RATIO,
        help="Target fire fraction (fire / total). Default from config.",
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
    """Balance each split and write balanced_manifest.json.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = build_parser().parse_args(argv)
    log_dir = (
        args.log_dir
        or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    )
    _, start_time = configure_logging("run_balance", log_dir, argv=sys.argv[1:])
    dataset_dir = Path(args.dataset_dir)
    meta_dir = dataset_dir / "metadata"

    # ── Load split manifest ───────────────────────────────────────────────────
    split_path = dataset_dir / "split_manifest.json"
    if not split_path.exists():
        logger.error("split_manifest.json not found in %s", dataset_dir)
        return log_run_footer(start_time, exit_code=1)

    with open(split_path, encoding="utf-8") as fh:
        split_manifest = json.load(fh)

    split_granules: Dict[str, List[str]] = {
        s: split_manifest.get(s, []) for s in ("train", "val", "test")
    }

    # ── Load label manifest ───────────────────────────────────────────────────
    manifest_path = dataset_dir / "label_manifest.json"
    if not manifest_path.exists():
        logger.error("label_manifest.json not found in %s", dataset_dir)
        return log_run_footer(start_time, exit_code=1)

    with open(manifest_path, encoding="utf-8") as fh:
        label_records: List[Dict] = json.load(fh)

    # ── Balance each split independently ─────────────────────────────────────
    all_retained: List[Dict] = []
    excluded_patches: Set[str] = set()

    for split_name, granule_ids in split_granules.items():
        if not granule_ids:
            logger.info("  Split '%s': no granules — skipped.", split_name)
            continue

        gid_set = set(granule_ids)
        split_records = [r for r in label_records if r["granule_id"] in gid_set]

        if not split_records:
            logger.info("  Split '%s': no records — skipped.", split_name)
            continue

        logger.info(
            "  Split '%s': %d records before balance …", split_name, len(split_records)
        )

        balanced = build_balanced_dataset(
            split_records,
            fire_ratio=args.fire_ratio,
            seed=args.seed,
        )

        retained_patches = {r["patch"] for r in balanced}
        for rec in split_records:
            if rec["patch"] not in retained_patches:
                excluded_patches.add(rec["patch"])

        all_retained.extend(balanced)

    # ── Mark excluded patches in per-image metadata ───────────────────────────
    n_marked = 0
    if excluded_patches and meta_dir.exists():
        for meta_file in meta_dir.glob("*.json"):
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    meta = json.load(fh)
                key = patch_key(meta.get("granule_id", ""), meta.get("patch_index", 0))
                if key in excluded_patches:
                    meta["excluded_by_balance"] = True
                    with open(meta_file, "w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=2)
                    n_marked += 1
            except Exception as exc:
                logger.warning("Could not update %s: %s", meta_file.name, exc)

    if n_marked:
        logger.info(
            "Marked %d per-image metadata files as excluded_by_balance.",
            n_marked,
        )

    # ── Write balanced manifest ───────────────────────────────────────────────
    balanced_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "fire_ratio": args.fire_ratio,
        "seed": args.seed,
        "total_retained": len(all_retained),
        "total_excluded": len(excluded_patches),
        "records": all_retained,
    }
    balanced_path = dataset_dir / "balanced_manifest.json"
    with open(balanced_path, "w", encoding="utf-8") as fh:
        json.dump(balanced_manifest, fh, indent=2)

    n_fire = sum(1 for r in all_retained if r.get("label") == 1)
    n_nofire = len(all_retained) - n_fire
    achieved = n_fire / len(all_retained) if all_retained else 0.0

    logger.info(
        "Balance complete: %d retained (fire=%d, nofire=%d, ratio=%.3f), "
        "%d excluded. → %s",
        len(all_retained),
        n_fire,
        n_nofire,
        achieved,
        len(excluded_patches),
        balanced_path,
    )
    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    sys.exit(main())
