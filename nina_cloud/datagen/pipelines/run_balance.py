"""Pipeline stage 6: Balance dataset to 50/50 cloud:clear per split.

Reads label_manifest.json (with split assignments from run_split.py),
applies undersampling independently within each split, and writes a new
balanced_manifest.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_cloud.datagen import config
from nina_cloud.datagen.dataset.balance import build_balanced_dataset

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 6: balance dataset to 50/50 cloud:clear"
    )
    parser.add_argument("--dataset_dir", required=True, help="04_dataset/ directory")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--log_dir", default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_balance", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "label_manifest.json"
    if not manifest_path.exists():
        logger.error("label_manifest.json not found in %s", dataset_dir)
        return log_run_footer(start_time, 1)

    with open(manifest_path) as f:
        records = json.load(f)

    balanced_all = []
    for split_name in ("train", "val", "test"):
        split_records = [r for r in records if r.get("split") == split_name]
        balanced_split = build_balanced_dataset(
            split_records,
            cloud_ratio=config.CLOUD_RATIO,
            seed=args.seed,
        )
        n_cloud = sum(1 for r in balanced_split if r["label"] == 1)
        n_clear = sum(1 for r in balanced_split if r["label"] == 0)
        logger.info(
            "[%s] before=%d  after=%d  cloud=%d  clear=%d",
            split_name,
            len(split_records),
            len(balanced_split),
            n_cloud,
            n_clear,
        )
        balanced_all.extend(balanced_split)

    balanced_path = dataset_dir / "balanced_manifest.json"
    with open(balanced_path, "w") as f:
        json.dump(
            {
                "stage": "run_balance",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total": len(balanced_all),
                "records": balanced_all,
            },
            f,
            indent=2,
        )

    logger.info(
        "Balance complete: total=%d  manifest=%s", len(balanced_all), balanced_path
    )
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
