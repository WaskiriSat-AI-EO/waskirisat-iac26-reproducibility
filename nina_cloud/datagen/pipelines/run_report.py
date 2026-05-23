"""Pipeline stage 7: Generate dataset_summary.json report.

Reads the balanced_manifest.json and writes a structured summary with all
statistics needed for the IAC paper.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_cloud.datagen.dataset.report import build_dataset_report

logger = logging.getLogger(__name__)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 7: generate dataset_summary.json"
    )
    parser.add_argument("--dataset_dir", required=True, help="04_dataset/ directory")
    parser.add_argument("--log_dir",     default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_report", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    dataset_dir = Path(args.dataset_dir)
    balanced_path = dataset_dir / "balanced_manifest.json"

    if balanced_path.exists():
        with open(balanced_path) as f:
            body = json.load(f)
        records = body.get("records", body) if isinstance(body, dict) else body
        logger.info("Using balanced_manifest.json (%d records)", len(records))
    else:
        label_path = dataset_dir / "label_manifest.json"
        if not label_path.exists():
            logger.error("Neither balanced_manifest.json nor label_manifest.json found")
            return log_run_footer(start_time, 1)
        with open(label_path) as f:
            records = json.load(f)
        logger.info("Using label_manifest.json (%d records)", len(records))

    summary = build_dataset_report(records, dataset_dir / "dataset_summary.json")
    logger.info(
        "Report written: %s  total=%d",
        dataset_dir / "dataset_summary.json", summary["total_images"],
    )
    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
