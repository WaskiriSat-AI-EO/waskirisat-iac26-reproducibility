"""Pipeline stage 2: Extract S2 bands from .SAFE directories to .npy arrays.

Reads B02/B03/B04 (10m BOA) and SCL (20m) from each .SAFE directory and
saves float32 RGB + uint8 SCL .npy files plus per-scene metadata JSON.

Outputs: 02_extracted/ directory with .npy files and extraction_manifest.json.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from nina_thermal.pipelines._logging import configure_logging, log_run_footer

from nina_cloud.ingestion.extract_bands import extract_s2_scene
from nina_cloud.ingestion.validate_scene import validate_scene
from nina_cloud.ingestion.load_scene import load_s2_scene

logger = logging.getLogger(__name__)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="nina_cloud stage 2: extract S2 bands from .SAFE to .npy"
    )
    parser.add_argument("--input_dir",  required=True, help="01_raw/ directory")
    parser.add_argument("--output_dir", required=True, help="02_extracted/ directory")
    parser.add_argument("--log_dir",    default="data/logs")
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_extraction", log_dir=args.log_dir, argv=sys.argv[1:]
    )

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_dirs = sorted(input_dir.glob("*.SAFE"))
    logger.info("Found %d .SAFE directories in %s", len(safe_dirs), input_dir)

    records = []
    n_ok = n_fail = n_invalid = 0

    for safe_dir in safe_dirs:
        scene_id = extract_s2_scene(safe_dir, output_dir, overwrite=False)
        if scene_id is None:
            logger.warning("Extraction failed: %s", safe_dir.name)
            records.append({"safe_dir": safe_dir.name, "status": "failed"})
            n_fail += 1
            continue

        scene = load_s2_scene(output_dir)
        if scene is None or not validate_scene(scene):
            logger.warning("Validation failed: %s", scene_id)
            records.append({"safe_dir": safe_dir.name, "scene_id": scene_id, "status": "invalid"})
            n_invalid += 1
            continue

        records.append({"safe_dir": safe_dir.name, "scene_id": scene_id, "status": "ok"})
        n_ok += 1

    manifest_path = output_dir / "extraction_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "stage": "run_extraction",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_ok": n_ok, "n_fail": n_fail, "n_invalid": n_invalid,
                "scenes": records,
            },
            f, indent=2,
        )
    logger.info(
        "Extraction complete: ok=%d  fail=%d  invalid=%d  manifest=%s",
        n_ok, n_fail, n_invalid, manifest_path,
    )
    return log_run_footer(start_time, 0 if n_fail == 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
