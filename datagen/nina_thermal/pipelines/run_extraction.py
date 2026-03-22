"""Pipeline: extract thermal channels from .SEN3 directories to .npy files.

Usage::

    python -m nina_thermal.pipelines.run_extraction \\
        --input_dir  data/01_raw \\
        --output_dir data/02_extracted

For each ``.SEN3`` directory found in ``input_dir``, calls
:func:`extract_thermal_channels` to produce four ``.npy`` files::

    {prefix}_F2_BT.npy   float32 brightness temperature [K]
    {prefix}_S8_BT.npy   float32 brightness temperature [K]
    {prefix}_lat.npy     float32 latitude  [degrees]
    {prefix}_lon.npy     float32 longitude [degrees]

Writes ``extraction_manifest.json`` to ``output_dir`` on completion.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from nina_thermal.ingestion.extract_channels import extract_thermal_channels
from nina_thermal import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_extraction",
        description="Extract F2_BT, S8_BT, lat, lon from .SEN3 directories to .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir",
        required=True,
        metavar="DIR",
        help="Directory containing .SEN3 sub-directories (output of run_acquisition).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory for extracted .npy files and extraction_manifest.json.",
    )
    return p


def main(argv=None) -> int:
    """Run the extraction pipeline.

    Returns:
        Exit code: 0 on success, 1 if no .SEN3 directories were found.
    """
    args = build_parser().parse_args(argv)
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sen3_dirs = sorted(input_dir.glob("*.SEN3"))
    if not sen3_dirs:
        logger.warning("No .SEN3 directories found in %s.", input_dir)
        _write_manifest(output_dir, [], args)
        return 1

    logger.info("Found %d .SEN3 directories.", len(sen3_dirs))

    records = []
    for i, sen3_dir in enumerate(sen3_dirs, 1):
        logger.info("[%d/%d] Extracting %s …", i, len(sen3_dirs), sen3_dir.name)

        result = extract_thermal_channels(sen3_dir, output_dir)
        if result is None:
            logger.warning("Extraction failed for %s — skipped.", sen3_dir.name)
            continue

        p_f2, p_s8, p_lat, p_lon = result
        # Derive prefix: filename stem of F2_BT file minus the _F2_BT suffix
        prefix = p_f2.stem[: -len("_F2_BT")]
        records.append({
            "granule_id":  sen3_dir.stem,
            "prefix":      prefix,
            "sen3_dir":    str(sen3_dir),
            "npy_files": {
                "F2_BT": str(p_f2),
                "S8_BT": str(p_s8),
                "lat":   str(p_lat),
                "lon":   str(p_lon),
            },
        })

    _write_manifest(output_dir, records, args)
    logger.info(
        "Extraction complete: %d/%d granules extracted.  "
        "Manifest → %s/extraction_manifest.json",
        len(records), len(sen3_dirs), output_dir,
    )
    return 0


def _write_manifest(output_dir: Path, records: list, args) -> None:
    manifest = {
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "args": {
            "input_dir":  args.input_dir,
            "output_dir": args.output_dir,
        },
        "granules":         records,
        "total_extracted":  len(records),
    }
    path = output_dir / "extraction_manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
