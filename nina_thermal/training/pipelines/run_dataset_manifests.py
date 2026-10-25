"""CLI: write the versioned dataset manifests behind Table 5 (data/manifests/).

Rebuilds the training view exactly as ``train_thermal`` does (merge roots ->
expand to the training profiles -> FRP relabel -> split) and records it:

- ``<prefix>_split_registry.json``: granule id -> split after merge reconciliation
- ``<prefix>_{train,val,test}_patches.tsv``: retained patches with FRP label
- ``<prefix>_firms_query.json``: FIRMS CSVs per root (product, dates, rows)
- ``<prefix>_dataset_summary.json``: frames / granules / fire / no-fire per split

Usage::

    python -m nina_thermal.training.pipelines.run_dataset_manifests \\
        --roots data data_202408 data_202409 data_202509 \\
        --out_dir data/manifests --prefix thermal_v1 \\
        --firms_bbox=-67.5,-20.5,-59.7,-11.8
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nina_thermal.training.dataset import (
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.frp_labels import DEFAULT_FRP_THRESHOLD_MW, apply_frp_labels
from nina_thermal.training.train_thermal import (
    DEFAULT_ROOTS,
    TRAIN_AGC_MODES,
    TRAIN_PROFILES,
)
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
_FIRMS_NAME = re.compile(
    r"firms_(?P<sensor>.+)_(?P<product>NRT|SP)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<days>\d+)d\.csv"
)


def _month(granule_id: str) -> str:
    # Sentinel-3 product name: sensing start YYYYMMDDTHHMMSS at chars 16-31
    stamp = granule_id[16:24]
    return f"{stamp[:4]}-{stamp[4:6]}"


def firms_inventory(roots: list[Path]) -> dict[str, Any]:
    """List the FIRMS CSVs each root's ``firms/`` holds.

    Args:
        roots: Data roots.

    Returns:
        ``{root name: [file entries]}`` with product, date window and row count.
    """
    inventory: dict[str, Any] = {}
    for root in roots:
        files = []
        for path in sorted((root / "firms").glob("*.csv")):
            match = _FIRMS_NAME.fullmatch(path.name)
            with path.open(encoding="utf-8") as f:
                rows = sum(1 for _ in csv.DictReader(f))
            entry: dict[str, Any] = {"file": path.name, "rows": rows}
            if match:
                start = date.fromisoformat(match["start"])
                entry.update(
                    sensor=match["sensor"],
                    product=match["product"],
                    start=start.isoformat(),
                    end=(start + timedelta(days=int(match["days"]) - 1)).isoformat(),
                )
            files.append(entry)
        inventory[root.name] = files
    return inventory


def build_manifests(
    roots: list[str | Path],
    frp_threshold: float = DEFAULT_FRP_THRESHOLD_MW,
    firms_bbox: str | None = None,
) -> dict[str, Any]:
    """Rebuild the training view and return the manifest contents.

    Args:
        roots: Data roots (same set training merged).
        frp_threshold: Summed-FRP relabel threshold [MW].
        firms_bbox: ``W,S,E,N`` the FIRMS CSVs were queried with (recorded only).

    Returns:
        Dict with ``registry``, ``patches`` (per split), ``firms`` and ``summary``.
    """
    roots = [Path(r) for r in roots]
    records, merge_stats = load_merged_records(roots)
    samples = expand_to_samples(records, TRAIN_PROFILES, TRAIN_AGC_MODES)
    samples, frp_summary = apply_frp_labels(samples, roots, frp_threshold)
    by_split = split_samples(samples)

    registry = {
        r["granule_id"]: r["split"]
        for r in sorted(records, key=lambda r: r["granule_id"])
    }
    patches = {
        split: sorted(
            (
                {
                    "patch_id": s["patch"],
                    "granule_id": s["granule_id"],
                    "label": s["label"],
                    "frp_sum_mw": round(float(s["frp_sum"]), 3),
                    "n_detections": s["n_detections"],
                }
                for s in by_split.get(split, [])
            ),
            key=lambda p: p["patch_id"],
        )
        for split in SPLITS
    }
    per_split = {}
    for split, rows in patches.items():
        fire = sum(p["label"] for p in rows)
        granules = {p["granule_id"] for p in rows}
        per_split[split] = {
            "frames": len(rows),
            "granules": len(granules),
            "positive": fire,
            "negative": len(rows) - fire,
            "positive_rate": round(fire / len(rows), 6) if rows else None,
            "granules_by_month": dict(
                sorted(Counter(_month(g) for g in granules).items())
            ),
        }
    all_months = dict(sorted(Counter(_month(g) for g in registry).items()))
    summary = {
        "profiles": TRAIN_PROFILES,
        "agc_modes": TRAIN_AGC_MODES,
        "one_image_per_patch": True,
        "merge": {k: v for k, v in merge_stats.items() if k != "roots"},
        "frp_relabel": frp_summary,
        "splits": per_split,
        "all_granules_by_month": all_months,
    }
    firms = firms_inventory([Path(r) for r in roots])
    # a month is covered when at least one non-empty CSV window touches it
    covered = {
        day.strftime("%Y-%m")
        for files in firms.values()
        for f in files
        if f["rows"] and "start" in f
        for day in (date.fromisoformat(f["start"]), date.fromisoformat(f["end"]))
    }
    return {
        "registry": registry,
        "patches": patches,
        "firms": {
            "queried_bbox_w_s_e_n": firms_bbox,
            "granule_months_without_firms": sorted(set(all_months) - covered),
            "roots": firms,
        },
        "summary": summary,
    }


def write_manifests(result: dict[str, Any], out_dir: Path, prefix: str) -> list[Path]:
    """Write the manifest files; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def dump(name: str, obj: Any) -> None:
        path = out_dir / f"{prefix}_{name}.json"
        path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    dump("split_registry", result["registry"])
    dump("firms_query", result["firms"])
    dump("dataset_summary", result["summary"])
    for split, rows in result["patches"].items():
        path = out_dir / f"{prefix}_{split}_patches.tsv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Write the dataset manifests for Table 5."
    )
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Data roots.")
    parser.add_argument("--out_dir", default="data/manifests", help="Output directory.")
    parser.add_argument("--prefix", default="thermal", help="File name prefix.")
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=DEFAULT_FRP_THRESHOLD_MW,
        help="Summed-FRP relabel threshold [MW]; must match training.",
    )
    parser.add_argument(
        "--firms_bbox", default=None, help="W,S,E,N the FIRMS CSVs were queried with."
    )
    parser.add_argument("--log_dir", default="data/logs", help="Log directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 success, 1 failure).
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "run_dataset_manifests",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )
    try:
        result = build_manifests(args.roots, args.frp_threshold, args.firms_bbox)
        written = write_manifests(result, Path(args.out_dir), args.prefix)
    except Exception:
        logger.exception("Manifest generation failed.")
        return log_run_footer(start_time, exit_code=1)
    for split, s in result["summary"]["splits"].items():
        logger.info(
            "%s: %d frames / %d granules (%d fire, %d no-fire)",
            split,
            s["frames"],
            s["granules"],
            s["positive"],
            s["negative"],
        )
    logger.info("Wrote %d files to %s", len(written), args.out_dir)
    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    sys.exit(main())
