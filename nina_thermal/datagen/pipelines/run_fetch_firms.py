"""Pipeline: download the FIRMS CSVs that label a data root.

Usage::

    python -m nina_thermal.datagen.pipelines.run_fetch_firms \\
        --start 20240910 --end 20240930 --out_dir data_202409b/firms

Fetches VIIRS active-fire detections for ``[start − 1 day, end + 1 day]`` over
the label bbox (``config.FIRMS_LABEL_BBOX``, which contains every patch
footprint) and writes ``firms_{source}_{date}_{n}d.csv`` files plus
``coverage.json`` (read by ``training/frp_labels.py``).

Rules enforced, each from a v3 label bug (``docs/TRAINING_REPORT_v3.md`` §2.2):

* one product per date: ``VIIRS_SNPP_SP``, ``VIIRS_NOAA20_SP`` for the
  Suomi-NPP outage; NRT products are not accepted, and the run refuses to add
  a second product to a date that already has a CSV;
* the queried bbox is declared in ``coverage.json`` and must not change
  within a root;
* a request that returns no detections over this ~5 Mkm² box is treated as
  missing data and fails the run (``--allow_empty`` to override), as does a
  window outside the product's published availability.

Re-running is safe: CSVs already on disk are kept (``--overwrite`` to refetch).
The Map Key is read from ``--firms_key`` or ``$FIRMS_MAP_KEY`` (``.env``).
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from nina_thermal.datagen import config
from nina_thermal.datagen.acquisition.firms_api import (
    fetch_area_csv,
    fetch_data_availability,
    firms_csv_name,
    plan_firms_chunks,
    source_conflicts,
    unavailable_chunks,
)
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)

COVERAGE_FILE = "coverage.json"
_SP_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.datagen.pipelines.run_fetch_firms",
        description="Download FIRMS label CSVs and coverage.json for one data root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--start", required=True, metavar="YYYYMMDD", help="First granule date."
    )
    p.add_argument(
        "--end", required=True, metavar="YYYYMMDD", help="Last granule date."
    )
    p.add_argument(
        "--out_dir", required=True, metavar="DIR", help="The root's firms/ dir."
    )
    p.add_argument(
        "--source",
        default=config.FIRMS_LABEL_SOURCE,
        choices=_SP_SOURCES,
        help="FIRMS product for all dates outside the Suomi-NPP outages.",
    )
    p.add_argument(
        "--outage_source",
        default=config.FIRMS_OUTAGE_SOURCE,
        choices=_SP_SOURCES,
        help="FIRMS product for dates inside config.FIRMS_SNPP_OUTAGES.",
    )
    p.add_argument(
        "--firms_key",
        default=None,
        metavar="KEY",
        help="NASA FIRMS Map Key. Falls back to $FIRMS_MAP_KEY.",
    )
    p.add_argument(
        "--allow_empty",
        action="store_true",
        help="Keep going when a request returns no detections (normally an error).",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Refetch CSVs already on disk."
    )
    p.add_argument(
        "--dry_run", action="store_true", help="Print the planned requests only."
    )
    p.add_argument(
        "--log_dir", default=None, metavar="DIR", help="Directory for log files."
    )
    return p


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def main(argv=None) -> int:
    """Run the FIRMS fetch.

    Returns:
        Exit code: 0 on success, 1 on any missing, empty or conflicting data.
    """
    args = build_parser().parse_args(argv)
    log_dir = args.log_dir or f"data/logs/run_{args.start}-{args.end}"
    _, start_time = configure_logging(
        "run_fetch_firms",
        log_dir,
        extra_id=f"{args.start}-{args.end}",
        argv=sys.argv[1:],
    )
    out_dir = Path(args.out_dir)
    bbox = config.FIRMS_LABEL_BBOX

    chunks = plan_firms_chunks(
        _parse_day(args.start),
        _parse_day(args.end),
        source=args.source,
        outage_source=args.outage_source,
    )
    for src, day0, n in chunks:
        logger.info("planned: %s", firms_csv_name(src, day0, n))
    if args.dry_run:
        for src, day0, n in chunks:
            print(firms_csv_name(src, day0, n))
        return log_run_footer(start_time, exit_code=0)

    # ── Guards on what is already in the root ────────────────────────────────
    coverage_path = out_dir / COVERAGE_FILE
    if coverage_path.is_file():
        declared = json.loads(coverage_path.read_text(encoding="utf-8")).get("bbox")
        if declared is None or [float(v) for v in declared] != list(bbox):
            logger.error(
                "%s declares bbox %s, not %s.  Use a fresh firms/ dir.",
                coverage_path,
                declared,
                list(bbox),
            )
            return log_run_footer(start_time, exit_code=1)
    existing = sorted(p.name for p in out_dir.glob("*.csv")) if out_dir.is_dir() else []
    conflicts = source_conflicts(existing, chunks)
    if conflicts:
        logger.error(
            "Dates already covered by another FIRMS product (would double-count FRP): %s",
            ", ".join(conflicts),
        )
        return log_run_footer(start_time, exit_code=1)

    load_dotenv()
    firms_key = args.firms_key or os.getenv("FIRMS_MAP_KEY")
    if not firms_key:
        logger.error("FIRMS Map Key missing: set $FIRMS_MAP_KEY or pass --firms_key.")
        return log_run_footer(start_time, exit_code=1)

    try:
        availability = fetch_data_availability(firms_key)
    except Exception as exc:  # the empty-response check below still guards
        logger.warning("FIRMS availability check failed (%s); continuing.", exc)
    else:
        missing = unavailable_chunks(chunks, availability)
        if missing:
            for src, day0, n in missing:
                logger.error(
                    "%s is not available for %s (+%d d); published range: %s",
                    src,
                    day0,
                    n,
                    availability.get(src),
                )
            logger.error("Refusing a partial fetch; SP products lag by months.")
            return log_run_footer(start_time, exit_code=1)

    # ── Fetch ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for src, day0, n in chunks:
        path = out_dir / firms_csv_name(src, day0, n)
        if path.is_file() and not args.overwrite:
            logger.info("exists, kept: %s", path.name)
            continue
        try:
            text = fetch_area_csv(firms_key, src, bbox, day0, n)
        except Exception as exc:
            logger.error("FIRMS request failed for %s: %s", path.name, exc)
            failed.append(path.name)
            continue
        n_rows = max(len(text.strip().splitlines()) - 1, 0)
        if n_rows == 0 and not args.allow_empty:
            logger.error(
                "%s: 0 detections over the label bbox — product missing for these "
                "dates, not a fire-free week.  Not written.",
                path.name,
            )
            failed.append(path.name)
            continue
        path.write_text(text, encoding="utf-8")
        logger.info("wrote %s (%d detections)", path.name, n_rows)

    if failed:
        logger.error("%d request(s) failed: %s", len(failed), ", ".join(failed))
        return log_run_footer(start_time, exit_code=1)

    _write_coverage(coverage_path, bbox, args)
    return log_run_footer(start_time, exit_code=0)


def _write_coverage(path: Path, bbox: tuple, args) -> None:
    """Write ``coverage.json``, appending this window to any earlier ones."""
    body = {}
    if path.is_file():
        body = json.loads(path.read_text(encoding="utf-8"))
    windows = body.get("windows", [])
    window = {"start": args.start, "end": args.end}
    if window not in windows:
        windows.append(window)
    body.update(
        {
            "bbox": list(bbox),
            "source": (
                f"{args.source} ({args.outage_source} for the S-NPP outages "
                f"{config.FIRMS_SNPP_OUTAGES})"
            ),
            "fetched_utc": datetime.now(timezone.utc).date().isoformat(),
            "windows": windows,
        }
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
