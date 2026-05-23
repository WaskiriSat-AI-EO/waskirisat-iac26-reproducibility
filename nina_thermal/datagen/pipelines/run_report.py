"""Pipeline: generate dataset summary report.

Usage::

    python -m nina_thermal.pipelines.run_report \\
        --dataset_dir data/04_dataset \\
        --output      data/04_dataset/dataset_summary.json

Calls :func:`build_dataset_report`, prints a human-readable summary table
to stdout, and saves ``dataset_summary.json``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from nina_thermal.datagen import config
from nina_thermal.datagen.dataset.report import build_dataset_report
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nina_thermal.pipelines.run_report",
        description=("Generate dataset_summary.json and print a summary table."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset_dir",
        required=True,
        metavar="DIR",
        help="Root dataset directory (output of run_balance or run_split).",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output path for dataset_summary.json.  Defaults to "
            "<dataset_dir>/dataset_summary.json."
        ),
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
    """Generate and print the dataset summary report.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = build_parser().parse_args(argv)
    log_dir = (
        args.log_dir
        or f"data/logs/run_{config.DEFAULT_START_DATE}-{config.DEFAULT_END_DATE}"
    )
    _, start_time = configure_logging("run_report", log_dir, argv=sys.argv[1:])
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        logger.error("dataset_dir does not exist: %s", dataset_dir)
        return log_run_footer(start_time, exit_code=1)

    # ── Build report ──────────────────────────────────────────────────────────
    logger.info("Building dataset report from %s …", dataset_dir)
    report = build_dataset_report(dataset_dir)

    # ── Optionally copy to requested output path ───────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Saved dataset_summary.json → %s", out_path)

    # ── Print human-readable summary table ────────────────────────────────────
    _print_summary(report)
    return log_run_footer(start_time, exit_code=0)


def _print_summary(report: dict) -> None:
    """Print a formatted summary table to stdout."""
    sep = "─" * 58
    sep2 = "═" * 58

    print(f"\n{sep2}")
    print("  NINA Thermal Dataset Summary")
    print(sep2)

    print(f"  {'Total granules':<30} {report.get('total_granules', '?'):>8}")
    print(
        f"  {'Patches (after NaN rejection)':<30} "
        f"{report.get('total_patches_after_rejection', '?'):>8}"
    )
    rr = report.get("rejection_rate", 0.0)
    print(f"  {'Rejection rate':<30} {rr:>8.1%}")

    # Splits
    print(f"\n{sep}")
    print(f"  {'Split':<10} {'Granules':>8} {'Patches':>8} {'Fire %':>8}")
    print(sep)
    for split_name in ("train", "val", "test"):
        s = report.get("splits", {}).get(split_name, {})
        ff = s.get("fire_fraction", 0.0)
        print(
            f"  {split_name:<10} {s.get('granules', 0):>8} "
            f"{s.get('patches', 0):>8} {ff:>8.1%}"
        )

    # Labels
    ld = report.get("label_distribution", {})
    print(f"\n{sep}")
    print("  Label distribution")
    print(sep)
    print(f"  {'Fire patches':<30} {ld.get('fire', 0):>8}")
    print(f"  {'No-fire patches':<30} {ld.get('no_fire', 0):>8}")
    ff = ld.get("fire_fraction", 0.0)
    print(f"  {'Fire fraction':<30} {ff:>8.3f}")

    # Surface types
    st = report.get("surface_types", {})
    print(f"\n{sep}")
    print("  Surface types")
    print(sep)
    for key in ("fire", "warm_surface", "nominal"):
        print(f"  {key:<30} {st.get(key, 0):>8}")

    # Profiles / AGC
    print(f"\n{sep}")
    prof_counts = report.get("profiles", {})
    agc_counts = report.get("agc_modes", {})
    print(
        f"  {'Profiles':<30} " + "  ".join(f"{k}={v}" for k, v in prof_counts.items())
    )
    print(
        f"  {'AGC modes':<30} " + "  ".join(f"{k}={v}" for k, v in agc_counts.items())
    )

    # Geographic coverage
    geo = report.get("geographic_coverage", {})
    if geo.get("lat_min") is not None:
        print(f"\n{sep}")
        print("  Geographic coverage")
        print(sep)
        print(
            f"  Lat {geo.get('lat_min', '?'):.2f} – {geo.get('lat_max', '?'):.2f}"
            f"   Lon {geo.get('lon_min', '?'):.2f} – {geo.get('lon_max', '?'):.2f}"
        )

    # Temporal coverage
    tc = report.get("temporal_coverage", {})
    if tc.get("date_min"):
        print(f"  Temporal {tc.get('date_min')} – {tc.get('date_max')}")

    # Processing parameters
    pp = report.get("processing_parameters", {})
    print(f"\n{sep}")
    print("  Processing parameters")
    print(sep)
    print(f"  {'AGG factor':<30} {pp.get('agg_factor', '?'):>8}")
    print(f"  {'Synthetic GSD [m]':<30} {pp.get('synthetic_gsd_m', '?'):>8.0f}")
    print(f"  {'NINA GSD [m]':<30} {pp.get('nina_lepton_gsd_m', '?'):>8.0f}")
    print(
        f"  {'GSD overestimation [%]':<30} {pp.get('gsd_overestimation_pct', '?'):>8.1f}"
    )
    print(f"  {'Fixed AGC range [K]':<30} {pp.get('fixed_agc_range_k', '?'):>8.1f}")
    print(f"  {'Fire ratio target':<30} {pp.get('fire_ratio_target', '?'):>8.3f}")
    print(f"  {'Script version':<30} {pp.get('script_version', '?'):>8}")
    print(sep2)


if __name__ == "__main__":
    sys.exit(main())
