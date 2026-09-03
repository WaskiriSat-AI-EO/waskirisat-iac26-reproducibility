"""CLI: run the feasibility gate BEFORE spending time on CNN training.

Reads the same data roots ``train_thermal`` merges, takes the train+val
granules (never test), and reports the grouped-by-granule cross-validated AUC
of a BT-statistics + logistic-regression baseline — the number a CNN must beat
to justify its complexity.

Run it twice::

    # reproduce the negative result that motivated the FRP relabel
    python -m nina_thermal.training.pipelines.run_baseline --label datagen

    # show the reformulated target is learnable
    python -m nina_thermal.training.pipelines.run_baseline --label frp

Writes ``baseline_report_<label>.json`` to ``--out_dir``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from nina_thermal.training.baseline import DEFAULT_CV_FOLDS, build_baseline_report
from nina_thermal.training.frp_labels import DEFAULT_FRP_THRESHOLD_MW
from nina_thermal.training.train_thermal import DEFAULT_ROOTS, set_seed
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Feasibility gate: BT-statistics + logistic-regression baseline."
    )
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Data roots.")
    parser.add_argument(
        "--label",
        choices=["datagen", "frp"],
        default="frp",
        dest="label_scheme",
        help="'datagen' = manifest label; 'frp' = summed-FRP relabel.",
    )
    parser.add_argument("--agc_mode", default="fixed", help="AGC variant to read.")
    parser.add_argument("--profile", default="B", help="Degradation profile to read.")
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=DEFAULT_FRP_THRESHOLD_MW,
        help="Summed-FRP threshold [MW] when --label frp.",
    )
    parser.add_argument(
        "--cv_folds", type=int, default=DEFAULT_CV_FOLDS, help="Requested grouped-CV folds."
    )
    parser.add_argument(
        "--patches_manifest",
        default=None,
        help="Optional 03_patches/patches_manifest.json for MIN_FIRE_FRACTION "
        "recalibration (real data only; read-only).",
    )
    parser.add_argument(
        "--out_dir", default="models/nina_thermal", help="Output directory."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--log_dir", default="data/logs", help="Log directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 success, 1 failure).
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "run_baseline",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )
    set_seed(args.seed)

    try:
        report = build_baseline_report(
            args.roots,
            label_scheme=args.label_scheme,
            agc_mode=args.agc_mode,
            profile=args.profile,
            frp_threshold=args.frp_threshold,
            cv_folds=args.cv_folds,
            seed=args.seed,
            patches_manifest_path=(
                Path(args.patches_manifest) if args.patches_manifest else None
            ),
        )
    except Exception:
        logger.exception("Baseline feasibility gate failed.")
        return log_run_footer(start_time, exit_code=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"baseline_report_{args.label_scheme}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cv = report["grouped_cv"]
    logger.info(
        "Baseline (%s): n=%d  fire_fraction=%.4f  grouped-CV AUC=%s +/- %s "
        "(folds=%d)  degenerate=%s -> %s",
        args.label_scheme,
        report["n_records_used"],
        report["class_balance"]["fire_fraction"],
        f"{cv['auc_mean']:.3f}" if cv["auc_mean"] is not None else "N/A",
        f"{cv['auc_std']:.3f}" if cv["auc_std"] is not None else "N/A",
        cv["n_splits"],
        report["degenerate"],
        report_path,
    )
    for w in report["degenerate_warnings"]:
        logger.warning("  %s", w)
    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    raise SystemExit(main())
