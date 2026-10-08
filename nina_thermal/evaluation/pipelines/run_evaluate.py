"""CLI: evaluate the trained fire classifier on the granule-disjoint test split.

Reads the same data roots ``train_thermal`` merged, applies the same FRP
relabel, takes the held-out ``test`` partition, and writes
``evaluation_summary.json`` + ``predictions.json`` (and, with ``--figures``,
per-slice confusion / ROC / PR panels).

Usage::

    python -m nina_thermal.evaluation.pipelines.run_evaluate \\
        --checkpoint models/nina_thermal/mobilenetv3s_fire_best.pt \\
        --out_dir models/nina_thermal --figures
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from nina_thermal.evaluation.evaluate import build_evaluation_report, save_figures
from nina_thermal.training.frp_labels import DEFAULT_FRP_THRESHOLD_MW
from nina_thermal.training.train_thermal import CHECKPOINT_NAME, DEFAULT_ROOTS, set_seed
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the fire classifier on the held-out test split."
    )
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Data roots.")
    parser.add_argument(
        "--checkpoint",
        default=f"models/nina_thermal/{CHECKPOINT_NAME}",
        help="Trained fp32 checkpoint (bare state dict).",
    )
    parser.add_argument(
        "--out_dir",
        default="models/nina_thermal",
        help="Where evaluation_summary.json / predictions.json are written.",
    )
    parser.add_argument(
        "--training_report",
        default=None,
        help="training_report.json to read selected_threshold / beta from "
        "(default: search --out_dir then the checkpoint directory).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the operating threshold instead of reading the report.",
    )
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=DEFAULT_FRP_THRESHOLD_MW,
        help="Summed-FRP relabel threshold [MW]; must match training.",
    )
    parser.add_argument(
        "--frp_negative_max",
        type=float,
        default=0.0,
        help=(
            "Negative class = summed FRP below this many MW (low fire "
            "activity).  0 keeps the strict zero-detection definition, which "
            "leaves almost no negatives once FIRMS covers the whole 640x480 km "
            "footprint.  Must match training."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument("--figures", action="store_true", help="Render figures/.")
    parser.add_argument("--log_dir", default="data/logs", help="Log directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 success, 1 failure).
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "run_evaluate",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )
    set_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        logger.error("Checkpoint not found: %s. Run train_thermal first.", checkpoint)
        return log_run_footer(start_time, exit_code=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    try:
        report = build_evaluation_report(
            args.roots,
            checkpoint,
            Path(args.out_dir),
            device,
            frp_threshold=args.frp_threshold,
            frp_negative_max=args.frp_negative_max,
            explicit_threshold=args.threshold,
            training_report=(
                Path(args.training_report) if args.training_report else None
            ),
            batch_size=args.batch_size,
            seed=args.seed,
        )
    except Exception:
        logger.exception("Evaluation failed.")
        return log_run_footer(start_time, exit_code=1)

    if args.figures:
        import json

        predictions = json.loads(
            (Path(args.out_dir) / "predictions.json").read_text(encoding="utf-8")
        )["records"]
        save_figures(report, predictions, Path(args.out_dir))

    nom = report["nominal"]
    ci = nom["bootstrap_ci"]["roc_auc"]
    logger.info(
        "Test (profile %s): recall=%.3f  specificity=%.3f  ROC-AUC=%.3f "
        "[%.3f, %.3f]  degenerate=%s",
        "B",
        nom["recall"],
        nom["specificity"],
        nom["roc_auc"] or 0.0,
        ci["ci_low"] or 0.0,
        ci["ci_high"] or 0.0,
        report["degenerate_eval"],
    )
    return log_run_footer(start_time, exit_code=0)


if __name__ == "__main__":
    raise SystemExit(main())
