"""Evaluation pipeline: load checkpoint, evaluate on test split, write report.

Reads ``balanced_manifest.json`` from the dataset directory, filters to the
test split, runs inference with the trained MobileNetV2 checkpoint, and writes
``evaluation_summary.json`` with overall and per-degradation-profile metrics.

Usage::

    python -m nina_cloud.evaluation.pipelines.run_evaluate \\
        --dataset_dir data/cloud/04_dataset

    # With optional PNG figures (confusion matrix + ROC curves):
    python -m nina_cloud.evaluation.pipelines.run_evaluate \\
        --dataset_dir data/cloud/04_dataset --figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_cloud.datagen import config
from nina_cloud.evaluation.evaluate import build_evaluation_report, save_figures

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the evaluation pipeline stage.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="nina_cloud evaluation: evaluate trained model on test split"
    )
    parser.add_argument("--dataset_dir", required=True, help="04_dataset/ directory")
    parser.add_argument(
        "--checkpoint",
        default="mobilenetv2_cloud_best.pt",
        help=(
            "Checkpoint filename or absolute path. "
            "If a bare filename, resolved relative to --dataset_dir. "
            "(default: mobilenetv2_cloud_best.pt)"
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Inference batch size (default: 16)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train"],
        help=(
            "Manifest split to evaluate (default: test). "
            "Use 'val' when the dataset has too few scenes to produce a test partition."
        ),
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Save confusion-matrix and ROC-curve PNG figures",
    )
    parser.add_argument("--log_dir", default="data/logs")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_evaluate", log_dir=args.log_dir, argv=sys.argv[1:]
    )
    logger.info("Log file: %s", log_file)

    dataset_dir = Path(args.dataset_dir)
    balanced_path = dataset_dir / "balanced_manifest.json"

    if not balanced_path.exists():
        logger.error("balanced_manifest.json not found: %s", balanced_path)
        return log_run_footer(start_time, 1)

    with open(balanced_path) as f:
        body = json.load(f)
    records = body.get("records", body) if isinstance(body, dict) else body
    logger.info("Loaded manifest: %d total records", len(records))

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = dataset_dir / checkpoint_path
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        logger.error(
            "Train the model first: python nina_cloud/training/train_cloud.py "
            "--dataset_dir %s",
            dataset_dir,
        )
        return log_run_footer(start_time, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_path = dataset_dir / "evaluation_summary.json"

    try:
        report = build_evaluation_report(
            records=records,
            image_dir=dataset_dir / "images",
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            device=device,
            batch_size=args.batch_size,
            split=args.split,
        )
    except ValueError as exc:
        logger.error("Evaluation failed: %s", exc)
        return log_run_footer(start_time, 1)

    overall = report["overall"]
    logger.info(
        "Overall — n=%d  accuracy=%.3f  f1=%.3f  roc_auc=%s",
        overall["n"],
        overall["accuracy"],
        overall["f1"],
        f"{overall['roc_auc']:.3f}" if overall["roc_auc"] is not None else "N/A",
    )
    for profile, m in sorted(report["by_profile"].items()):
        logger.info(
            "  profile %s — n=%d  accuracy=%.3f  f1=%.3f  roc_auc=%s",
            profile,
            m["n"],
            m["accuracy"],
            m["f1"],
            f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A",
        )

    if args.figures:
        pred_path = dataset_dir / "predictions.json"
        with open(pred_path) as f:
            preds_body = json.load(f)
        predictions = (
            preds_body["records"] if isinstance(preds_body, dict) else preds_body
        )
        save_figures(report, predictions, dataset_dir)
        logger.info("Figures saved to %s/figures/", dataset_dir)

    return log_run_footer(start_time, 0)


if __name__ == "__main__":
    sys.exit(main())
