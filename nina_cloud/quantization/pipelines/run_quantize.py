"""QAT quantization pipeline: fp32 → int8 TFLite for OpenMV N6 / Neural-ART.

Reads ``balanced_manifest.json`` from the dataset directory, runs a PTQ smoke
test, performs QAT fine-tuning, exports an int8 TFLite with uint8 input (ready
for ``ml.Model()`` in OpenMV MicroPython), re-sweeps the operating threshold on
the int8 model's val scores, then writes a ``quantization_report.json`` with
fp32-vs-int8 parity metrics for the IAC paper.

Prerequisite
------------
The dataset must have a class-balanced val split (profile B) and a class-
balanced test split (A/B/C) with at least ``MIN_CLEAR_FOR_EVAL`` clear samples.
The pipeline checks this and exits early if the data is degenerate (current
state: val is all-cloud, test is empty — regenerate the dataset first).

Stages
------
1. PTQ smoke test — abort early if the graph doesn't trace.
2. QAT fine-tune — observer warmup + fine-tuning epochs.
3. TFLite export + checksum.
4. Threshold re-sweep on int8 val scores.
5. Parity report (SQNR, KL, per-metric deltas, acceptance verdict).
6. Write ``models/mobilenetv2_cloud_int8.tflite`` + ``quantization_report.json``.

Usage::

    python -m nina_cloud.quantization.pipelines.run_quantize \\
        --dataset_dir data/cloud/04_dataset \\
        --epochs 5 --lr 1e-5 --recall_target 0.99 \\
        --out models/

    # Skip PTQ smoke test (e.g. already verified):
    python -m nina_cloud.quantization.pipelines.run_quantize \\
        --dataset_dir data/cloud/04_dataset --no_smoke_test
"""

import argparse
import json
import logging
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_cloud.datagen import config as datagen_config
from nina_cloud.evaluation.evaluate import build_model as build_fp32_model
from nina_cloud.quantization.export_tflite import (
    export_tflite,
    run_ptq_smoke_test,
    write_checksum,
)
from nina_cloud.quantization.parity import (
    build_parity_report,
    collect_tflite_scores,
    select_threshold_from_scores,
)
from nina_cloud.quantization.qat import (
    IMAGE_SIZE,
    build_model_for_qat,
    calibrate_observers,
    collect_val_scores_fp32,
    convert_qat_model,
    prepare_for_qat,
    qat_eval_transform,
    qat_fine_tune,
)

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the QAT quantization pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "nina_cloud quantization: QAT → int8 TFLite for OpenMV N6 Neural-ART NPU"
        )
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="04_dataset/ directory (must contain balanced_manifest.json)",
    )
    parser.add_argument(
        "--checkpoint",
        default="mobilenetv2_cloud_best.pt",
        help=(
            "Checkpoint filename or absolute path.  "
            "If a bare filename, resolved relative to --dataset_dir.  "
            "(default: mobilenetv2_cloud_best.pt)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="QAT fine-tune epochs (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help=(
            "QAT learning rate — much smaller than original training LR (1e-3).  "
            "(default: 1e-5)"
        ),
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=3.0,
        help=(
            "FN:FP cost ratio for BCEWithLogitsLoss.  Must match the value "
            "used in original training.  (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--recall_target",
        type=float,
        default=0.99,
        help="Minimum cloud recall for the int8 threshold re-sweep.  (default: 0.99)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for fine-tuning and calibration.  (default: 16)",
    )
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=10,
        help="Calibration batches for observer warmup before QAT.  (default: 10)",
    )
    parser.add_argument(
        "--recall_tol",
        type=float,
        default=0.005,
        help=(
            "Acceptance gate: maximum cloud recall *drop* (fp32 − int8).  "
            "(default: 0.005 = 0.5 pp)"
        ),
    )
    parser.add_argument(
        "--fbeta_tol",
        type=float,
        default=0.010,
        help="Acceptance gate: maximum F-beta *drop*.  (default: 0.010 = 1.0 pp)",
    )
    parser.add_argument(
        "--out",
        default="models/",
        help="Output directory for TFLite artifact and report.  (default: models/)",
    )
    parser.add_argument(
        "--no_smoke_test",
        action="store_true",
        help="Skip the PTQ smoke test (use only if traceability has been verified).",
    )
    parser.add_argument(
        "--parity_split",
        default="val",
        choices=["val", "test"],
        help=(
            "Manifest split to use for parity evaluation.  "
            "Use 'test' when a balanced test set is available.  (default: val)"
        ),
    )
    parser.add_argument("--log_dir", default="data/logs")
    parser.add_argument("--seed", type=int, default=datagen_config.DEFAULT_SEED)
    args = parser.parse_args(argv)

    log_file, start_time = configure_logging(
        "run_quantize", log_dir=args.log_dir, argv=sys.argv[1:]
    )
    logger.info("Log file: %s", log_file)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_dir / "balanced_manifest.json"
    if not manifest_path.exists():
        logger.error("balanced_manifest.json not found: %s", manifest_path)
        return log_run_footer(start_time, 1)

    with open(manifest_path) as f:
        body = json.load(f)
    records = body.get("records", body) if isinstance(body, dict) else body
    logger.info("Loaded manifest: %d total records", len(records))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = dataset_dir / checkpoint_path
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return log_run_footer(start_time, 1)

    image_dir = dataset_dir / "images"

    # ── Split records ─────────────────────────────────────────────────────────
    train_records = [r for r in records if r.get("split") == "train"]
    val_records = [r for r in records if r.get("split") == "val"]
    parity_records = [r for r in records if r.get("split") == args.parity_split]

    logger.info(
        "Split sizes — train: %d  val: %d  parity (%s): %d",
        len(train_records),
        len(val_records),
        args.parity_split,
        len(parity_records),
    )

    # ── Degenerate-data guard ─────────────────────────────────────────────────
    n_val_clear = sum(1 for r in val_records if r.get("label") == 0)
    if n_val_clear < datagen_config.MIN_CLEAR_FOR_EVAL:
        logger.error(
            "Val split has only %d clear samples (minimum: %d).  "
            "The threshold re-sweep and parity comparison require a balanced val set.  "
            "Regenerate the dataset before running quantization.",
            n_val_clear,
            datagen_config.MIN_CLEAR_FOR_EVAL,
        )
        return log_run_footer(start_time, 1)

    if not train_records:
        logger.error("No training records found in manifest.")
        return log_run_footer(start_time, 1)
    if not parity_records:
        logger.error(
            "No records with split=='%s' found — cannot run parity evaluation.",
            args.parity_split,
        )
        return log_run_footer(start_time, 1)

    # ── Device + hyperparameters ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    beta = math.sqrt(args.pos_weight)
    logger.info(
        "pos_weight=%.1f  beta=%.3f  recall_target=%.2f",
        args.pos_weight,
        beta,
        args.recall_target,
    )

    # Resolve float32 threshold from training_report.json
    report_path = dataset_dir / "training_report.json"
    fp32_threshold = 0.5
    if report_path.exists():
        with open(report_path) as f:
            tr = json.load(f)
        fp32_threshold = float(tr.get("selected_threshold", 0.5))
        reported_beta = tr.get("hyperparameters", {}).get("beta")
        if reported_beta is not None:
            beta = float(reported_beta)
        logger.info(
            "Float32 threshold from training_report.json: %.4f  beta=%.3f",
            fp32_threshold,
            beta,
        )
    else:
        logger.warning(
            "training_report.json not found — using fp32_threshold=0.5 as baseline."
        )

    # ── Representative sample input for PT2E tracing ──────────────────────────
    sample_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # ── Stage 1: PTQ smoke test ───────────────────────────────────────────────
    if not args.no_smoke_test:
        logger.info("=== Stage 1: PTQ smoke test ===")
        try:
            model_fp32_wrapped = build_model_for_qat(checkpoint_path, device)
            model_fp32_wrapped.eval()
            with tempfile.TemporaryDirectory() as tmp:
                run_ptq_smoke_test(model_fp32_wrapped, sample_input, Path(tmp))
        except Exception as exc:
            logger.error(
                "PTQ smoke test FAILED — aborting before QAT.  "
                "Check ai-edge-torch installation and op coverage.  Error: %s",
                exc,
            )
            return log_run_footer(start_time, 1)
    else:
        logger.info("PTQ smoke test skipped (--no_smoke_test).")

    # ── Stage 2: QAT fine-tune ────────────────────────────────────────────────
    logger.info(
        "=== Stage 2: QAT fine-tune (%d epochs, lr=%.2e) ===", args.epochs, args.lr
    )
    model_for_qat = build_model_for_qat(checkpoint_path, device)

    model_qat_prepared = prepare_for_qat(model_for_qat, sample_input)

    from torch.utils.data import DataLoader
    from nina_cloud.training.train_cloud import CloudDataset

    calib_loader = DataLoader(
        CloudDataset(train_records, image_dir, qat_eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    calibrate_observers(model_qat_prepared, calib_loader, device, args.calib_batches)

    qat_fine_tune(
        model_qat_prepared,
        train_records=train_records,
        val_records=val_records,
        image_dir=image_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight,
        seed=args.seed,
    )

    # ── Stage 3: TFLite export ────────────────────────────────────────────────
    logger.info("=== Stage 3: TFLite export ===")
    quantized_model = convert_qat_model(model_qat_prepared)
    tflite_path = out_dir / "mobilenetv2_cloud_int8.tflite"

    try:
        export_tflite(quantized_model, sample_input, tflite_path)
        write_checksum(tflite_path)
    except Exception as exc:
        logger.error("TFLite export failed: %s", exc)
        return log_run_footer(start_time, 1)

    # ── Stage 4: Threshold re-sweep on int8 val scores ────────────────────────
    logger.info("=== Stage 4: Int8 threshold re-sweep ===")
    try:
        int8_val_y_true, int8_val_y_score = collect_tflite_scores(
            tflite_path, val_records, image_dir, IMAGE_SIZE
        )
    except Exception as exc:
        logger.error("Int8 val inference failed: %s", exc)
        return log_run_footer(start_time, 1)

    int8_threshold, threshold_degenerate = select_threshold_from_scores(
        int8_val_y_true, int8_val_y_score, args.recall_target
    )
    logger.info(
        "Int8 threshold: %.4f  (fp32 was: %.4f)  degenerate: %s",
        int8_threshold,
        fp32_threshold,
        threshold_degenerate,
    )

    # ── Stage 5: Parity report ────────────────────────────────────────────────
    logger.info("=== Stage 5: Parity evaluation on '%s' split ===", args.parity_split)
    try:
        # fp32 scores on the same parity split
        fp32_model = build_fp32_model(checkpoint_path, device)
        fp32_y_true, fp32_y_score = collect_val_scores_fp32(
            fp32_model, parity_records, image_dir, device, args.batch_size
        )
        # int8 scores on the same parity split
        int8_y_true, int8_y_score = collect_tflite_scores(
            tflite_path, parity_records, image_dir, IMAGE_SIZE
        )
    except Exception as exc:
        logger.error("Parity score collection failed: %s", exc)
        return log_run_footer(start_time, 1)

    parity = build_parity_report(
        fp32_y_true=fp32_y_true,
        fp32_y_score=fp32_y_score,
        int8_y_true=int8_y_true,
        int8_y_score=int8_y_score,
        int8_threshold=int8_threshold,
        fp32_threshold=fp32_threshold,
        beta=beta,
        recall_tol=args.recall_tol,
        fbeta_tol=args.fbeta_tol,
    )

    nom = parity["nominal_deltas"]
    logger.info(
        "SQNR=%.1f dB  KL=%.4f nats  "
        "Δrecall=%+.4f  Δf_beta=%+.4f  Δroc_auc=%+.4f  accepted=%s",
        parity["sqnr_db"] or 0.0,
        parity["kl_divergence"],
        nom.get("recall", {}).get("delta", float("nan")),
        nom.get("f_beta", {}).get("delta", float("nan")),
        nom.get("roc_auc", {}).get("delta", float("nan")),
        parity["acceptance"]["passed"],
    )

    # ── Stage 6: Write quantization_report.json ───────────────────────────────
    logger.info("=== Stage 6: Writing quantization_report.json ===")
    report = {
        "stage": "run_quantize",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path.name,
        "tflite": tflite_path.name,
        "quantization_strategy": "QAT",
        "framework": "PyTorch PT2E + ai-edge-torch",
        "target": "STM32N6 Neural-ART NPU (OpenMV N6, MicroPython ml.Model)",
        "qat_hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "beta": round(beta, 4),
            "recall_target": args.recall_target,
            "batch_size": args.batch_size,
            "calib_batches": args.calib_batches,
            "seed": args.seed,
        },
        "thresholds": {
            "fp32_selected_threshold": round(fp32_threshold, 6),
            "int8_selected_threshold": round(int8_threshold, 6),
            "threshold_degenerate": threshold_degenerate,
            "note": (
                "selected_threshold_int8 is derived by re-sweeping the val-set "
                "recall-target on the int8 model's scores.  Use this value in "
                "the OpenMV MicroPython inference script, not the fp32 threshold."
            ),
        },
        "parity_split": args.parity_split,
        "parity": parity,
        "acceptance_gate": parity["acceptance"],
    }

    report_path_out = out_dir / "quantization_report.json"
    with open(report_path_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Quantization report written: %s", report_path_out)

    if not parity["acceptance"]["passed"]:
        logger.warning(
            "ACCEPTANCE GATE FAILED — the int8 model exceeds metric tolerances.  "
            "Consider increasing --epochs or --calib_batches.  "
            "See quantization_report.json for details."
        )

    return log_run_footer(start_time, 0 if parity["acceptance"]["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main())
