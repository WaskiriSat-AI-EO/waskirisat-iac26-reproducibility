"""QAT quantization pipeline: fp32 → int8 TFLite for OpenMV N6 / Neural-ART.

Loads the checkpoint written by ``train_thermal.py``, runs a PTQ smoke test,
optionally fine-tunes with QAT, exports an int8 TFLite with a uint8 input
tensor, re-sweeps the operating threshold on the int8 model's validation
scores, and writes ``quantization_report.json`` with fp32-vs-int8 parity.

Stages
------
1. PTQ smoke test — abort in seconds if the graph does not trace or convert.
2. QAT fine-tune (skippable with ``--epochs 0`` for a pure-PTQ artifact).
3. TFLite export + SHA-256 checksum + size check against the mission budget.
4. Threshold re-sweep on int8 val scores.
5. Parity report + acceptance gate — fp32-vs-int8 metric deltas, score-space
   SQNR and KL, checked against the four deployment tolerances (recall drop
   <= 0.5 pp, F-beta drop <= 1.0 pp, SQNR >= 30 dB, KL <= 0.05 nats). A failed
   gate is a non-zero exit.

Note on transforms
------------------
Every loader here uses ``qat_transform``, which omits ``Normalize`` because
:class:`~nina_thermal.quantization.qat.NormalizedFireNet` folds normalization
into the graph.  Reusing ``train_thermal.build_transforms`` would normalize
twice and silently produce a garbage artifact.

Usage::

    python -m nina_thermal.quantization.pipelines.run_quantize \\
        --checkpoint models/nina_thermal/mobilenetv3s_fire_best.pt \\
        --epochs 5 --lr 1e-5 --out_dir models/nina_thermal
"""

import argparse
import json
import logging
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_thermal.quantization import parity as parity_lib
from nina_thermal.quantization import qat as qat_lib
from nina_thermal.training.dataset import (
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.train_thermal import (
    DEFAULT_ROOTS,
    TRAIN_AGC_MODES,
    TRAIN_PROFILES,
    collect_scores,
    set_seed,
)
from nina_thermal.training.frp_labels import (
    DEFAULT_FRP_THRESHOLD_MW,
    apply_frp_labels,
)
from nina_thermal.training.threshold import (
    average_precision,
    metrics_at_threshold,
    roc_auc,
    select_threshold,
)

logger = logging.getLogger(__name__)

# Mission budget for the on-orbit model artifact.
#
# 1.0 MB is NOT reachable with this backbone.  MobileNetV3-Small has 927,585
# parameters, so int8 weights alone are 0.88 MB, and the flatbuffer adds int32
# bias tensors, per-tensor quantization parameters and graph metadata — a
# measured floor of ~1.25 MB.  Getting under 1 MB requires a smaller backbone
# (e.g. MobileNetV2 with width_mult=0.35, ~0.38 MB of weights, but no
# torchvision ImageNet weights) or structured pruning.
SIZE_BUDGET_MB = 2.0


def qat_transform(augment: bool) -> transforms.Compose:
    """Transform for QAT and calibration — deliberately without Normalize.

    Args:
        augment: Apply the geometric augmentation used in training.

    Returns:
        Composed transform producing float tensors in [0, 1].
    """
    steps = []
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        ]
    steps += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # NOTE: no Normalize — NormalizedFireNet applies it inside the graph.
    ]
    return transforms.Compose(steps)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="QAT quantize the fire classifier to int8 TFLite"
    )
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Data roots.")
    parser.add_argument(
        "--profiles", nargs="+", default=TRAIN_PROFILES, help="Profiles."
    )
    parser.add_argument(
        "--agc_modes", nargs="+", default=TRAIN_AGC_MODES, help="AGC modes."
    )
    parser.add_argument(
        "--checkpoint",
        default="models/nina_thermal/mobilenetv3s_fire_best.pt",
        help="fp32 checkpoint from train_thermal.py.",
    )
    parser.add_argument(
        "--out_dir", default="models/nina_thermal", help="Artifact directory."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="QAT fine-tune epochs; 0 exports PTQ only.",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="QAT learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--pos_weight", type=float, default=3.0, help="FN:FP cost ratio."
    )
    parser.add_argument(
        "--recall_target", type=float, default=0.95, help="Recall floor."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers."
    )
    parser.add_argument("--log_dir", default="data/logs", help="Log directory.")
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=None,
        help=(
            "Relabel from FIRMS summed FRP, exactly as train_thermal does.  "
            "MUST match the value the checkpoint was trained with, or the QAT "
            "fine-tune and every parity number are computed against a "
            f"different target.  Suggested: {DEFAULT_FRP_THRESHOLD_MW:g}."
        ),
    )
    parser.add_argument(
        "--no_smoke_test", action="store_true", help="Skip the PTQ smoke test."
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        default=True,
        help="Per-output-channel int8 weights (default; more accurate).",
    )
    parser.add_argument(
        "--per_tensor",
        dest="per_channel",
        action="store_false",
        help="Per-tensor int8 weights — smaller artifact, slightly less accurate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "run_quantize",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )
    set_seed(args.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        logger.error("Checkpoint not found: %s.  Run train_thermal first.", checkpoint)
        return log_run_footer(start_time, exit_code=1)

    # ── Data ──────────────────────────────────────────────────────────────────
    records, _ = load_merged_records(args.roots)
    samples = expand_to_samples(records, args.profiles, args.agc_modes)
    frp_summary = None
    if args.frp_threshold is not None:
        samples, frp_summary = apply_frp_labels(samples, args.roots, args.frp_threshold)
    by_split = split_samples(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        ThermalDataset(by_split["train"], qat_transform(augment=True)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ThermalDataset(by_split["val"], qat_transform(augment=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info(
        "train=%d  val=%d images  device=%s",
        len(by_split["train"]),
        len(by_split["val"]),
        device,
    )

    model = qat_lib.build_model_for_qat(checkpoint, device)
    example = qat_lib.sample_input(device)
    beta = math.sqrt(args.pos_weight)

    # ── fp32 reference ────────────────────────────────────────────────────────
    y_true, fp32_scores = collect_scores(model, val_loader, device)
    fp32_threshold, degenerate = select_threshold(
        y_true, fp32_scores, args.recall_target
    )
    fp32_metrics = metrics_at_threshold(y_true, fp32_scores, fp32_threshold, beta)
    fp32_auc = roc_auc(y_true, fp32_scores)
    logger.info(
        "fp32 reference — ROC-AUC=%.4f  threshold=%.4f  recall=%.3f  precision=%.3f",
        fp32_auc,
        fp32_threshold,
        fp32_metrics["recall"],
        fp32_metrics["precision"],
    )

    # ── Stage 1: PTQ smoke test ───────────────────────────────────────────────
    if not args.no_smoke_test:
        logger.info("Stage 1: PTQ smoke test …")
        try:
            prepared, ptq_quantizer = qat_lib.prepare_for_ptq(
                model, example, per_channel=args.per_channel
            )
            with torch.no_grad():
                prepared(example)
            with tempfile.TemporaryDirectory() as tmp:
                smoke = qat_lib.export_tflite(
                    qat_lib.convert(prepared),
                    example,
                    Path(tmp) / "smoke.tflite",
                    quantizer=ptq_quantizer,
                )
                logger.info(
                    "PTQ smoke test PASSED (%.3f MB).",
                    smoke.stat().st_size / 1024 / 1024,
                )
        except Exception:
            logger.exception(
                "PTQ smoke test FAILED — the graph does not convert.  MobileNetV3 "
                "hardswish/squeeze-excite ops are the usual cause; see qat.py."
            )
            return log_run_footer(start_time, exit_code=1)

    # ── Stage 2: QAT fine-tune ────────────────────────────────────────────────
    logger.info("Stage 2: QAT preparation …")
    qat_model, quantizer = qat_lib.prepare_for_qat(
        model, example, per_channel=args.per_channel
    )
    qat_lib.calibrate(qat_model, train_loader, device)

    if args.epochs > 0:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight], device=device)
        )
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            qat_lib.set_mode(qat_model, training=True)
            total, seen = 0.0, 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.unsqueeze(1).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(qat_model(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.item() * images.size(0)
                seen += images.size(0)
            logger.info(
                "  QAT epoch %d/%d  loss=%.4f", epoch, args.epochs, total / seen
            )
    else:
        logger.info("Stage 2 skipped (--epochs 0) — exporting a PTQ artifact.")

    # ── Stage 3: convert and export ───────────────────────────────────────────
    logger.info("Stage 3: convert + export …")
    out_dir = Path(args.out_dir)
    tflite_path = out_dir / "mobilenetv3s_fire_int8.tflite"
    converted = qat_lib.convert(qat_model)
    qat_lib.export_tflite(converted, example, tflite_path, quantizer=quantizer)
    checksum_path = qat_lib.write_checksum(tflite_path)

    size_mb = tflite_path.stat().st_size / 1024 / 1024
    within_budget = size_mb <= SIZE_BUDGET_MB
    io_description = qat_lib.describe_tflite_io(tflite_path)
    if not io_description["fully_quantized"]:
        logger.warning(
            "Artifact I/O is %s → %s, not integer.  Weights are int8 but the "
            "graph boundary is float, so the Neural-ART NPU cannot own the "
            "whole graph — a dequantize runs on the Cortex-M55 first.",
            io_description["input"]["dtype"],
            io_description["output"]["dtype"],
        )
    logger.info(
        "Artifact %.3f MB — %s %.1f MB budget",
        size_mb,
        "within" if within_budget else "OVER",
        SIZE_BUDGET_MB,
    )

    # ── Stage 4: int8 scores + threshold re-sweep ─────────────────────────────
    logger.info("Stage 4: scoring the int8 model on val …")
    int8_scores = qat_lib.tflite_scores(tflite_path, val_loader)
    int8_true = y_true[: len(int8_scores)]
    int8_threshold, int8_degenerate = select_threshold(
        int8_true, int8_scores, args.recall_target
    )
    int8_metrics = metrics_at_threshold(int8_true, int8_scores, int8_threshold, beta)
    int8_auc = roc_auc(int8_true, int8_scores)
    logger.info(
        "int8 — ROC-AUC=%.4f  threshold=%.4f  recall=%.3f  precision=%.3f",
        int8_auc,
        int8_threshold,
        int8_metrics["recall"],
        int8_metrics["precision"],
    )

    # ── Stage 5: parity report + acceptance gate ─────────────────────────────
    auc_delta = (int8_auc - fp32_auc) if (int8_auc and fp32_auc) else None
    paired_fp32_scores = fp32_scores[: len(int8_scores)]
    sqnr_db = parity_lib.compute_sqnr(paired_fp32_scores, int8_scores)
    kl_nats = parity_lib.compute_kl_divergence(paired_fp32_scores, int8_scores)
    acceptance = parity_lib.check_acceptance(
        fp32_recall=fp32_metrics["recall"],
        int8_recall=int8_metrics["recall"],
        fp32_fbeta=fp32_metrics["fbeta"],
        int8_fbeta=int8_metrics["fbeta"],
        sqnr_db=sqnr_db,
        kl_nats=kl_nats,
    )
    logger.info(
        "SQNR=%s dB  KL=%.4f nats  acceptance=%s",
        "inf" if math.isinf(sqnr_db) else f"{sqnr_db:.1f}",
        kl_nats,
        "PASS" if acceptance["passed"] else "FAIL",
    )
    report = {
        "stage": "run_quantize",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "frp_relabel": frp_summary,
        "artifact": {
            "path": str(tflite_path),
            "checksum_file": str(checksum_path),
            "size_mb": round(size_mb, 4),
            "size_budget_mb": SIZE_BUDGET_MB,
            "within_budget": within_budget,
        },
        "quantization": {
            "weights": (
                "per-output-channel symmetric int8"
                if args.per_channel
                else "per-tensor symmetric int8"
            ),
            "activations": "per-tensor asymmetric uint8",
            "io": io_description,
            "normalization": "folded into graph via NormalizedFireNet",
            "method": "QAT" if args.epochs > 0 else "PTQ",
            "epochs": args.epochs,
            "lr": args.lr,
        },
        "fp32": {
            "roc_auc": fp32_auc,
            "average_precision": average_precision(y_true, fp32_scores),
            "threshold": fp32_threshold,
            "degenerate": degenerate,
            "metrics_at_threshold": fp32_metrics,
        },
        "int8": {
            "roc_auc": int8_auc,
            "average_precision": average_precision(int8_true, int8_scores),
            "threshold": int8_threshold,
            "degenerate": int8_degenerate,
            "metrics_at_threshold": int8_metrics,
        },
        "parity": {
            "roc_auc_delta": round(auc_delta, 6) if auc_delta is not None else None,
            "recall_delta": round(int8_metrics["recall"] - fp32_metrics["recall"], 6),
            "precision_delta": round(
                int8_metrics["precision"] - fp32_metrics["precision"], 6
            ),
            "fbeta_delta": round(int8_metrics["fbeta"] - fp32_metrics["fbeta"], 6),
            "sqnr_db": None if math.isinf(sqnr_db) else round(sqnr_db, 2),
            "kl_divergence_nats": round(kl_nats, 6),
        },
        "acceptance": acceptance,
        "deployment_note": (
            "Load on OpenMV N6 with ml.Model('/rom/mobilenetv3s_fire_int8.tflite'); "
            "feed the Lepton frame replicated to 3 channels, apply sigmoid to the "
            "logit and compare against int8.threshold.  Check artifact.io: this "
            "PT2E export quantizes weights to int8 but leaves the graph boundary "
            "in float32, so it is not yet a full-NPU graph.  Achieving uint8 I/O "
            "needs the ai_edge_quantizer static_wi8_ai8 recipe applied to a float "
            "export, which is not yet wired into this pipeline."
        ),
    }
    report_path = out_dir / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Quantization report → %s", report_path)

    ok = within_budget and acceptance["passed"]
    return log_run_footer(start_time, exit_code=0 if ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
