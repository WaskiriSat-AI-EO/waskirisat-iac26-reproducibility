"""CLI: recompute the int8 acceptance gate for an existing .tflite.

Runs stages "fp32 reference", 4 and 5 of ``run_quantize`` without QAT or
export: the same validation split, transforms, threshold sweep and
``parity.check_acceptance``.  Use it to gate an artifact that was exported
before the gate existed (the HIL thermal artifact, #15) or to re-check one.

Usage::

    python -m nina_thermal.quantization.pipelines.run_gate \\
        --checkpoint models/nina_thermal/mobilenetv3s_fire_best.pt \\
        --tflite models/nina_thermal/mobilenetv3s_fire_int8.tflite \\
        --out gate.json
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from nina_thermal.quantization import parity as parity_lib
from nina_thermal.quantization import qat as qat_lib
from nina_thermal.quantization.pipelines.run_quantize import qat_transform
from nina_thermal.training.dataset import (
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.frp_labels import DEFAULT_FRP_THRESHOLD_MW, apply_frp_labels
from nina_thermal.training.threshold import metrics_at_threshold, roc_auc, select_threshold
from nina_thermal.training.train_thermal import (
    CHECKPOINT_NAME,
    DEFAULT_ROOTS,
    TRAIN_AGC_MODES,
    TRAIN_PROFILES,
    collect_scores,
    set_seed,
)
from shared.pipeline_logging import configure_logging, log_run_footer

logger = logging.getLogger(__name__)


def compute_gate(
    roots: list[str | Path],
    checkpoint: Path,
    tflite: Path,
    device: torch.device,
    *,
    frp_threshold: float = DEFAULT_FRP_THRESHOLD_MW,
    pos_weight: float = 3.0,
    recall_target: float = 0.95,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Score fp32 and int8 on validation and apply the acceptance gate.

    Args:
        roots: Data roots (same set the model was trained on).
        checkpoint: fp32 checkpoint the artifact was quantized from.
        tflite: int8 artifact to gate.
        device: Device for the fp32 reference.
        frp_threshold: Summed-FRP relabel threshold [MW]; must match training.
        pos_weight: FN:FP cost ratio; F-beta uses ``beta = sqrt(pos_weight)``.
        recall_target: Recall floor for both threshold sweeps.
        batch_size: DataLoader batch size.

    Returns:
        Dict with the val composition, fp32 / int8 metrics, the ``acceptance``
        block (same schema as ``quantization_report.json``) and per-patch
        val scores.
    """
    records, _ = load_merged_records(roots)
    samples = expand_to_samples(records, TRAIN_PROFILES, TRAIN_AGC_MODES)
    samples, _ = apply_frp_labels(samples, roots, frp_threshold)
    val = split_samples(samples)["val"]
    val_loader = DataLoader(
        ThermalDataset(val, qat_transform(augment=False)),
        batch_size=batch_size,
        shuffle=False,
    )
    beta = math.sqrt(pos_weight)

    model = qat_lib.build_model_for_qat(checkpoint, device)
    y_true, fp32_scores = collect_scores(model, val_loader, device)
    fp32_threshold, _ = select_threshold(y_true, fp32_scores, recall_target)
    fp32_metrics = metrics_at_threshold(y_true, fp32_scores, fp32_threshold, beta)

    int8_scores = qat_lib.tflite_scores(tflite, val_loader)
    int8_true = y_true[: len(int8_scores)]
    int8_threshold, _ = select_threshold(int8_true, int8_scores, recall_target)
    int8_metrics = metrics_at_threshold(int8_true, int8_scores, int8_threshold, beta)

    paired_fp32_scores = fp32_scores[: len(int8_scores)]
    acceptance = parity_lib.check_acceptance(
        fp32_recall=fp32_metrics["recall"],
        int8_recall=int8_metrics["recall"],
        fp32_fbeta=fp32_metrics["fbeta"],
        int8_fbeta=int8_metrics["fbeta"],
        sqnr_db=parity_lib.compute_sqnr(paired_fp32_scores, int8_scores),
        kl_nats=parity_lib.compute_kl_divergence(paired_fp32_scores, int8_scores),
    )
    return {
        "artifact": Path(tflite).name,
        "val_composition": {
            "patches": len({s["patch"] for s in val}),
            "granules": len({s["granule_id"] for s in val}),
            "fire": sum(1 for s in val if s["label"] == 1),
        },
        "fp32": {
            "threshold": fp32_threshold,
            "roc_auc": roc_auc(y_true, fp32_scores),
            **fp32_metrics,
        },
        "int8": {
            "threshold": int8_threshold,
            "roc_auc": roc_auc(int8_true, int8_scores),
            **int8_metrics,
        },
        "acceptance": acceptance,
        "val_scores": [
            {
                "patch": s["patch"],
                "label": y,
                "fp32_score": round(float(f), 6),
                "int8_score": round(float(i), 6),
            }
            for s, y, f, i in zip(val, y_true, paired_fp32_scores, int8_scores, strict=True)
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Recompute the int8 acceptance gate for an existing .tflite."
    )
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS, help="Data roots.")
    parser.add_argument(
        "--checkpoint",
        default=f"models/nina_thermal/{CHECKPOINT_NAME}",
        help="fp32 checkpoint the artifact was quantized from.",
    )
    parser.add_argument("--tflite", required=True, help="int8 .tflite to gate.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--frp_threshold",
        type=float,
        default=DEFAULT_FRP_THRESHOLD_MW,
        help="Summed-FRP relabel threshold [MW]; must match training.",
    )
    parser.add_argument(
        "--pos_weight", type=float, default=3.0, help="FN:FP cost ratio."
    )
    parser.add_argument(
        "--recall_target", type=float, default=0.95, help="Recall floor."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--log_dir", default="data/logs", help="Log directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 gate passed, 1 gate failed or error).
    """
    args = parse_args(argv)
    _, start_time = configure_logging(
        "run_gate",
        log_dir=args.log_dir,
        argv=argv if argv is not None else sys.argv[1:],
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        result = compute_gate(
            args.roots,
            Path(args.checkpoint),
            Path(args.tflite),
            device,
            frp_threshold=args.frp_threshold,
            pos_weight=args.pos_weight,
            recall_target=args.recall_target,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception("Gate computation failed.")
        return log_run_footer(start_time, exit_code=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    measured = result["acceptance"]["measured"]
    logger.info(
        "Gate %s -> %s  recall_drop=%.4f  fbeta_drop=%.4f  SQNR=%s dB  KL=%.4f nats",
        "PASSED" if result["acceptance"]["passed"] else "FAILED",
        out,
        measured["recall_drop"],
        measured["fbeta_drop"],
        measured["sqnr_db"],
        measured["kl_nats"],
    )
    return log_run_footer(start_time, exit_code=0 if result["acceptance"]["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main())
