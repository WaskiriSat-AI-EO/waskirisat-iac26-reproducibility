"""Diagnose int8 conversion: stage-by-stage scores and first diverging node.

Usage::

    python -m nina_thermal.quantization.pipelines.run_quant_diagnose \\
        --checkpoint models/nina_thermal_frp20000_v3_bc/mobilenetv3s_fire_best.pt \\
        --frp_threshold 20000 --frp_negative_max 1000 --profiles B C \\
        --mode qat --variant baseline no_inplace no_se

Each variant is a small, reversible change to what gets exported (see
``docs/PLAN_int8_fix.md`` Phase 1).  A faithful conversion shows
``corr_vs_prepared`` > 0.99 for the converted and TFLite stages.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.pipeline_logging import configure_logging, log_run_footer

from nina_thermal.quantization import diagnostics as diag
from nina_thermal.quantization import qat as qat_lib
from nina_thermal.quantization.pipelines.run_quantize import qat_transform
from nina_thermal.training.dataset import (
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.frp_labels import apply_frp_labels
from nina_thermal.training.train_thermal import DEFAULT_ROOTS, set_seed

logger = logging.getLogger(__name__)


def _no_inplace(model: nn.Module) -> nn.Module:
    """Disable in-place activations and in-place residual adds."""
    from torchvision.models.mobilenetv3 import InvertedResidual

    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
        if isinstance(module, InvertedResidual):

            def forward(x, _m=module):
                out = _m.block(x)
                return out + x if _m.use_res_connect else out

            module.forward = forward
    return model


def _quantizer_without_se(qat: bool):
    """Quantizer that leaves squeeze-excite blocks in float."""
    from torchvision.ops.misc import SqueezeExcitation

    quantizer = qat_lib.build_quantizer(qat=qat)
    quantizer.set_module_type(SqueezeExcitation, None)
    return quantizer


VARIANTS = {
    "baseline": {},
    "no_inplace": {"model_hook": _no_inplace},
    "no_se": {"quantizer": _quantizer_without_se},
    "no_inplace_no_se": {"model_hook": _no_inplace, "quantizer": _quantizer_without_se},
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--profiles", nargs="+", default=["B"])
    parser.add_argument("--frp_threshold", type=float, default=20000.0)
    parser.add_argument("--frp_negative_max", type=float, default=0.0)
    parser.add_argument("--mode", choices=["ptq", "qat", "both"], default="both")
    parser.add_argument("--variant", nargs="+", choices=sorted(VARIANTS), default=["baseline"])
    parser.add_argument("--calibration_batches", type=int, default=256)
    parser.add_argument("--no_tflite", action="store_true", help="Skip TFLite export.")
    parser.add_argument("--trace", action="store_true", help="Print the per-node trace.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--log_dir", default="data/logs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _, start = configure_logging("run_quant_diagnose", log_dir=args.log_dir,
                                 argv=argv if argv is not None else sys.argv[1:])
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records, _ = load_merged_records(args.roots)
    samples = expand_to_samples(records, args.profiles, ["fixed"])
    samples, _ = apply_frp_labels(samples, args.roots, args.frp_threshold, args.frp_negative_max)
    by_split = split_samples(samples)
    val_loader = DataLoader(ThermalDataset(by_split["val"], qat_transform(augment=False)),
                            batch_size=1, shuffle=False)
    cal_loader = DataLoader(ThermalDataset(by_split["train"], qat_transform(augment=False)),
                            batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    labels = [int(s["label"]) for s in by_split["val"]]
    model = qat_lib.build_model_for_qat(Path(args.checkpoint), device)

    modes = ["ptq", "qat"] if args.mode == "both" else [args.mode]
    results = {}
    for variant in args.variant:
        spec = VARIANTS[variant]
        for mode in modes:
            qat = mode == "qat"
            quantizer = spec["quantizer"](qat) if "quantizer" in spec else None
            try:
                report = diag.stage_report(
                    model, val_loader, cal_loader, labels, device, qat=qat,
                    calibration_batches=args.calibration_batches, quantizer=quantizer,
                    model_hook=spec.get("model_hook"), tflite=not args.no_tflite,
                )
            except Exception as exc:  # a variant failing to export is itself a finding
                logger.exception("%s/%s failed", variant, mode)
                results[f"{variant}/{mode}"] = {"error": repr(exc)}
                continue
            results[f"{variant}/{mode}"] = report
            logger.info("── %s / %s  (BN nodes after convert: %d)", variant, mode,
                        report["bn_nodes_after_convert"])
            for row in report["stages"]:
                logger.info("   %-26s AUC %-6s corr %-7s SQNR %-7s corr_vs_prepared %s",
                            row["stage"], row.get("roc_auc"), row.get("corr", "-"),
                            row.get("sqnr_db", "-"), row.get("corr_vs_prepared", "-"))

    if args.trace:
        image = next(iter(val_loader))[0].to(device)
        for mode in modes:
            prepare = qat_lib.prepare_for_qat if mode == "qat" else qat_lib.prepare_for_ptq
            prepared, _ = prepare(model, qat_lib.sample_input(device),
                                  quantizer=qat_lib.build_quantizer(qat=mode == "qat"))
            diag.calibrate(prepared, cal_loader, device, args.calibration_batches)
            rows = diag.compare_graphs(prepared, qat_lib.convert(prepared), image)
            jump = diag.first_jump(rows)
            logger.info("── trace %s: first jump %s", mode, jump)
            for row in rows[:60]:
                logger.info("   %-28s %-40s %.4f", row["node"], row["target"][:40], row["rel_err"])

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return log_run_footer(start, exit_code=0)


if __name__ == "__main__":
    raise SystemExit(main())
