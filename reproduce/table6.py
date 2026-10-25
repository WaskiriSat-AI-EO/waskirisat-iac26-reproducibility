"""Table 6 — test performance (T2) (owners: Leyla for cloud, Osvaldo for thermal).

Thermal: implemented (#15), from `models/thermal/frp20000/evaluation_summary.json`. Cloud: from the INT8 evaluation report of the HIL file (`f528086c…`).

Contract: `compute()` returns a flat dict whose keys are
exactly those in `reproduce/expected/table6.json`, computed from the repo's own
artifacts (dataset manifests under data/manifests/, models listed in
models/MANIFEST.json, the payload's evaluation output). Split the work by
payload: implement `compute_cloud()` / `compute_thermal()` and merge them.
`python -m reproduce.check` reports this item as PENDING until it is done.
"""

from __future__ import annotations

import sys

from reproduce.lib import ROOT, load_json, report

CLOUD_INT8_REPORT = (
    ROOT
    / "nina_cloud_results"
    / "evaluation"
    / "int8"
    / "reports"
    / "reports_alpha075_fullinteger"
    / "fullinteger_int8_report.json"
)
THERMAL_EVAL = ROOT / "models" / "thermal" / "frp20000" / "evaluation_summary.json"


def compute_cloud() -> dict:
    """Test profiles A/B/C of the full-integer INT8 model, at the validation-B threshold."""
    metrics = load_json(CLOUD_INT8_REPORT)["metrics"]
    out = {}
    for profile in ("A", "B", "C"):
        block = metrics[f"test_{profile}"]
        out[f"cloud.int8.{profile}.recall"] = round(block["recall_cloud"], 3)
        out[f"cloud.int8.{profile}.precision"] = round(block["precision_cloud"], 3)
        out[f"cloud.int8.{profile}.roc_auc"] = round(block["roc_auc"], 3)
        out[f"cloud.int8.{profile}.pr_auc"] = round(block["pr_auc"], 3)
    return out


def compute_thermal() -> dict:
    """Test profile B, FP32 and INT8 (HIL artifact at 0.2875), from run_evaluate output."""
    summary = load_json(THERMAL_EVAL)
    out = {}
    for model, block in (
        ("fp32", summary["nominal"]),
        ("int8", summary["int8"]["nominal"]),
    ):
        out[f"thermal.{model}.B.recall"] = round(block["recall"], 3)
        out[f"thermal.{model}.B.precision"] = round(block["precision"], 3)
        out[f"thermal.{model}.B.roc_auc"] = round(block["roc_auc"], 3)
        out[f"thermal.{model}.B.pr_auc"] = round(block["average_precision"], 3)
    ci = summary["nominal"]["bootstrap_ci"]["roc_auc"]
    out["thermal.fp32.B.roc_auc_ci90_low"] = round(ci["ci_low"], 3)
    out["thermal.fp32.B.roc_auc_ci90_high"] = round(ci["ci_high"], 3)
    return out


def compute() -> dict:
    return {**compute_cloud(), **compute_thermal()}


if __name__ == "__main__":
    # `python -m reproduce.table6 thermal` checks one payload while the other is pending
    payload = sys.argv[1] if len(sys.argv) > 1 else None
    got = (
        {"cloud": compute_cloud, "thermal": compute_thermal}[payload]()
        if payload
        else compute()
    )
    sys.exit(report("table6", got, payload))
