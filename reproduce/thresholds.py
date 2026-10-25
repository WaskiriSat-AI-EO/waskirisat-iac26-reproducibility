"""Section 4 — INT8 thresholds and acceptance gate (owners: Leyla for cloud, Osvaldo for thermal).

Thermal: implemented (#15), from `models/thermal/frp20000/*_report.json`. Cloud: pending.

Contract: `compute()` returns a flat dict whose keys are
exactly those in `reproduce/expected/thresholds.json`, computed from the repo's own
artifacts (dataset manifests under data/manifests/, models listed in
models/MANIFEST.json, the payload's evaluation output). Split the work by
payload: implement `compute_cloud()` / `compute_thermal()` and merge them.
`python -m reproduce.check` reports this item as PENDING until it is done.
"""

from __future__ import annotations

import sys

from reproduce.lib import ROOT, load_json, report

THERMAL_DIR = ROOT / "models" / "thermal" / "frp20000"


def compute_cloud() -> dict:
    raise NotImplementedError("cloud: see issue #14")


def compute_thermal() -> dict:
    """FP32 threshold from training, INT8 threshold of the HIL artifact (val, recall >= 0.95)."""
    return {
        "thermal.fp32.threshold": round(
            load_json(THERMAL_DIR / "training_report.json")["selected_threshold"], 4
        ),
        "thermal.int8.threshold": round(
            load_json(THERMAL_DIR / "quantization_report.json")["int8"]["threshold"], 4
        ),
    }


def compute() -> dict:
    return {**compute_cloud(), **compute_thermal()}


if __name__ == "__main__":
    # `python -m reproduce.thresholds thermal` checks one payload while the other is pending
    payload = sys.argv[1] if len(sys.argv) > 1 else None
    got = (
        {"cloud": compute_cloud, "thermal": compute_thermal}[payload]()
        if payload
        else compute()
    )
    sys.exit(report("thresholds", got, payload))
