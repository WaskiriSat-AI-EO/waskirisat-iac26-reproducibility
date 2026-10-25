"""Table 5 — dataset composition (owners: Leyla for cloud, Osvaldo for thermal).

Thermal: implemented (#15), from `data/manifests/thermal_v1_*`. Cloud: pending.

Contract: `compute()` returns a flat dict whose keys are
exactly those in `reproduce/expected/table5.json`, computed from the repo's own
artifacts (dataset manifests under data/manifests/, models listed in
models/MANIFEST.json, the payload's evaluation output). Split the work by
payload: implement `compute_cloud()` / `compute_thermal()` and merge them.
`python -m reproduce.check` reports this item as PENDING until it is done.
"""

from __future__ import annotations

import sys

from reproduce.lib import ROOT, load_json, report

THERMAL_SUMMARY = ROOT / "data" / "manifests" / "thermal_v1_dataset_summary.json"


def compute_cloud() -> dict:
    raise NotImplementedError("cloud: see issue #14")


def compute_thermal() -> dict:
    """Split composition from the v1 dataset manifest (one frame per patch, profile B)."""
    splits = load_json(THERMAL_SUMMARY)["splits"]
    out = {}
    for split in ("train", "val", "test"):
        s = splits[split]
        for key in ("frames", "granules", "positive", "negative"):
            out[f"thermal.{split}.{key}"] = s[key]
        out[f"thermal.{split}.positive_rate"] = round(s["positive_rate"], 3)
    return out


def compute() -> dict:
    return {**compute_cloud(), **compute_thermal()}


if __name__ == "__main__":
    # `python -m reproduce.table5 thermal` checks one payload while the other is pending
    payload = sys.argv[1] if len(sys.argv) > 1 else None
    got = (
        {"cloud": compute_cloud, "thermal": compute_thermal}[payload]()
        if payload
        else compute()
    )
    sys.exit(report("table5", got, payload))
