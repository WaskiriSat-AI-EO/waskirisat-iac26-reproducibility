"""Table 5 — dataset composition (owners: Leyla for cloud, Osvaldo for thermal).

Thermal: from `data/manifests/thermal_v1_*`. Cloud: counted from the delivered dataset manifest.

Contract: `compute()` returns a flat dict whose keys are
exactly those in `reproduce/expected/table5.json`, computed from the repo's own
artifacts (dataset manifests under data/manifests/, models listed in
models/MANIFEST.json, the payload's evaluation output). Split the work by
payload: implement `compute_cloud()` / `compute_thermal()` and merge them.
`python -m reproduce.check` reports this item as PENDING until it is done.
"""

from __future__ import annotations

import sys
from collections import Counter

from reproduce.lib import ROOT, load_json, report

THERMAL_SUMMARY = ROOT / "data" / "manifests" / "thermal_v1_dataset_summary.json"
CLOUD_MANIFEST = (
    ROOT
    / "nina_cloud_results"
    / "dataset"
    / "dataset_final_repo"
    / "final_manifest.json"
)


def compute_cloud() -> dict:
    """Tiles per split and profile, counted from the cloud `final_manifest.json`."""
    counts: Counter = Counter()
    for tile in load_json(CLOUD_MANIFEST):
        counts[(tile["split"], tile["profile"], tile["class_name"])] += 1
    out = {}
    for split, profile in (("val", "B"), ("test", "A"), ("test", "B"), ("test", "C")):
        positive = counts[(split, profile, "cloud")]
        negative = counts[(split, profile, "clear")]
        prefix = f"cloud.{split}_profile_{profile.lower()}"
        out[f"{prefix}.frames"] = positive + negative
        out[f"{prefix}.positive"] = positive
        out[f"{prefix}.negative"] = negative
    return out


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
