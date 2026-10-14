"""Table 8 — device-path fidelity against the host reference runtime (T1).

Recomputed from the per-sample records of HIL Test 1, not read from the
summary block. The third row of the paper (cloud, host-generated input
tensor: 97/100, r = 0.997) comes from a separate controlled experiment whose
raw per-sample output is not in the repo yet (see issue on HIL).
"""

from __future__ import annotations

import sys

import numpy as np

from reproduce.lib import HIL_RESULTS, load_json, report

SOURCES = {
    "thermal": "fire_detection_results.json",
    "cloud": "cloud_detection_results.json",
}


def fidelity(records: list[dict]) -> dict:
    device = np.array([r["prediction"]["positive_probability"] for r in records])
    host = np.array([r["desktop_reference"]["positive_probability"] for r in records])
    agree = sum(r["comparison"]["decision_matches_desktop"] for r in records)
    return {
        "agreement": agree,
        "n": len(records),
        "correlation": round(float(np.corrcoef(device, host)[0, 1]), 3),
        "bias": round(float(np.mean(device - host)), 3),
    }


def compute() -> dict:
    out = {}
    for payload, filename in SOURCES.items():
        data = load_json(HIL_RESULTS / "test1" / filename)
        f = fidelity(data["results"])
        out[f"{payload}.device_path.agreement"] = f["agreement"]
        out[f"{payload}.device_path.n"] = f["n"]
        out[f"{payload}.device_path.correlation"] = f["correlation"]
        out[f"{payload}.device_path.bias"] = f["bias"]
    return out


if __name__ == "__main__":
    sys.exit(report("table8", compute()))
