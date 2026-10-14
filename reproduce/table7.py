"""Table 7 — surrogate execution on the OpenMV H7 Plus (T1), from HIL Test 1."""

from __future__ import annotations

import sys

from reproduce.lib import HIL_RESULTS, load_json, report

SOURCES = {
    "thermal": "fire_detection_results.json",
    "cloud": "cloud_detection_results.json",
}


def compute() -> dict:
    out = {}
    for payload, filename in SOURCES.items():
        data = load_json(HIL_RESULTS / "test1" / filename)
        out[f"{payload}.inference_ms_mean"] = round(
            data["summary"]["timing"]["inference_estimated_ms"]["mean"], 1
        )
        out[f"{payload}.arena_bytes"] = data["model"]["arena_bytes"]
        out[f"{payload}.min_free_heap_bytes"] = data["summary"]["memory"][
            "minimum_free_bytes"
        ]
        out[f"{payload}.model_size_bytes"] = data["model"]["size_bytes"]
        out[f"{payload}.images"] = data["summary"]["images_processed"]
        out[f"{payload}.image_errors"] = data["summary"]["image_errors"]
    return out


if __name__ == "__main__":
    sys.exit(report("table7", compute()))
