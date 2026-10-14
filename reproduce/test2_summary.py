"""Section 6.6 — OBC integration (HIL Test 2).

Reads the final ESP32 summary record of the stable log and compares every
committed sample against the embedded outputs of Test 1 (matched by image
name). The paper's 199/200 figure also uses earlier, interrupted logs that
are not in the repo yet; this script reproduces the stable-log 185/185 part.
"""

from __future__ import annotations

import json
import sys

from reproduce.lib import HIL_RESULTS, load_json, report

TEST1 = {
    "cloud_detection": "cloud_detection_results.json",
    "fire_detection": "fire_detection_results.json",
}


def parse_log(path) -> tuple[dict, list[dict]]:
    summary, telemetry = {}, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("record_type") == "summary":
            summary = rec  # keep the last one
        elif rec.get("record_type") == "telemetry":
            telemetry.append(rec)
    return summary, telemetry


def compute() -> dict:
    summary, telemetry = parse_log(HIL_RESULTS / "test2" / "Results_Test2.txt")
    link = summary["link"]
    reference = {}
    for dataset, filename in TEST1.items():
        for rec in load_json(HIL_RESULTS / "test1" / filename)["results"]:
            reference[(dataset, rec["filename"])] = rec["prediction"]

    matched = compared = 0
    for t in telemetry:
        o = t["openmv"]
        ref = reference.get((o["dataset"], o["image_name"]))
        if ref is None:
            continue
        compared += 1
        p = o["prediction"]
        matched += (
            p["predicted_id"] == ref["predicted_id"]
            and abs(p["positive_probability"] - ref["positive_probability"]) < 1e-6
            and p["raw_quantized_estimate"] == ref["raw_quantized_estimate"]
        )
    return {
        "mission_completed": summary["mission_completed"],
        "committed_samples": sum(m["committed"] for m in summary["modes"]),
        "frames": link["rx_frames"],
        "bytes": link["rx_bytes"],
        "integrity_errors": sum(
            link[k]
            for k in (
                "header_crc_errors",
                "payload_crc_errors",
                "length_errors",
                "timeouts",
                "sequence_mismatches",
            )
        ),
        "compared_with_test1": compared,
        "identical_to_test1": matched,
    }


if __name__ == "__main__":
    sys.exit(report("test2_summary", compute()))
