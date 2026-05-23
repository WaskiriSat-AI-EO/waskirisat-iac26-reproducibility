"""Dataset summary report generation.

Produces dataset_summary.json with all statistics needed for the IAC paper.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from nina_cloud.datagen import config

logger = logging.getLogger(__name__)


def build_dataset_report(
    label_manifest: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    """Compute dataset statistics and write dataset_summary.json.

    Args:
        label_manifest: List of per-image metadata dicts (from save_label_manifest).
        output_path: Path to write the JSON report.

    Returns:
        The summary dict (also written to disk).
    """
    splits = {"train": [], "val": [], "test": []}
    for rec in label_manifest:
        s = rec.get("split")
        if s in splits:
            splits[s].append(rec)

    def split_stats(recs: List[Dict]) -> Dict[str, Any]:
        n_total = len(recs)
        n_cloud = sum(1 for r in recs if r["label"] == 1)
        n_clear = sum(1 for r in recs if r["label"] == 0)
        scenes  = len({r["scene_id"] for r in recs})
        cloud_fracs = [r["cloud_frac"] for r in recs]
        profiles = {}
        for r in recs:
            p = r.get("profile", "?")
            profiles[p] = profiles.get(p, 0) + 1
        return {
            "n_total":      n_total,
            "n_cloud":      n_cloud,
            "n_clear":      n_clear,
            "n_scenes":     scenes,
            "cloud_frac_mean": round(sum(cloud_fracs) / len(cloud_fracs), 4) if cloud_fracs else 0.0,
            "profiles":     profiles,
        }

    summary = {
        "script_version": config.SCRIPT_VERSION,
        "sensor":         "PAG7936",
        "gsd_m":          config.PAG7936_GSD_M,
        "source":         "Sentinel-2 L2A",
        "tile_size_px":   [config.TILE_HEIGHT_PX, config.TILE_WIDTH_PX],
        "cloud_classes":  sorted(config.SCL_CLOUD_CLASSES),
        "label_thresholds": {
            "high_thresh": config.CLOUD_LABEL_HIGH_THRESH,
            "low_thresh":  config.CLOUD_LABEL_LOW_THRESH,
        },
        "target_cloud_ratio": config.CLOUD_RATIO,
        "total_images":   len(label_manifest),
        "splits": {
            split_name: split_stats(recs)
            for split_name, recs in splits.items()
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "dataset report: total=%d  splits=%s",
        len(label_manifest),
        {k: v["n_total"] for k, v in summary["splits"].items()},
    )
    return summary
