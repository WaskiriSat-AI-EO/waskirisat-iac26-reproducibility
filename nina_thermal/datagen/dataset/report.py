"""Dataset summary report generation.

Reads all per-image metadata JSON files and the label manifest from a
dataset directory and produces ``dataset_summary.json`` — the primary
deliverable for the paper's dataset contribution section.  Every number
in the paper's dataset table comes from this file.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from nina_thermal.datagen import config


def build_dataset_report(dataset_dir: Path) -> Dict[str, Any]:
    """Read all metadata and produce a ``dataset_summary.json`` report.

    Scans *dataset_dir* for ``*.json`` metadata files (excluding
    ``label_manifest.json`` and ``dataset_summary.json``), aggregates
    statistics, and writes ``dataset_summary.json`` to *dataset_dir*.

    Args:
        dataset_dir: Directory containing per-image ``*.json`` metadata
            files and ``label_manifest.json``.

    Returns:
        The summary dict (also written to disk as ``dataset_summary.json``).

    Structure of the returned dict::

        {
            "total_granules": int,
            "total_patches_before_rejection": int,
            "total_patches_after_rejection": int,
            "rejection_rate": float,
            "splits": {
                "train": {"granules": int, "patches": int, "fire_fraction": float},
                "val":   {...},
                "test":  {...},
            },
            "label_distribution": {
                "fire": int, "no_fire": int, "fire_fraction": float
            },
            "surface_types": {"fire": int, "warm_surface": int, "nominal": int},
            "agc_modes":  {"fixed": int, "global": int, "auto": int},
            "profiles":   {"A": int, "B": int, "C": int},
            "bt_statistics": {
                "fire_patches":   {"mean_k": float, "std_k": float,
                                   "p2_k": float, "p98_k": float},
                "nofire_patches": {...},
            },
            "geographic_coverage": {
                "lat_min": float, "lat_max": float,
                "lon_min": float, "lon_max": float,
                "patch_centroids": [{"lat": float, "lon": float, "label": int}]
            },
            "temporal_coverage": {
                "granule_dates": [str],
                "date_min": str,
                "date_max": str,
            },
            "processing_parameters": {
                "agg_factor": int,
                "synthetic_gsd_m": float,
                "nina_lepton_gsd_m": float,
                "gsd_overestimation_pct": float,
                "fixed_agc_range_k": float,
                "fire_ratio_target": float,
                "seed": int,
                "script_version": str,
            },
        }
    """
    dataset_dir = Path(dataset_dir)

    # ── Load label manifest ───────────────────────────────────────────────────
    manifest_path = dataset_dir / "label_manifest.json"
    manifest: List[Dict] = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

    total_after = len(manifest)

    # ── Load per-image metadata files ─────────────────────────────────────────
    # Metadata files are written to metadata/ by run_simulation.
    # Fall back to the dataset_dir root for backwards compatibility,
    # skipping all known manifest files.
    _meta_subdir = dataset_dir / "metadata"
    _MANIFEST_NAMES = {
        "label_manifest.json",
        "dataset_summary.json",
        "simulation_manifest.json",
        "split_manifest.json",
        "balanced_manifest.json",
    }
    if _meta_subdir.is_dir():
        meta_files = list(_meta_subdir.glob("*.json"))
    else:
        meta_files = [
            p for p in dataset_dir.glob("*.json")
            if p.name not in _MANIFEST_NAMES
        ]

    metadata_records: List[Dict] = []
    for p in meta_files:
        try:
            with open(p, encoding="utf-8") as fh:
                metadata_records.append(json.load(fh))
        except Exception:
            pass

    total_before = len(metadata_records) if metadata_records else total_after
    rejection_rate = (
        (total_before - total_after) / total_before if total_before > 0 else 0.0
    )

    # ── Aggregate from manifest ───────────────────────────────────────────────
    splits: Dict[str, Dict] = {s: {"granules": 0, "patches": 0, "fire_fraction": 0.0}
                                for s in ("train", "val", "test")}
    split_granules: Dict[str, set] = {s: set() for s in splits}
    split_fire: Dict[str, int] = defaultdict(int)

    surface_type_counts: Dict[str, int] = defaultdict(int)
    n_fire = 0
    n_nofire = 0

    for rec in manifest:
        lbl  = rec.get("label", 0)
        splt = rec.get("split", "train")
        gid  = rec.get("granule_id", "")
        st   = rec.get("surface_type", "nominal")

        if lbl == 1:
            n_fire += 1
        else:
            n_nofire += 1

        surface_type_counts[st] += 1

        if splt in splits:
            splits[splt]["patches"] += 1
            split_granules[splt].add(gid)
            if lbl == 1:
                split_fire[splt] += 1

    for splt in splits:
        n = splits[splt]["patches"]
        splits[splt]["granules"] = len(split_granules[splt])
        splits[splt]["fire_fraction"] = (split_fire[splt] / n) if n > 0 else 0.0

    total_labels = n_fire + n_nofire
    fire_frac = n_fire / total_labels if total_labels > 0 else 0.0

    # ── Aggregate from metadata records ──────────────────────────────────────
    agc_counts: Dict[str, int] = defaultdict(int)
    profile_counts: Dict[str, int] = defaultdict(int)
    centroids: List[Dict] = []
    granule_ids: set = set()
    granule_dates: List[str] = []
    seeds: List[int] = []

    fire_t_mins, fire_t_maxs = [], []
    nofire_t_mins, nofire_t_maxs = [], []

    for m in metadata_records:
        agc_counts[m.get("agc_mode", "fixed")] += 1
        profile_counts[m.get("profile", "A")] += 1
        gid = m.get("granule_id", "")
        granule_ids.add(gid)

        lat = m.get("centroid_lat")
        lon = m.get("centroid_lon")
        lbl = m.get("label", 0)
        if lat is not None and lon is not None:
            centroids.append({"lat": lat, "lon": lon, "label": lbl})

        # Extract date from granule_id (Sentinel-3 format contains date)
        date_str = _extract_date(gid)
        if date_str:
            granule_dates.append(date_str)

        if m.get("seed") is not None:
            seeds.append(m["seed"])

        t_min = m.get("t_min", config.FIXED_TEMP_MIN_K)
        t_max = m.get("t_max", config.FIXED_TEMP_MAX_K)
        if lbl == 1:
            fire_t_mins.append(t_min)
            fire_t_maxs.append(t_max)
        else:
            nofire_t_mins.append(t_min)
            nofire_t_maxs.append(t_max)

    # ── Geographic coverage ───────────────────────────────────────────────────
    lats = [c["lat"] for c in centroids]
    lons = [c["lon"] for c in centroids]
    geo = {
        "lat_min": min(lats) if lats else None,
        "lat_max": max(lats) if lats else None,
        "lon_min": min(lons) if lons else None,
        "lon_max": max(lons) if lons else None,
        "patch_centroids": centroids,
    }

    # ── Temporal coverage ─────────────────────────────────────────────────────
    unique_dates = sorted(set(granule_dates))
    temporal = {
        "granule_dates": unique_dates,
        "date_min": unique_dates[0] if unique_dates else None,
        "date_max": unique_dates[-1] if unique_dates else None,
    }

    # ── BT statistics (from t_min/t_max as proxies) ───────────────────────────
    def _stats(vals: List[float]) -> Dict:
        if not vals:
            return {"mean_k": None, "std_k": None, "p2_k": None, "p98_k": None}
        import numpy as np
        a = np.array(vals, dtype=float)
        return {
            "mean_k": float(a.mean()),
            "std_k":  float(a.std()),
            "p2_k":   float(np.percentile(a, 2)),
            "p98_k":  float(np.percentile(a, 98)),
        }

    bt_stats = {
        "fire_patches":   _stats(fire_t_mins),
        "nofire_patches": _stats(nofire_t_mins),
    }

    # ── Assemble report ───────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "total_granules":                len(granule_ids) or len(split_granules.get("train", set()) |
                                          split_granules.get("val", set()) |
                                          split_granules.get("test", set())),
        "total_patches_before_rejection": total_before,
        "total_patches_after_rejection":  total_after,
        "rejection_rate":                 rejection_rate,
        "splits":                         splits,
        "label_distribution": {
            "fire":          n_fire,
            "no_fire":       n_nofire,
            "fire_fraction": fire_frac,
        },
        "surface_types": {
            "fire":         surface_type_counts.get("fire", 0),
            "warm_surface": surface_type_counts.get("warm_surface", 0),
            "nominal":      surface_type_counts.get("nominal", 0),
        },
        "agc_modes": {
            "fixed":  agc_counts.get("fixed", 0),
            "global": agc_counts.get("global", 0),
            "auto":   agc_counts.get("auto", 0),
        },
        "profiles": {
            "A": profile_counts.get("A", 0),
            "B": profile_counts.get("B", 0),
            "C": profile_counts.get("C", 0),
        },
        "bt_statistics":        bt_stats,
        "geographic_coverage":  geo,
        "temporal_coverage":    temporal,
        "processing_parameters": {
            "agg_factor":            config.SLSTR_AGG_FACTOR,
            "synthetic_gsd_m":       config.SLSTR_SYNTHETIC_GSD_M,
            "nina_lepton_gsd_m":     config.NINA_LEPTON_GSD_M,
            "gsd_overestimation_pct": round(config.SLSTR_GSD_OVERESTIMATION * 100, 2),
            "fixed_agc_range_k":     config.FIXED_TEMP_RANGE_K,
            "fire_ratio_target":     config.FIRE_RATIO,
            "seed":                  seeds[0] if seeds else config.DEFAULT_SEED,
            "script_version":        config.SCRIPT_VERSION,
        },
    }

    out_path = dataset_dir / "dataset_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def _extract_date(granule_id: str) -> Optional[str]:
    """Extract an ISO-8601 date string from a Sentinel-3 granule ID.

    Sentinel-3 granule IDs embed the sensing start time in the format
    ``S3A_SL_1_RBT____YYYYMMDDTHHMMSS_...``.

    Args:
        granule_id: Granule identifier string.

    Returns:
        ``"YYYY-MM-DD"`` string, or ``None`` if no date found.
    """
    import re
    m = re.search(r"(\d{8})T\d{6}", granule_id)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return None
