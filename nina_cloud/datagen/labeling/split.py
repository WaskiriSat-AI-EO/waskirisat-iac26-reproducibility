"""Registry-based train/val/test split assignment.

CRITICAL: splits are always at the S2 scene level.  Tiles from the same
scene share atmospheric conditions, sun angle, and cloud patterns.
Tile-level splitting would produce optimistic validation metrics due to
spatial autocorrelation.

Split assignment uses a deterministic hash of the scene_id
(``shared.scene_registry.assign_split``) — the same scene always maps to
the same split regardless of run order, dataset size, or which operator
runs the pipeline.  Splits are stored in the scene registry at acquisition
time and read back here; scenes not yet in the registry fall back to
``assign_split`` automatically.
"""

import logging
from typing import Any, Dict, List

from shared.scene_registry import Registry, split_for

logger = logging.getLogger(__name__)


def assign_splits_from_registry(
    records: List[Dict[str, Any]],
    registry: Registry,
) -> List[Dict[str, Any]]:
    """Stamp each record with its split assignment from *registry*.

    Reads the split from the scene registry for every record's
    ``scene_id``.  Scenes absent from the registry fall back to
    ``assign_split`` (deterministic hash) with a logged warning.

    The function mutates *records* in place and also returns the list for
    chaining convenience.

    Args:
        records: List of manifest record dicts, each containing at minimum
            ``"scene_id"`` (str) and ``"label"`` (int 0/1/-1).
        registry: Loaded scene registry (from
            ``shared.scene_registry.load_registry``).  May be an empty
            dict for pipelines that have not yet adopted the registry.

    Returns:
        The same *records* list with every record's ``"split"`` field
        populated (one of ``"train"``, ``"val"``, ``"test"``).
    """
    scene_splits: Dict[str, str] = {}

    for rec in records:
        sid = rec["scene_id"]
        if sid not in scene_splits:
            scene_splits[sid] = split_for(registry, sid)
        rec["split"] = scene_splits[sid]

    # Diagnostic summary — mirrors the logging that existed in the old
    # scene_level_split so operators can spot distribution drift.
    split_stats: Dict[str, Dict[str, Any]] = {
        s: {"scenes": set(), "tiles": 0, "cloud": 0, "definitive": 0}
        for s in ("train", "val", "test")
    }
    for rec in records:
        sp = rec["split"]
        if sp not in split_stats:
            continue
        st = split_stats[sp]
        st["scenes"].add(rec["scene_id"])
        st["tiles"] += 1
        lbl = rec.get("label", -1)
        if lbl == 1:
            st["cloud"] += 1
            st["definitive"] += 1
        elif lbl == 0:
            st["definitive"] += 1

    for sp, st in split_stats.items():
        frac = (st["cloud"] / st["definitive"]) if st["definitive"] else 0.0
        logger.info(
            "[%s] scenes=%d  tiles=%d  cloud_frac=%.3f",
            sp,
            len(st["scenes"]),
            st["tiles"],
            frac,
        )

    return records
