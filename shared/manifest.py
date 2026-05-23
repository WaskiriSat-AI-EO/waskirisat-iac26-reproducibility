"""Shared manifest helpers for nina pipeline stages.

Provides normalised read/write and cross-stage chain validation so that data
loss between stages is detected and reported rather than silently propagated.

Usage::

    from shared.manifest import load_records, expected_scene_ids, assert_chain

    with open(path) as fh:
        body = json.load(fh)
    records = load_records(body, key="scenes")
    upstream_ids = expected_scene_ids(records)

    produced_ids = {r["scene_id"] for r in output_records}
    ok = assert_chain(upstream_ids, produced_ids, logger)
    if not ok:
        return log_run_footer(start_time, 1)
"""

import logging
from typing import Any, Dict, Iterable, List, Set


def load_records(body: Any, key: str) -> List[Dict]:
    """Extract the record list from an envelope-or-bare-array manifest body.

    Args:
        body: Parsed JSON — either a dict with a records list under *key*, or
              directly a list (bare-array format used by label_manifest.json).
        key:  The envelope key to look up when *body* is a dict (e.g. ``"scenes"``,
              ``"tiles"``, ``"records"``).

    Returns:
        List of record dicts. Empty list if the key is absent or body is empty.
    """
    if isinstance(body, list):
        return body
    return body.get(key, [])


def expected_scene_ids(
    records: Iterable[Dict],
    status_field: str = "status",
    ok_value: str = "ok",
) -> Set[str]:
    """Collect the scene_ids from an upstream manifest that should propagate downstream.

    Records without a *status_field* key are assumed to always be included
    (e.g. tiles_manifest records have no status field).  Records where
    *status_field* is present are only included if its value equals *ok_value*.
    Records missing a ``"scene_id"`` key (e.g. failed extraction records) are
    silently skipped.

    Args:
        records:      Iterable of manifest record dicts.
        status_field: The field name that indicates record health.
        ok_value:     The value that means the record is valid.

    Returns:
        Set of scene_id strings expected in the next stage.
    """
    ids: Set[str] = set()
    for rec in records:
        if status_field in rec and rec[status_field] != ok_value:
            continue
        sid = rec.get("scene_id")
        if sid is not None:
            ids.add(sid)
    return ids


def assert_chain(
    expected_ids: Set[str],
    produced_ids: Set[str],
    logger: logging.Logger,
) -> bool:
    """Check that all expected scene_ids appear in the produced set.

    Logs an ERROR for every missing scene_id and returns False when any are
    missing so the caller can fail fast (return exit code 1).  Scenes that
    appear in *produced_ids* but not *expected_ids* are ignored (e.g. a stage
    might produce extra debug records).

    Args:
        expected_ids: Scene IDs that must be present downstream.
        produced_ids: Scene IDs actually produced by the current stage.
        logger:       Logger to emit diagnostics on.

    Returns:
        True if no scenes are missing; False otherwise.
    """
    missing = expected_ids - produced_ids
    if not missing:
        return True

    logger.error(
        "Manifest chain broken: %d scene(s) expected but absent in output — "
        "investigate upstream failures before re-running. Missing: %s",
        len(missing),
        ", ".join(sorted(missing)),
    )
    return False
