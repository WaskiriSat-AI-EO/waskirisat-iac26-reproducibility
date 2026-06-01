"""Scene registry for incremental, multi-operator dataset accumulation.

The registry is a JSON file (default ``data/cloud/scene_registry.json``)
mapping *scene_id* → record dict.  It is the single source of truth for:

- **Split assignment** — written once per scene, never changed.
- **Deduplication** — operators skip scenes already registered.
- **Provenance** — who downloaded it, when, in what cloud mode.

Split assignment is a *deterministic pure function* of ``scene_id``
(``assign_split``), so the same split is produced whether read from the
registry or re-derived from scratch.  The registry is an audit trail;
losing it does not corrupt existing data.

Atomic writes (write-then-rename) protect against concurrent corruption
when two operators race on the same shared drive.

Usage::

    from shared.scene_registry import (
        assign_split, load_registry, save_registry,
        register_scene, is_registered, split_for, operator_name,
    )

    registry = load_registry(Path("data/cloud/scene_registry.json"))
    if not is_registered(registry, scene_id):
        register_scene(registry, scene_id, cloud_mode="clear", operator=operator_name())
        save_registry(Path("data/cloud/scene_registry.json"), registry)

    split = split_for(registry, scene_id)   # "train" | "val" | "test"
"""

import getpass
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from shared.seeding import derive_seed

logger = logging.getLogger(__name__)

# Type alias
Registry = Dict[str, Dict]

# Default split cut-points (0-99 bucket space).
# These match TRAIN_FRAC=0.70, VAL_FRAC=0.15, TEST_FRAC=0.15 from
# nina_cloud.datagen.config.  Callers that use a different split ratio
# should pass custom cut points to assign_split().
_DEFAULT_TRAIN_CUT: int = 70
_DEFAULT_VAL_CUT: int = 85


def assign_split(
    scene_id: str,
    train_cut: int = _DEFAULT_TRAIN_CUT,
    val_cut: int = _DEFAULT_VAL_CUT,
) -> str:
    """Return the split for *scene_id* — pure, deterministic, seed-free.

    Maps the scene_id to a bucket in [0, 100) via a blake2b hash
    (``derive_seed(scene_id, "split") % 100``) and thresholds it:

    - bucket  < *train_cut*          → ``"train"``
    - bucket  < *val_cut*            → ``"val"``
    - otherwise                      → ``"test"``

    The result is independent of Python's salted ``hash()``, run order,
    dataset size, and operator.  The same scene always maps to the same
    split as long as *train_cut* and *val_cut* are unchanged.

    Args:
        scene_id:  Canonical scene identifier (SAFE product name without
            the ``.SAFE`` suffix).
        train_cut: Upper-exclusive bucket threshold for ``"train"``.
            Default 70 → 70 % train.
        val_cut:   Upper-exclusive bucket threshold for ``"val"``.
            Default 85 → 15 % val, 15 % test.

    Returns:
        One of ``"train"``, ``"val"``, ``"test"``.
    """
    bucket = derive_seed(scene_id, "split") % 100
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def load_registry(path: Path) -> Registry:
    """Load the registry from *path*, returning an empty dict if absent.

    Args:
        path: Path to the JSON registry file.

    Returns:
        Dict mapping scene_id → record dict.  Empty dict if the file does
        not exist yet (first run).
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def save_registry(path: Path, registry: Registry) -> None:
    """Atomically write *registry* to *path*.

    Writes to a sibling temp file then calls ``os.replace`` — atomic on
    POSIX, effectively atomic on Windows (replaces without a window where
    the file is absent).

    Args:
        path:     Destination file path.
        registry: Dict to serialise as JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".registry_tmp_")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(registry, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def register_scene(
    registry: Registry,
    scene_id: str,
    *,
    cloud_mode: str,
    operator: str,
    status: str = "acquired",
) -> bool:
    """Register *scene_id* in *registry* — write-once split, mutable provenance.

    If the scene is already present the split field is **never changed**.
    Only ``status``, ``cloud_mode``, ``operator``, and ``last_seen_at``
    are updated.

    Args:
        registry:   In-memory registry dict (mutated in place).
        scene_id:   Canonical scene identifier.
        cloud_mode: ``"clear"`` or ``"cloudy"``.
        operator:   Operator identifier (e.g. login name).
        status:     Processing status string.

    Returns:
        ``True`` if this is a **new** registration, ``False`` if the scene
        was already present (caller may skip re-downloading).
    """
    now = datetime.now(timezone.utc).isoformat()
    if scene_id in registry:
        registry[scene_id].update(
            {
                "status": status,
                "cloud_mode": cloud_mode,
                "operator": operator,
                "last_seen_at": now,
            }
        )
        return False
    registry[scene_id] = {
        "split": assign_split(scene_id),
        "cloud_mode": cloud_mode,
        "registered_at": now,
        "last_seen_at": now,
        "operator": operator,
        "status": status,
    }
    return True


def is_registered(registry: Registry, scene_id: str) -> bool:
    """Return ``True`` if *scene_id* is present in *registry*."""
    return scene_id in registry


def split_for(registry: Registry, scene_id: str) -> str:
    """Return the split for *scene_id* from registry, falling back to hash.

    If the scene is in the registry its recorded split is returned.
    Otherwise ``assign_split`` is called and a warning is logged — this
    handles scenes that were acquired before the registry existed.

    Args:
        registry: Loaded registry dict.
        scene_id: Canonical scene identifier.

    Returns:
        One of ``"train"``, ``"val"``, ``"test"``.
    """
    if scene_id in registry:
        return registry[scene_id]["split"]
    logger.warning(
        "scene_id %r not in registry — using deterministic assign_split fallback",
        scene_id,
    )
    return assign_split(scene_id)


def operator_name() -> str:
    """Return the current operator identifier.

    Checks the ``NINA_OPERATOR`` environment variable first; falls back
    to the OS login name via ``getpass.getuser()``.
    """
    return os.environ.get("NINA_OPERATOR") or getpass.getuser()
