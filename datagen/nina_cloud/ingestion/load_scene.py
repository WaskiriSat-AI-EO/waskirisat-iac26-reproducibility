"""Load a Sentinel-2 L2A scene into the standard array dict.

Two execution paths are selected automatically:

* **Path A (fast)** — pre-extracted ``.npy`` files are present.
* **Path B (fallback)** — reads directly from the ``.SAFE`` directory via
  :func:`extract_bands.extract_s2_scene` into a temporary location.

Both paths return an identical dict structure with ``"rgb"`` and ``"scl"`` keys.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_s2_scene(scene_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load an S2 scene, choosing the fastest available data path.

    Path A — ``.npy`` files present (preferred):
        Detects any ``*_rgb.npy`` file in *scene_dir*, loads the companion
        ``*_scl.npy``, and returns the dict immediately.

    Path B — ``.SAFE`` directory fallback:
        Locates the ``.SAFE`` directory (either *scene_dir* itself or a
        direct sub-directory), calls :func:`extract_s2_scene` into a
        temporary cache inside *scene_dir*, then loads the resulting .npy.

    Args:
        scene_dir: Directory that contains either ``.npy`` files or a
            ``.SAFE`` sub-directory (or is itself a ``.SAFE`` directory).

    Returns:
        Dict with keys ``"rgb"`` (float32 H×W×3) and ``"scl"`` (uint8 H/2×W/2),
        or ``None`` on any failure.
    """
    scene_dir = Path(scene_dir)

    # ── Path A: .npy files ────────────────────────────────────────────────────
    npy_hits = list(scene_dir.glob("*_rgb.npy"))
    if npy_hits:
        return _load_from_npy(scene_dir, npy_hits[0])

    # ── Path B: .SAFE fallback ────────────────────────────────────────────────
    safe_dir = _find_safe(scene_dir)
    if safe_dir is not None:
        return _load_from_safe(safe_dir, scene_dir)

    logger.error(
        "load_s2_scene: no .npy files and no .SAFE directory found in %s",
        scene_dir,
    )
    return None


def _load_from_npy(
    directory: Path, rgb_npy: Path
) -> Optional[Dict[str, np.ndarray]]:
    try:
        prefix = rgb_npy.stem[: -len("_rgb")]
        rgb = np.load(directory / f"{prefix}_rgb.npy").astype(np.float32)
        scl = np.load(directory / f"{prefix}_scl.npy").astype(np.uint8)
        logger.debug("load_s2_scene (npy path): %s  rgb=%s  scl=%s", prefix, rgb.shape, scl.shape)
        return {"rgb": rgb, "scl": scl}
    except Exception as exc:
        logger.error("load_s2_scene (npy path) failed: %s", exc)
        return None


def _find_safe(base: Path) -> Optional[Path]:
    if base.suffix == ".SAFE" and base.is_dir():
        return base
    for child in base.iterdir():
        if child.suffix == ".SAFE" and child.is_dir():
            return child
    return None


def _load_from_safe(
    safe_dir: Path, output_dir: Path
) -> Optional[Dict[str, np.ndarray]]:
    from nina_cloud.ingestion.extract_bands import extract_s2_scene

    scene_id = extract_s2_scene(safe_dir, output_dir, overwrite=False)
    if scene_id is None:
        return None

    rgb_npy = output_dir / f"{scene_id}_rgb.npy"
    return _load_from_npy(output_dir, rgb_npy)
