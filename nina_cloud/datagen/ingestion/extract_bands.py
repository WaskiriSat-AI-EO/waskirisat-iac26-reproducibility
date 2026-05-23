"""Extract Sentinel-2 L2A bands from a .SAFE directory to .npy arrays.

Reads:
  - B02 (Blue), B03 (Green), B04 (Red) from IMG_DATA/R10m/ as float32 BOA [0,1]
  - SCL from IMG_DATA/R20m/ as uint8

Saves:
  - {scene_id}_rgb.npy   — float32 (H, W, 3), BOA reflectance
  - {scene_id}_scl.npy   — uint8  (H//2, W//2), SCL
  - {scene_id}_meta.json — sensing date, cloud_cover, MGRS tile, scene_id

Band order in the RGB array: R=B04, G=B03, B=B02 (standard RGB).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from nina_cloud.datagen import config
from nina_cloud.datagen.processing.reflectance import normalize_reflectance

logger = logging.getLogger(__name__)


def extract_s2_scene(
    safe_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Optional[str]:
    """Extract B02/B03/B04 and SCL from a Sentinel-2 L2A .SAFE directory.

    Args:
        safe_dir: Path to the .SAFE directory (e.g. ``S2A_MSIL2A_...SAFE``).
        output_dir: Directory where .npy and .json files will be written.
        overwrite: If False, skip extraction if output files already exist.

    Returns:
        ``scene_id`` string on success, ``None`` on failure.
    """
    try:
        import rasterio  # type: ignore
    except ImportError:
        logger.error("rasterio is required for .SAFE extraction: pip install rasterio")
        return None

    safe_dir = Path(safe_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id = _parse_scene_id(safe_dir.name)
    out_rgb = output_dir / f"{scene_id}_rgb.npy"
    out_scl = output_dir / f"{scene_id}_scl.npy"
    out_meta = output_dir / f"{scene_id}_meta.json"

    if not overwrite and out_rgb.exists() and out_scl.exists():
        logger.info("extract_s2_scene: already exists, skipping — %s", scene_id)
        return scene_id

    granule_dir = _find_granule_dir(safe_dir)
    if granule_dir is None:
        logger.error("extract_s2_scene: no GRANULE dir in %s", safe_dir)
        return None

    r10m_dir = granule_dir / "IMG_DATA" / "R10m"
    r20m_dir = granule_dir / "IMG_DATA" / "R20m"

    try:
        b04 = _read_jp2_band(r10m_dir, "*B04_10m.jp2", rasterio)
        b03 = _read_jp2_band(r10m_dir, "*B03_10m.jp2", rasterio)
        b02 = _read_jp2_band(r10m_dir, "*B02_10m.jp2", rasterio)
    except FileNotFoundError as e:
        logger.error("extract_s2_scene: missing 10m band — %s", e)
        return None

    rgb_dn = np.stack([b04, b03, b02], axis=-1)  # (H, W, 3) R-G-B order
    rgb_float = normalize_reflectance(rgb_dn)

    try:
        scl_raw = _read_jp2_band(r20m_dir, "*SCL_20m.jp2", rasterio)
    except FileNotFoundError as e:
        logger.error("extract_s2_scene: missing SCL band — %s", e)
        return None

    scl_uint8 = scl_raw.astype(np.uint8)

    if scl_uint8.shape != (rgb_float.shape[0] // 2, rgb_float.shape[1] // 2):
        logger.warning(
            "extract_s2_scene: unexpected SCL shape %s vs RGB %s for %s",
            scl_uint8.shape,
            rgb_float.shape[:2],
            scene_id,
        )

    np.save(out_rgb, rgb_float)
    np.save(out_scl, scl_uint8)

    meta = _build_meta(safe_dir, scene_id, rgb_float.shape)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "extracted %s: rgb=%s  scl=%s",
        scene_id,
        rgb_float.shape,
        scl_uint8.shape,
    )
    return scene_id


def _read_jp2_band(directory: Path, pattern: str, rasterio) -> np.ndarray:
    """Read the first JP2 file matching *pattern* from *directory*.

    Args:
        directory: Directory to search.
        pattern: Glob pattern for the JP2 filename.
        rasterio: The rasterio module (passed to avoid repeated imports).

    Returns:
        2-D uint16 array of raw DNs.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    with rasterio.open(matches[0]) as ds:
        return ds.read(1)


def _find_granule_dir(safe_dir: Path) -> Optional[Path]:
    """Locate the GRANULE sub-directory inside a .SAFE hierarchy."""
    granule_root = safe_dir / "GRANULE"
    if not granule_root.exists():
        return None
    children = [p for p in granule_root.iterdir() if p.is_dir()]
    return children[0] if children else None


def _parse_scene_id(safe_name: str) -> str:
    """Extract a compact scene ID from the .SAFE directory name.

    Uses the full name minus the trailing '.SAFE' suffix.
    """
    return safe_name.removesuffix(".SAFE")


def _build_meta(safe_dir: Path, scene_id: str, rgb_shape: tuple) -> dict:
    """Build a metadata dict for the extracted scene."""
    name = safe_dir.name
    sensing_date = ""
    cloud_cover = None
    mgrs_tile = ""

    # Parse sensing date from S2 standard naming: S2X_MSIL2A_YYYYMMDDTHHMMSS_...
    m = re.search(r"_(\d{8}T\d{6})_", name)
    if m:
        sensing_date = m.group(1)

    # Parse MGRS tile: ...T<tile>_...
    m2 = re.search(r"_T(\d{2}[A-Z]{3})_", name)
    if m2:
        mgrs_tile = m2.group(1)

    return {
        "scene_id": scene_id,
        "sensing_date": sensing_date,
        "mgrs_tile": mgrs_tile,
        "cloud_cover": cloud_cover,
        "rgb_shape": list(rgb_shape),
        "s2_gsd_m": config.S2_GSD_M,
        "script_version": config.SCRIPT_VERSION,
    }
