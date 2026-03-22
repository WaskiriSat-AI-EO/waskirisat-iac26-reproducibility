"""Load a Sentinel-3 SLSTR granule into a standard array dictionary.

Two execution paths are selected automatically:

* **Path A (fast)** — pre-extracted ``.npy`` files are present in the
  directory. Masks have already been applied during extraction.
* **Path B (fallback)** — reads directly from the ``.SEN3`` NetCDF
  directory, applies cloud / exception bitmasks on the fly, and derives
  ``delta_T``.

Both paths return an identical dict structure.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nina_thermal.ingestion.extract_channels import SLSTR_NADIR_FILES

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("F2_BT", "S8_BT", "delta_T", "lat", "lon")


def load_slstr_granule(granule_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a SLSTR granule, choosing the fastest available data path.

    Path A — ``.npy`` files present (preferred):
        Detects any ``*_F2_BT.npy`` file in *granule_dir*, extracts the
        shared prefix, then loads the four companion arrays.  ``delta_T``
        is derived as ``F2_BT - S8_BT``.  No masking is applied because
        masks were already baked in during extraction.

    Path B — ``.SEN3`` NetCDF fallback:
        Locates the ``.SEN3`` directory (either *granule_dir* itself or a
        direct sub-directory), reads ``F2_BT.nc``, ``S8_BT.nc``,
        ``geodetic_in.nc``, and ``flags_in.nc``, replaces ``_FillValue``
        sentinels with ``NaN``, applies bitmasks, and derives ``delta_T``.

    Args:
        granule_dir: Directory that contains either ``.npy`` files or a
            ``.SEN3`` sub-directory (or is itself a ``.SEN3`` directory).

    Returns:
        A dict with keys ``"F2_BT"``, ``"S8_BT"``, ``"delta_T"``,
        ``"lat"``, ``"lon"`` — all ``float32 (rows, cols)`` arrays —
        or ``None`` on any failure.

    Example::

        >>> from pathlib import Path
        >>> g = load_slstr_granule(Path("/data/extracted/S3A_20230601"))
        >>> # Path A (fast): .npy files found
        >>> sorted(g.keys())
        ['F2_BT', 'S8_BT', 'delta_T', 'lat', 'lon']
        >>> g["F2_BT"].dtype
        dtype('float32')

        >>> g2 = load_slstr_granule(Path("/data/raw/S3A_SL_1_RBT____20230601.SEN3"))
        >>> # Path B (fallback): NetCDF read + masking
        >>> sorted(g2.keys())
        ['F2_BT', 'S8_BT', 'delta_T', 'lat', 'lon']
    """
    granule_dir = Path(granule_dir)

    # ── Path A: .npy files ────────────────────────────────────────────────────
    npy_hits = list(granule_dir.glob("*_F2_BT.npy"))
    if npy_hits:
        return _load_from_npy(granule_dir, npy_hits[0])

    # ── Path B: NetCDF fallback ───────────────────────────────────────────────
    sen3_dir = _find_sen3(granule_dir)
    if sen3_dir is not None:
        return _load_from_netcdf(sen3_dir)

    logger.error(
        "load_slstr_granule: no .npy files and no .SEN3 directory found in %s",
        granule_dir,
    )
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_from_npy(
    directory: Path, f2_npy: Path
) -> Optional[Dict[str, np.ndarray]]:
    """Load pre-extracted .npy arrays for a granule.

    Args:
        directory: Parent directory containing the .npy files.
        f2_npy: Path to the ``*_F2_BT.npy`` file (used to derive prefix).

    Returns:
        Granule dict or None on failure.
    """
    try:
        prefix = f2_npy.stem[: -len("_F2_BT")]

        f2_bt = np.load(directory / f"{prefix}_F2_BT.npy").astype(np.float32)
        s8_bt = np.load(directory / f"{prefix}_S8_BT.npy").astype(np.float32)
        lat   = np.load(directory / f"{prefix}_lat.npy").astype(np.float32)
        lon   = np.load(directory / f"{prefix}_lon.npy").astype(np.float32)

        return {
            "F2_BT":   f2_bt,
            "S8_BT":   s8_bt,
            "delta_T": (f2_bt - s8_bt).astype(np.float32),
            "lat":     lat,
            "lon":     lon,
        }
    except Exception as exc:
        logger.error("load_slstr_granule (npy path) failed for %s: %s", directory, exc)
        return None


def _find_sen3(base: Path) -> Optional[Path]:
    """Locate the .SEN3 directory at *base* or as a direct child.

    Args:
        base: Directory to inspect.

    Returns:
        Path to .SEN3 directory, or None if not found.
    """
    if base.suffix == ".SEN3" and base.is_dir():
        return base
    for child in base.iterdir():
        if child.suffix == ".SEN3" and child.is_dir():
            return child
    return None


def _load_from_netcdf(sen3_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Read a .SEN3 directory via NetCDF, apply bitmasks, return granule dict.

    Args:
        sen3_dir: Path to the .SEN3 directory.

    Returns:
        Granule dict or None on failure.
    """
    try:
        import netCDF4  # type: ignore
    except ImportError:
        logger.error(
            "netCDF4 is required for .SEN3 loading. Install it with: pip install netCDF4"
        )
        return None

    try:
        # F2_BT
        with netCDF4.Dataset(sen3_dir / SLSTR_NADIR_FILES["F2_BT"]) as ds:
            var = ds.variables["F2_BT_in"]
            f2_bt = np.array(var[:], dtype=np.float32)
            _replace_fill(f2_bt, getattr(var, "_FillValue", None))

        # S8_BT
        with netCDF4.Dataset(sen3_dir / SLSTR_NADIR_FILES["S8_BT"]) as ds:
            var = ds.variables["S8_BT_in"]
            s8_bt = np.array(var[:], dtype=np.float32)
            _replace_fill(s8_bt, getattr(var, "_FillValue", None))

        # Geodetic
        with netCDF4.Dataset(sen3_dir / SLSTR_NADIR_FILES["geodetic"]) as ds:
            lat = np.array(ds.variables["latitude_in"][:], dtype=np.float32)
            lon = np.array(ds.variables["longitude_in"][:], dtype=np.float32)

        # Bitmasks
        with netCDF4.Dataset(sen3_dir / SLSTR_NADIR_FILES["flags"]) as ds:
            cloud_in      = np.array(ds.variables["cloud_in"][:])
            confidence_in = np.array(ds.variables["confidence_in"][:])

        cloud_mask     = (cloud_in & 1).astype(bool)
        exception_mask = (confidence_in & 2).astype(bool)
        combined       = cloud_mask | exception_mask

        f2_bt[combined] = np.nan
        s8_bt[combined] = np.nan

        return {
            "F2_BT":   f2_bt,
            "S8_BT":   s8_bt,
            "delta_T": (f2_bt - s8_bt).astype(np.float32),
            "lat":     lat,
            "lon":     lon,
        }

    except Exception as exc:
        logger.error("load_slstr_granule (NetCDF path) failed for %s: %s", sen3_dir, exc)
        return None


def _replace_fill(arr: np.ndarray, fill_value) -> None:
    """Replace *fill_value* sentinels in *arr* with NaN (in-place).

    Args:
        arr: float32 array to modify in place.
        fill_value: Scalar fill sentinel, or None to skip.
    """
    if fill_value is not None:
        arr[arr == float(fill_value)] = np.nan
