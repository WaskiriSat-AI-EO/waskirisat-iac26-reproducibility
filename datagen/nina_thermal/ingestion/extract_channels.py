"""Extract thermal channels from a Sentinel-3 SLSTR .SEN3 directory.

Reads F2_BT, S8_BT, latitude, and longitude from NetCDF files,
applies cloud and exception bitmasks, and saves the results as .npy files.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_thermal_channels(
    sen3_dir: Path,
    output_dir: Path,
    prefix: Optional[str] = None,
) -> Optional[Tuple[Path, Path, Path, Path]]:
    """Extract F2_BT, S8_BT, lat, lon arrays from a .SEN3 directory.

    Reads NetCDF variables, replaces _FillValue sentinels with NaN,
    applies a combined cloud + exception bitmask (masked pixels → NaN),
    and writes four .npy files to output_dir.

    Args:
        sen3_dir: Path to the .SEN3 directory containing NetCDF files.
        output_dir: Destination directory for the four .npy files.
        prefix: Filename prefix for the output files. Defaults to the
            .SEN3 directory stem (without the .SEN3 extension).

    Returns:
        A 4-tuple of Paths:
            (F2_BT.npy, S8_BT.npy, lat.npy, lon.npy)
        or None if extraction fails for any reason.

    The output files follow the naming convention::

        {prefix}_F2_BT.npy  — float32 brightness temperature [K]
        {prefix}_S8_BT.npy  — float32 brightness temperature [K]
        {prefix}_lat.npy    — float32 latitude  [degrees]
        {prefix}_lon.npy    — float32 longitude [degrees]

    Bitmask logic applied before saving:

        cloud_mask     = (cloud_in     & 1).astype(bool)
        exception_mask = (confidence_in & 2).astype(bool)
        combined       = cloud_mask | exception_mask
        F2_BT[combined] = NaN
        S8_BT[combined] = NaN

    Example::

        >>> from pathlib import Path
        >>> paths = extract_thermal_channels(
        ...     Path("/data/S3A_SL_1_RBT____20230601.SEN3"),
        ...     Path("/data/extracted"),
        ... )
        >>> # Returns (Path(..._F2_BT.npy), Path(..._S8_BT.npy),
        >>> #          Path(..._lat.npy),   Path(..._lon.npy))
        >>> # or None on failure.
    """
    try:
        import netCDF4  # type: ignore
    except ImportError:
        logger.error("netCDF4 is required for .SEN3 extraction. Install it with: pip install netCDF4")
        return None

    try:
        sen3_dir = Path(sen3_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            stem = sen3_dir.stem
            prefix = stem[:-5] if stem.endswith(".SEN3") else stem

        # ── F2_BT ────────────────────────────────────────────────────────────
        f2_path = sen3_dir / "F2_BT.nc"
        with netCDF4.Dataset(f2_path) as ds:
            var = ds.variables["F2_BT"]
            f2_bt = np.array(var[:], dtype=np.float32)
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                f2_bt[f2_bt == float(fill)] = np.nan

        # ── S8_BT ────────────────────────────────────────────────────────────
        s8_path = sen3_dir / "S8_BT.nc"
        with netCDF4.Dataset(s8_path) as ds:
            var = ds.variables["S8_BT"]
            s8_bt = np.array(var[:], dtype=np.float32)
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                s8_bt[s8_bt == float(fill)] = np.nan

        # ── Geodetic ─────────────────────────────────────────────────────────
        geo_path = sen3_dir / "geodetic_in.nc"
        with netCDF4.Dataset(geo_path) as ds:
            lat = np.array(ds.variables["latitude_in"][:], dtype=np.float32)
            lon = np.array(ds.variables["longitude_in"][:], dtype=np.float32)

        # ── Bitmasks ─────────────────────────────────────────────────────────
        flags_path = sen3_dir / "flags_in.nc"
        with netCDF4.Dataset(flags_path) as ds:
            cloud_in = np.array(ds.variables["cloud_in"][:])
            confidence_in = np.array(ds.variables["confidence_in"][:])

        cloud_mask = (cloud_in & 1).astype(bool)
        exception_mask = (confidence_in & 2).astype(bool)
        combined = cloud_mask | exception_mask

        f2_bt[combined] = np.nan
        s8_bt[combined] = np.nan

        # ── Save ─────────────────────────────────────────────────────────────
        p_f2 = output_dir / f"{prefix}_F2_BT.npy"
        p_s8 = output_dir / f"{prefix}_S8_BT.npy"
        p_lat = output_dir / f"{prefix}_lat.npy"
        p_lon = output_dir / f"{prefix}_lon.npy"

        np.save(p_f2, f2_bt)
        np.save(p_s8, s8_bt)
        np.save(p_lat, lat)
        np.save(p_lon, lon)

        logger.info("Extracted %s → %s", sen3_dir.name, output_dir)
        return p_f2, p_s8, p_lat, p_lon

    except Exception as exc:
        logger.error("Failed to extract %s: %s", sen3_dir, exc)
        return None
