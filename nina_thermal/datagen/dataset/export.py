"""Dataset export: save images, metadata, and manifests to disk.

All file I/O for the dataset construction pipeline lives here (Rule 4).
Processing modules never open files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from nina_thermal.datagen import config


# ── Image and metadata persistence ───────────────────────────────────────────


def save_patch_image(img_uint8: np.ndarray, path: Path) -> None:
    """Save a (H, W) uint8 grayscale array as a PNG file.

    No colour conversion is performed; the image is written as a
    single-channel grayscale PNG.

    Args:
        img_uint8: uint8 (H, W) grayscale image.
        path: Destination file path (parent directory must exist).
    """
    from PIL import Image  # type: ignore

    Image.fromarray(img_uint8, mode="L").save(path)


def save_patch_metadata(metadata_dict: Dict[str, Any], path: Path) -> None:
    """Serialise *metadata_dict* to a JSON file with 2-space indentation.

    Args:
        metadata_dict: Arbitrary JSON-serialisable dict.
        path: Destination file path (parent directory must exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata_dict, fh, indent=2)


def build_image_metadata(
    granule_id: str,
    patch_index: int,
    label: int,
    surface_type: str,
    agc_mode: str,
    profile: str,
    t_min: float,
    t_max: float,
    agc_range_k: float,
    nan_fraction: float,
    centroid_lat: float,
    centroid_lon: float,
    split: str,
    seed: int,
) -> Dict[str, Any]:
    """Assemble the complete per-image metadata dict.

    Includes provenance chain, flight-config flag, and script version
    so that every image in the dataset is fully reproducible.

    Args:
        granule_id: Sentinel-3 granule identifier string.
        patch_index: Zero-based index of the patch within the granule.
        label: Fire label (0 or 1).
        surface_type: One of ``"fire"``, ``"warm_surface"``, ``"nominal"``.
        agc_mode: AGC mode string (``"fixed"``, ``"global"``, or ``"auto"``).
        profile: Degradation profile key (``"A"``, ``"B"``, or ``"C"``).
        t_min: AGC lower temperature bound [K].
        t_max: AGC upper temperature bound [K].
        agc_range_k: AGC temperature range (``t_max - t_min``) [K].
        nan_fraction: Fraction of NaN pixels in the original BT patch.
        centroid_lat: Patch centroid latitude [degrees].
        centroid_lon: Patch centroid longitude [degrees].
        split: Dataset split assignment (``"train"``, ``"val"``, ``"test"``).
        seed: Pipeline RNG seed used to produce this image.

    Returns:
        Dict with all metadata fields.  The ``"provenance"`` list
        documents the transformation chain from raw BT to final image.
        ``"flight_config"`` is ``True`` when *agc_mode* matches
        ``config.FLIGHT_AGC_MODE``.

    Example::

        >>> m = build_image_metadata(
        ...     granule_id="S3A_20230601T120000",
        ...     patch_index=7,
        ...     label=1,
        ...     surface_type="fire",
        ...     agc_mode="fixed",
        ...     profile="A",
        ...     t_min=258.15,
        ...     t_max=358.15,
        ...     agc_range_k=100.0,
        ...     nan_fraction=0.02,
        ...     centroid_lat=-15.3,
        ...     centroid_lon=-62.7,
        ...     split="train",
        ...     seed=42,
        ... )
        >>> sorted(m.keys())  # doctest: +NORMALIZE_WHITESPACE
        ['agc_mode', 'agc_range_k', 'centroid_lat', 'centroid_lon', 'flight_config', 'granule_id', 'label', 'nan_fraction', 'patch_index', 'profile', 'provenance', 'script_version', 'seed', 'split', 'surface_type', 't_max', 't_min']
        >>> m["flight_config"]
        True
        >>> m["script_version"] == "1.0.0"
        True
        >>> m["provenance"][0]
        'SLSTR L1B NetCDF'
    """
    provenance = [
        "SLSTR L1B NetCDF",
        "cloud+exception bitmask",
        "broadband spectral correction",
        f"radiance-consistent spatial aggregation (factor={config.SLSTR_AGG_FACTOR})",
        f"AGC normalisation ({agc_mode}, {t_min:.2f}–{t_max:.2f} K)",
        f"sensor simulation profile {profile}",
        "agc_to_uint8",
    ]

    return {
        "granule_id": granule_id,
        "patch_index": patch_index,
        "label": label,
        "surface_type": surface_type,
        "agc_mode": agc_mode,
        "profile": profile,
        "t_min": t_min,
        "t_max": t_max,
        "agc_range_k": agc_range_k,
        "nan_fraction": nan_fraction,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "split": split,
        "seed": seed,
        "flight_config": agc_mode == config.FLIGHT_AGC_MODE,
        "script_version": config.SCRIPT_VERSION,
        "provenance": provenance,
    }


def save_label_manifest(records: List[Dict[str, Any]], path: Path) -> None:
    """Save the label manifest as a JSON array.

    Each element must contain at minimum:
    ``"patch"``, ``"label"``, ``"granule_id"``, ``"surface_type"``,
    ``"split"``.

    Args:
        records: List of manifest record dicts.
        path: Destination file path (parent directory must exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
