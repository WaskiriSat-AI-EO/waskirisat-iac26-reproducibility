"""Dataset export: save images, metadata, and manifests to disk.

All file I/O for the dataset construction pipeline lives here (Rule 4).
Processing modules never open files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from nina_cloud.datagen import config


def save_tile_image(img_uint8: np.ndarray, path: Path) -> None:
    """Save a (H, W, 3) uint8 RGB array as a PNG file.

    Args:
        img_uint8: uint8 (H, W, 3) RGB image.
        path: Destination file path (parent directory must exist).
    """
    from PIL import Image  # type: ignore

    Image.fromarray(img_uint8, mode="RGB").save(path)


def save_tile_metadata(metadata_dict: Dict[str, Any], path: Path) -> None:
    """Serialise *metadata_dict* to a JSON file with 2-space indentation.

    Args:
        metadata_dict: Arbitrary JSON-serialisable dict.
        path: Destination file path (parent directory must exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata_dict, fh, indent=2)


def build_image_metadata(
    scene_id: str,
    tile_index: int,
    label: int,
    tile_type: str,
    cloud_frac: float,
    nodata_frac: float,
    profile: str,
    origin_row: int,
    origin_col: int,
    split: str,
    seed: int,
) -> Dict[str, Any]:
    """Assemble the complete per-image metadata dict for a nina_cloud tile.

    Provides a full provenance chain from S2 scene ID to the final training
    PNG image so that every image in the dataset is fully traceable.

    Args:
        scene_id: Sentinel-2 scene identifier string.
        tile_index: Zero-based index of the tile within the scene.
        label: Cloud label (0 = clear, 1 = cloud, -1 = ambiguous).
        tile_type: Human-readable tile type: "cloud", "clear", or "ambiguous".
        cloud_frac: Cloud fraction in the SCL tile.
        nodata_frac: Nodata fraction in the SCL tile.
        profile: Degradation profile key ("A", "B", or "C").
        origin_row: Top-left row in S2 10m pixels.
        origin_col: Top-left column in S2 10m pixels.
        split: Dataset split assignment ("train", "val", "test").
        seed: Pipeline RNG seed used to produce this image.

    Returns:
        Dict with all metadata fields.
    """
    provenance = [
        "S2 L2A .SAFE (BOA reflectance + SCL)",
        "B04/B03/B02 DN → float32 [0,1] (scale 10000)",
        "tile_scene_to_patches (1280×800 S2 px = 12.8km × 8.0km)",
        f"sensor simulation profile {profile} "
        f"(PSF σ≈{config.PAG7936_PSF_SIGMA_S2_PX:.1f}px → shot/read noise → uint8 → FPN → JPEG)",
    ]

    return {
        "scene_id":    scene_id,
        "tile_index":  tile_index,
        "label":       label,
        "tile_type":   tile_type,
        "cloud_frac":  round(cloud_frac, 4),
        "nodata_frac": round(nodata_frac, 4),
        "profile":     profile,
        "origin_row":  origin_row,
        "origin_col":  origin_col,
        "tile_width":  config.TILE_WIDTH_PX,
        "tile_height": config.TILE_HEIGHT_PX,
        "s2_gsd_m":    config.S2_GSD_M,
        "pag7936_gsd_m": config.PAG7936_GSD_M,
        "split":       split,
        "seed":        seed,
        "script_version": config.SCRIPT_VERSION,
        "provenance":  provenance,
    }


def save_label_manifest(records: List[Dict[str, Any]], path: Path) -> None:
    """Save the label manifest as a JSON array.

    Args:
        records: List of manifest record dicts.
        path: Destination file path (parent directory must exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
