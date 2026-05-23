"""Compute a pixel-level boolean mask for an ROI polygon against SLSTR lat/lon grids."""

import json
import logging
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape

logger = logging.getLogger(__name__)


def load_roi_polygon(geojson_path: Path):
    """Load the first feature from a GeoJSON file and return a Shapely geometry."""
    with open(geojson_path) as f:
        gj = json.load(f)

    if gj["type"] == "FeatureCollection":
        geom_dict = gj["features"][0]["geometry"]
    elif gj["type"] == "Feature":
        geom_dict = gj["geometry"]
    else:
        geom_dict = gj

    return shape(geom_dict)


def compute_roi_pixel_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    roi_polygon,
) -> np.ndarray:
    """Return a bool (H, W) mask: True where the pixel centre is inside roi_polygon.

    Uses a bounding-box pre-filter then shapely 2.x vectorised contains_xy for
    speed.  For a full SLSTR granule (~2400 × 3000 pixels) this runs in < 1 s.
    """
    H, W = lat_grid.shape
    flat_lon = lon_grid.ravel().astype(np.float64)
    flat_lat = lat_grid.ravel().astype(np.float64)

    minx, miny, maxx, maxy = roi_polygon.bounds
    in_bbox = (
        (flat_lat >= miny) & (flat_lat <= maxy)
        & (flat_lon >= minx) & (flat_lon <= maxx)
    )

    mask_flat = np.zeros(H * W, dtype=bool)
    if in_bbox.any():
        mask_flat[in_bbox] = shapely.contains_xy(
            roi_polygon, flat_lon[in_bbox], flat_lat[in_bbox]
        )

    n_inside = int(mask_flat.sum())
    logger.info(
        "  ROI mask: %d / %d pixels inside polygon (%.1f%%)",
        n_inside, H * W, 100.0 * n_inside / (H * W),
    )
    return mask_flat.reshape(H, W)
