"""SCL class colour overlay for paper figures.

ISOLATED from processing/simulation/labeling — never called in the pipeline.
Only used for manual inspection and paper visualizations.
"""

from typing import Optional

import numpy as np

from nina_cloud import config


def render_scl_overlay(
    rgb_uint8: np.ndarray,
    scl_uint8: np.ndarray,
    alpha: float = 0.4,
    class_colors: Optional[dict] = None,
) -> np.ndarray:
    """Overlay SCL class colours on an RGB image.

    Args:
        rgb_uint8: uint8 (H, W, 3) RGB image.
        scl_uint8: uint8 (H, W) SCL array (must be the same spatial resolution
            as rgb_uint8 — upsample with upsample_scl_to_rgb_scale if needed).
        alpha: Blend weight for the colour overlay [0=no overlay, 1=full colour].
        class_colors: Dict mapping SCL class integer → (R, G, B) tuple.
            Defaults to config.SCL_CLASS_COLORS.

    Returns:
        uint8 (H, W, 3) blended RGB image.
    """
    if class_colors is None:
        class_colors = config.SCL_CLASS_COLORS

    color_layer = np.zeros_like(rgb_uint8)
    for cls, rgb_color in class_colors.items():
        mask = scl_uint8 == cls
        color_layer[mask] = rgb_color

    blended = (
        (1.0 - alpha) * rgb_uint8.astype(np.float32)
        + alpha * color_layer.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_cloud_mask_overlay(
    rgb_uint8: np.ndarray,
    scl_uint8: np.ndarray,
    alpha: float = 0.5,
    cloud_color: tuple = (180, 210, 255),
    cloud_classes: Optional[frozenset] = None,
) -> np.ndarray:
    """Highlight cloud pixels (SCL classes 8/9/10) on an RGB image.

    Args:
        rgb_uint8: uint8 (H, W, 3) RGB image.
        scl_uint8: uint8 (H, W) SCL array at RGB resolution.
        alpha: Blend weight for the cloud colour [0=no overlay, 1=full colour].
        cloud_color: (R, G, B) tuple for cloud pixels.
        cloud_classes: Set of cloud SCL class integers.
            Defaults to config.SCL_CLOUD_CLASSES.

    Returns:
        uint8 (H, W, 3) blended image with cloud pixels tinted.
    """
    if cloud_classes is None:
        cloud_classes = config.SCL_CLOUD_CLASSES

    cloud_mask = np.isin(scl_uint8, list(cloud_classes))
    overlay = rgb_uint8.astype(np.float32).copy()
    overlay[cloud_mask] = (
        (1.0 - alpha) * overlay[cloud_mask]
        + alpha * np.array(cloud_color, dtype=np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)
