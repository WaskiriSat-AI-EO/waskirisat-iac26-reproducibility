"""Cloud / clear / ambiguous tile labeling from S2 SCL pixel masks."""

from typing import FrozenSet, Optional, Tuple

import numpy as np

from nina_cloud import config


def compute_tile_cloud_label(
    scl_patch: np.ndarray,
    cloud_classes: Optional[FrozenSet[int]] = None,
    invalid_classes: Optional[FrozenSet[int]] = None,
    high_thresh: Optional[float] = None,
    low_thresh: Optional[float] = None,
) -> Tuple[int, float]:
    """Compute a binary cloud label from an SCL tile.

    Cloud fraction is defined as:

        cloud_frac = #{pixels in cloud_classes}
                     ─────────────────────────────────────────────
                     #{pixels NOT in invalid_classes}

    Label mapping:
      cloud_frac > high_thresh  →  1  (cloud)
      cloud_frac < low_thresh   →  0  (clear)
      otherwise                 → -1  (ambiguous — excluded from dataset)

    The function is resolution-agnostic: pass either the 20m native SCL
    tile or the 10m upsampled version; cloud fraction is the same.

    Args:
        scl_patch: uint8 SCL array (any 2-D shape).
        cloud_classes: SCL values counted as cloud; default SCL_CLOUD_CLASSES.
        invalid_classes: SCL values excluded from the denominator;
            default SCL_INVALID_CLASSES.
        high_thresh: cloud_frac above this → label 1; default 0.70.
        low_thresh: cloud_frac below this → label 0; default 0.30.

    Returns:
        (label, cloud_frac) where label ∈ {-1, 0, 1}.
    """
    cloud_classes   = cloud_classes   or config.SCL_CLOUD_CLASSES
    invalid_classes = invalid_classes or config.SCL_INVALID_CLASSES
    high_thresh     = high_thresh     if high_thresh is not None else config.CLOUD_LABEL_HIGH_THRESH
    low_thresh      = low_thresh      if low_thresh  is not None else config.CLOUD_LABEL_LOW_THRESH

    flat = scl_patch.ravel()

    valid_mask = ~np.isin(flat, list(invalid_classes))
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return -1, 0.0

    n_cloud = int(np.isin(flat[valid_mask], list(cloud_classes)).sum())
    cloud_frac = n_cloud / n_valid

    if cloud_frac > high_thresh:
        label = 1
    elif cloud_frac < low_thresh:
        label = 0
    else:
        label = -1

    return label, cloud_frac


def classify_tile_type(label: int, cloud_frac: float) -> str:
    """Return a human-readable tile type string.

    Args:
        label: Integer label from compute_tile_cloud_label.
        cloud_frac: Cloud fraction from compute_tile_cloud_label.

    Returns:
        "cloud", "clear", or "ambiguous".
    """
    if label == 1:
        return "cloud"
    if label == 0:
        return "clear"
    return "ambiguous"
