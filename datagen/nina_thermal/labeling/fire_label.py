"""Fire labeling functions for thermal patch classification."""

import numpy as np

from nina_thermal import config


def compute_patch_label(
    f2_bt_patch: np.ndarray,
    delta_t_patch: np.ndarray,
    fire_delta_t_threshold_k: float = None,
    fire_f2_abs_threshold_k: float = None,
    min_fire_fraction: float = None,
) -> int:
    """Return 1 if the patch contains enough fire pixels, else 0.

    A pixel counts as fire only when ALL three conditions hold:
      1. ``delta_T > fire_delta_t_threshold_k``
      2. ``F2_BT  > fire_f2_abs_threshold_k``
      3. The pixel is finite in both arrays (``np.isfinite``)

    The dual criterion prevents warm-but-uniform surfaces (hot desert,
    bare rock) from triggering fire labels. The fraction threshold allows
    sub-pixel fires at the 3 563 m GSD to produce positive labels.

    Args:
        f2_bt_patch: float32 (H, W) brightness temperature [K].
        delta_t_patch: float32 (H, W) F2_BT − S8_BT [K].
        fire_delta_t_threshold_k: Minimum delta-T for a fire pixel.
            Defaults to ``config.FIRE_DELTA_T_THRESHOLD_K``.
        fire_f2_abs_threshold_k: Minimum absolute F2_BT for a fire pixel.
            Defaults to ``config.FIRE_F2_ABS_THRESHOLD_K``.
        min_fire_fraction: Minimum fraction of fire pixels required for
            label 1. Defaults to ``config.MIN_FIRE_FRACTION``.

    Returns:
        ``1`` if the fire-pixel fraction ≥ *min_fire_fraction*, else ``0``.
    """
    if fire_delta_t_threshold_k is None:
        fire_delta_t_threshold_k = config.FIRE_DELTA_T_THRESHOLD_K
    if fire_f2_abs_threshold_k is None:
        fire_f2_abs_threshold_k = config.FIRE_F2_ABS_THRESHOLD_K
    if min_fire_fraction is None:
        min_fire_fraction = config.MIN_FIRE_FRACTION

    finite_mask = np.isfinite(f2_bt_patch) & np.isfinite(delta_t_patch)
    fire_mask = (
        finite_mask
        & (delta_t_patch > fire_delta_t_threshold_k)
        & (f2_bt_patch > fire_f2_abs_threshold_k)
    )

    total = f2_bt_patch.size
    if total == 0:
        return 0

    fire_fraction = fire_mask.sum() / total
    return 1 if fire_fraction >= min_fire_fraction else 0


def classify_patch_surface_type(
    f2_bt_patch: np.ndarray,
    delta_t_patch: np.ndarray,
    label: int,
) -> str:
    """Classify a patch's surface type for dataset reporting.

    Returns one of three strings:

    * ``"fire"``         — *label* == 1
    * ``"warm_surface"`` — *label* == 0 and ``F2_BT.max() > WARM_SURFACE_THRESH_K``
    * ``"nominal"``      — *label* == 0 and no anomalously warm pixels

    Warm-surface patches are the primary source of false-positive risk and
    must be tracked separately in the dataset report (paper methodology
    section).

    Args:
        f2_bt_patch: float32 (H, W) brightness temperature [K].
        delta_t_patch: float32 (H, W) F2_BT − S8_BT [K]. (Unused
            directly, kept for API symmetry with ``compute_patch_label``.)
        label: Integer label (0 or 1) from ``compute_patch_label``.

    Returns:
        ``"fire"``, ``"warm_surface"``, or ``"nominal"``.
    """
    if label == 1:
        return "fire"

    finite_vals = f2_bt_patch[np.isfinite(f2_bt_patch)]
    if finite_vals.size > 0 and finite_vals.max() > config.WARM_SURFACE_THRESH_K:
        return "warm_surface"

    return "nominal"
