"""Dataset balancing via clear-sky undersampling.

Applied independently within each split (train / val / test) AFTER
scene-level splitting.  Cloud patches are never removed; clear patches are
undersampled to reach the target cloud ratio (default 50/50).
"""

import logging
from typing import Dict, List

import numpy as np

from nina_cloud.datagen import config

logger = logging.getLogger(__name__)


def build_balanced_dataset(
    records: List[Dict],
    cloud_ratio: float = None,
    seed: int = None,
) -> List[Dict]:
    """Undersample clear records to approximate *cloud_ratio*.

    Cloud patches are never removed.  Clear patches are randomly
    undersampled to reach::

        n_clear_target = round(n_cloud * (1 - cloud_ratio) / cloud_ratio)

    capped at the available clear supply.

    Ambiguous records (label == -1) are silently excluded before balancing.

    Args:
        records: List of dicts, each containing at minimum:
            ``"label"`` (int 0/1/-1), ``"scene_id"`` (str),
            ``"tile_idx"`` (int).
        cloud_ratio: Target cloud fraction in [0, 1].
            Defaults to ``config.CLOUD_RATIO`` (0.50).
        seed: RNG seed for reproducible undersampling.
            Defaults to ``config.DEFAULT_SEED``.

    Returns:
        Balanced list of record dicts (cloud + undersampled clear),
        in random order.  Ambiguous records are excluded.
    """
    if cloud_ratio is None:
        cloud_ratio = config.CLOUD_RATIO
    if seed is None:
        seed = config.DEFAULT_SEED

    cloud   = [r for r in records if r["label"] == 1]
    clear   = [r for r in records if r["label"] == 0]
    n_ambig = sum(1 for r in records if r["label"] == -1)

    n_cloud = len(cloud)
    n_clear = len(clear)

    if n_cloud == 0:
        n_clear_target = n_clear
    else:
        n_clear_target = min(
            round(n_cloud * (1.0 - cloud_ratio) / cloud_ratio),
            n_clear,
        )

    rng = np.random.default_rng(seed)
    if n_clear_target < n_clear:
        idx = rng.choice(n_clear, size=n_clear_target, replace=False)
        clear_selected = [clear[i] for i in idx]
    else:
        clear_selected = list(clear)

    total = n_cloud + len(clear_selected)
    achieved = (n_cloud / total) if total > 0 else 0.0

    logger.info(
        "balance: cloud=%d  clear_orig=%d  clear_target=%d  "
        "clear_removed=%d  ambiguous_excluded=%d  achieved_cloud_frac=%.3f",
        n_cloud, n_clear, n_clear_target,
        n_clear - len(clear_selected), n_ambig, achieved,
    )

    combined = cloud + clear_selected
    indices = rng.permutation(len(combined))
    return [combined[i] for i in indices]
