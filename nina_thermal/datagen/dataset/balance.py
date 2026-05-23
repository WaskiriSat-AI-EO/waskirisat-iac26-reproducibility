"""Dataset balancing via no-fire undersampling.

Applied independently within each split (train / val / test) AFTER
granule-level splitting.  Fire patches are never removed.
"""

import logging
import math
from typing import List, Dict

import numpy as np

from nina_thermal.datagen import config

logger = logging.getLogger(__name__)


def build_balanced_dataset(
    records: List[Dict],
    fire_ratio: float = None,
    seed: int = None,
) -> List[Dict]:
    """Undersample no-fire records to approximate *fire_ratio*.

    Fire patches are never removed.  No-fire patches are randomly
    undersampled to reach::

        n_nofire_target = round(n_fire * (1 - fire_ratio) / fire_ratio)

    capped at the available no-fire supply.

    Args:
        records: List of dicts, each containing at minimum:
            ``"label"`` (int 0/1), ``"granule_id"`` (str),
            ``"patch"`` (str).
        fire_ratio: Target fire fraction in [0, 1].
            Defaults to ``config.FIRE_RATIO`` (0.20).
        seed: RNG seed for reproducible undersampling.
            Defaults to ``config.DEFAULT_SEED``.

    Returns:
        Balanced list of record dicts (fire + undersampled no-fire),
        in random order.

    Prints:
        Original counts, target counts, and achieved fire fraction.
    """
    if fire_ratio is None:
        fire_ratio = config.FIRE_RATIO
    if seed is None:
        seed = config.DEFAULT_SEED

    fire = [r for r in records if r["label"] == 1]
    nofire = [r for r in records if r["label"] == 0]

    n_fire = len(fire)
    n_nofire = len(nofire)

    if n_fire == 0:
        n_nofire_target = n_nofire
    else:
        n_nofire_target = min(
            round(n_fire * (1.0 - fire_ratio) / fire_ratio),
            n_nofire,
        )

    rng = np.random.default_rng(seed)
    if n_nofire_target < n_nofire:
        idx = rng.choice(n_nofire, size=n_nofire_target, replace=False)
        nofire_selected = [nofire[i] for i in idx]
    else:
        nofire_selected = list(nofire)

    total = n_fire + len(nofire_selected)
    achieved = n_fire / total if total > 0 else 0.0

    logger.info(
        "  balance: fire=%d  nofire_orig=%d  nofire_target=%d  "
        "nofire_removed=%d  achieved_fire_frac=%.3f",
        n_fire,
        n_nofire,
        n_nofire_target,
        n_nofire - len(nofire_selected),
        achieved,
    )

    combined = fire + nofire_selected
    # Shuffle so callers don't see label-grouped order
    indices = rng.permutation(len(combined))
    return [combined[i] for i in indices]
