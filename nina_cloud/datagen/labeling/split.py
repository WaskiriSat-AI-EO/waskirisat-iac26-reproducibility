"""Scene-level train/val/test splitting with cloud-fraction stratification.

CRITICAL: splits are always at the S2 scene level. Tiles from the same
scene share atmospheric conditions, sun angle, and cloud patterns.
Tile-level splitting would produce optimistic validation metrics due to
spatial autocorrelation.
"""

import logging
import math
from typing import Dict, List

import numpy as np

from nina_cloud.datagen import config

logger = logging.getLogger(__name__)


def scene_level_split(
    scene_ids_with_labels: Dict[str, List[int]],
    train_frac: float = None,
    val_frac: float = None,
    test_frac: float = None,
    seed: int = None,
) -> Dict[str, List[str]]:
    """Assign scenes to train / val / test splits with stratification.

    Scenes are stratified into two cloud-coverage tiers before shuffling:

    * **high-cloud** — scene cloud fraction > ``HIGH_CLOUD_SCENE_THRESH``
    * **low-cloud**  — everything else

    Each tier is shuffled independently with the seeded RNG and then
    distributed proportionally across the three splits. This guarantees
    that every split contains scenes from both tiers (provided there
    are enough scenes in each tier).

    Args:
        scene_ids_with_labels: Dict mapping ``scene_id`` (str) →
            list of integer labels (0 or 1) for all tiles in that scene.
            Ambiguous tiles (label -1) are excluded from the cloud-fraction
            computation but their scenes are still assigned to a split.
        train_frac: Fraction of scenes for training.
            Defaults to ``config.TRAIN_FRAC`` (0.70).
        val_frac: Fraction of scenes for validation.
            Defaults to ``config.VAL_FRAC`` (0.15).
        test_frac: Fraction of scenes for test.
            Defaults to ``config.TEST_FRAC`` (0.15).
        seed: RNG seed for reproducibility.
            Defaults to ``config.DEFAULT_SEED``.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` where each
        value is a list of scene ID strings.  The union of the three
        lists equals the full set of input scene IDs, and the lists are
        pairwise disjoint (no scene appears in more than one split).

    Raises:
        ValueError: If the fractions do not sum to approximately 1.0.
    """
    if train_frac is None:
        train_frac = config.TRAIN_FRAC
    if val_frac is None:
        val_frac = config.VAL_FRAC
    if test_frac is None:
        test_frac = config.TEST_FRAC
    if seed is None:
        seed = config.DEFAULT_SEED

    total_frac = train_frac + val_frac + test_frac
    if not math.isclose(total_frac, 1.0, abs_tol=1e-6):
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac:.6f}")

    rng = np.random.default_rng(seed)

    scene_ids = list(scene_ids_with_labels.keys())

    # Cloud fraction: use only definitive labels (0 or 1; skip -1)
    cloud_fracs: Dict[str, float] = {}
    for sid, labels in scene_ids_with_labels.items():
        definitive = [l for l in labels if l in (0, 1)]
        cloud_fracs[sid] = (sum(definitive) / len(definitive)) if definitive else 0.0

    high_cloud = [s for s in scene_ids if cloud_fracs[s] > config.HIGH_CLOUD_SCENE_THRESH]
    low_cloud  = [s for s in scene_ids if cloud_fracs[s] <= config.HIGH_CLOUD_SCENE_THRESH]

    logger.info(
        "stratification: high_cloud=%d (cloud_frac>%.2f)  low_cloud=%d  total=%d",
        len(high_cloud), config.HIGH_CLOUD_SCENE_THRESH, len(low_cloud), len(scene_ids),
    )

    rng.shuffle(high_cloud)
    rng.shuffle(low_cloud)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for tier, tier_name in zip((high_cloud, low_cloud), ("high_cloud", "low_cloud")):
        n = len(tier)
        n_train = round(n * train_frac)
        n_val   = round(n * val_frac)
        n_test  = n - n_train - n_val

        splits["train"].extend(tier[:n_train])
        splits["val"].extend(tier[n_train : n_train + n_val])
        splits["test"].extend(tier[n_train + n_val : n_train + n_val + n_test])

        logger.info("  %s → train=%d  val=%d  test=%d", tier_name, n_train, n_val, n_test)

    for split_name, sids in splits.items():
        n_tiles = sum(len(scene_ids_with_labels[s]) for s in sids)
        cloud_count = sum(
            sum(1 for l in scene_ids_with_labels[s] if l == 1) for s in sids
        )
        definitive_count = sum(
            sum(1 for l in scene_ids_with_labels[s] if l in (0, 1)) for s in sids
        )
        frac = (cloud_count / definitive_count) if definitive_count else 0.0
        logger.info(
            "  [%s] scenes=%d  tiles=%d  cloud_frac=%.3f",
            split_name, len(sids), n_tiles, frac,
        )

    return splits
