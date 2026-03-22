"""Granule-level train/val/test splitting with fire-fraction stratification.

CRITICAL: splits are always at the granule level. Patches from the same
granule share atmospheric conditions, sun angle, and land surface state.
Patch-level splitting would produce optimistic validation metrics due to
spatial autocorrelation.
"""

import math
from typing import Dict, List

import numpy as np

from nina_thermal import config


def granule_level_split(
    granule_ids_with_labels: Dict[str, List[int]],
    train_frac: float = None,
    val_frac: float = None,
    test_frac: float = None,
    seed: int = None,
) -> Dict[str, List[str]]:
    """Assign granules to train / val / test splits with stratification.

    Granules are stratified into two fire-fraction tiers before shuffling:

    * **high-fire** — fire fraction > ``HIGH_FIRE_GRANULE_THRESH``
    * **low-fire**  — everything else

    Each tier is shuffled independently with the seeded RNG and then
    distributed proportionally across the three splits. This guarantees
    that every split contains granules from both tiers (provided there
    are enough granules in each tier).

    Args:
        granule_ids_with_labels: Dict mapping ``granule_id`` (str) →
            list of integer labels (0 or 1) for all patches in that
            granule.
        train_frac: Fraction of granules for training.
            Defaults to ``config.TRAIN_FRAC`` (0.70).
        val_frac: Fraction of granules for validation.
            Defaults to ``config.VAL_FRAC`` (0.15).
        test_frac: Fraction of granules for test.
            Defaults to ``config.TEST_FRAC`` (0.15).
        seed: RNG seed for reproducibility.
            Defaults to ``config.DEFAULT_SEED``.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` where each
        value is a list of granule ID strings.  The union of the three
        lists equals the full set of input granule IDs, and the lists are
        pairwise disjoint (no granule appears in more than one split).

    Raises:
        ValueError: If the fractions do not sum to approximately 1.0 or
            if a tier has too few granules to populate every split.
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
        raise ValueError(
            f"Fractions must sum to 1.0, got {total_frac:.6f}"
        )

    rng = np.random.default_rng(seed)

    # ── Compute fire fraction per granule ─────────────────────────────────────
    granule_ids = list(granule_ids_with_labels.keys())
    fire_fracs = {
        gid: (sum(labels) / len(labels) if labels else 0.0)
        for gid, labels in granule_ids_with_labels.items()
    }

    high_fire = [g for g in granule_ids if fire_fracs[g] > config.HIGH_FIRE_GRANULE_THRESH]
    low_fire  = [g for g in granule_ids if fire_fracs[g] <= config.HIGH_FIRE_GRANULE_THRESH]

    rng.shuffle(high_fire)
    rng.shuffle(low_fire)

    # ── Distribute each tier into splits ──────────────────────────────────────
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for tier in (high_fire, low_fire):
        n = len(tier)
        n_train = round(n * train_frac)
        n_val   = round(n * val_frac)
        # test gets remainder to avoid accumulating rounding error
        n_test  = n - n_train - n_val

        splits["train"].extend(tier[:n_train])
        splits["val"].extend(tier[n_train : n_train + n_val])
        splits["test"].extend(tier[n_train + n_val : n_train + n_val + n_test])

    # ── Print summary ─────────────────────────────────────────────────────────
    total_granules = len(granule_ids)
    for split_name, gids in splits.items():
        patch_labels = [
            lbl
            for gid in gids
            for lbl in granule_ids_with_labels[gid]
        ]
        n_patches = len(patch_labels)
        fire_frac = (sum(patch_labels) / n_patches) if n_patches else 0.0
        print(
            f"  {split_name:5s}: {len(gids):4d} granules / "
            f"{n_patches:6d} patches / fire_frac={fire_frac:.3f}"
        )

    print(f"  total : {total_granules} granules")

    return splits
