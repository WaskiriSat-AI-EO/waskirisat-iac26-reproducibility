"""Multi-root dataset assembly for the fire/no-fire classifier.

The thermal datagen pipeline is *incremental*: each acquisition window is
materialised into its own ``04_dataset/`` directory (``data/``,
``data_202408/``, ``data_202409/``, ``data_202509/``, …).  Training needs the
union of those windows, and assembling that union naively corrupts the
experiment in two distinct ways that this module exists to prevent.

Failure mode 1 — duplicate patches
-----------------------------------
The cumulative root (``data/``) already contains every patch of some later
windows.  ``data_202408``'s 171 patches are a strict subset of ``data/``'s 576.
Concatenating the manifests would feed those patches twice per epoch, silently
re-weighting them 2:1 against the rest of the training set.  Records are
therefore de-duplicated on ``patch``, which is globally unique (it embeds the
granule id).

Failure mode 2 — contradictory split assignments
-------------------------------------------------
``run_split`` re-derives the train/val/test assignment independently per root,
seeded identically but over a *different granule population*.  A granule that
lands in ``train`` for one root can land in ``test`` for another — 19 of the 41
granules shared between ``data/`` and ``data_202408/`` disagree this way.
Taking the first assignment seen would place test granules into the training
set, and every reported metric downstream would be measured on data the model
had already fitted.

:func:`reconcile_splits` resolves this with a *conservative precedence*:
``test`` > ``val`` > ``train``.  A granule that is held out anywhere is held out
everywhere.  This can only ever shrink the training set — it can never leak an
evaluation granule into it.

Sample granularity
------------------
The unit of *labelling* is the patch; the unit of *training* is the image, and
one patch yields several images (one per AGC mode × degradation profile).  All
images derived from a patch necessarily share its split, and all patches from a
granule share the granule's split, so grouping integrity holds at every level.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from nina_thermal.datagen.dataset.export import image_filename

logger = logging.getLogger(__name__)

# ── Native Lepton geometry ────────────────────────────────────────────────────
# nina_thermal.datagen.config.LEPTON_RESOLUTION is (width, height) = (160, 120).
# Exported PNGs are already at this size, so the training transform performs no
# resampling — a resize here would either interpolate away the very sensor
# artefacts the simulation injected, or silently mask a datagen regression.
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

# Ordered least- to most-restrictive.  Higher index wins during reconciliation.
_SPLIT_PRECEDENCE = ("train", "val", "test")


# ── Root resolution ───────────────────────────────────────────────────────────


def resolve_dataset_root(root: str | Path) -> Path:
    """Return the ``04_dataset`` directory for a user-supplied root path.

    Accepts either the data root (``data/``) or the dataset directory itself
    (``data/04_dataset``) so the CLI is forgiving about which one is passed.

    Args:
        root: Path to a data root or directly to an ``04_dataset`` directory.

    Returns:
        Path to the directory containing ``balanced_manifest.json``.

    Raises:
        FileNotFoundError: If no ``balanced_manifest.json`` is found either way.
    """
    root = Path(root)
    candidates = [root, root / "04_dataset"]
    for candidate in candidates:
        if (candidate / "balanced_manifest.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"No balanced_manifest.json under {root} or {root / '04_dataset'}.  "
        "Run the datagen pipeline (stages 1–7) for this root first."
    )


# ── Merge and reconciliation ──────────────────────────────────────────────────


def reconcile_splits(assignments: dict[str, set[str]]) -> dict[str, str]:
    """Collapse per-granule split assignments to one split per granule.

    Applies the conservative precedence ``test`` > ``val`` > ``train``: if any
    root holds a granule out for evaluation, it is held out everywhere.  The
    alternative (first-seen wins) admits evaluation granules into training and
    invalidates every downstream metric.

    Args:
        assignments: Maps ``granule_id`` to the set of splits observed for it
            across all roots.

    Returns:
        Maps ``granule_id`` to its single reconciled split.

    Raises:
        ValueError: If a granule carries a split name outside the known set.
    """
    resolved: dict[str, str] = {}
    for granule_id, splits in assignments.items():
        unknown = splits - set(_SPLIT_PRECEDENCE)
        if unknown:
            raise ValueError(
                f"Granule {granule_id} carries unknown split(s) {sorted(unknown)}; "
                f"expected a subset of {list(_SPLIT_PRECEDENCE)}."
            )
        resolved[granule_id] = max(splits, key=_SPLIT_PRECEDENCE.index)
    return resolved


def load_merged_records(roots: list[str | Path]) -> tuple[list[dict], dict]:
    """Load and merge ``balanced_manifest.json`` from every root.

    De-duplicates on ``patch`` and reconciles split assignments at granule
    level.  Ambiguous records (``label == -1``) are dropped, matching the
    convention in ``nina_cloud.training.train_cloud``.

    Args:
        roots: Data roots or ``04_dataset`` directories, in priority order.
            Order affects only which root supplies a duplicated patch's image
            file — never its label or split, which are reconciled globally.

    Returns:
        ``(records, merge_stats)``.  Each record is the manifest record plus
        ``_root`` (the ``04_dataset`` Path it was read from) and a ``split``
        overwritten with the reconciled value.  ``merge_stats`` records what
        the merge had to fix, for the training report.

    Raises:
        FileNotFoundError: If a root has no manifest.
    """
    by_patch: dict[str, dict] = {}
    split_votes: dict[str, set[str]] = defaultdict(set)
    per_root: dict[str, int] = {}
    duplicate_patches = 0

    for root in roots:
        dataset_dir = resolve_dataset_root(root)
        manifest = json.loads(
            (dataset_dir / "balanced_manifest.json").read_text(encoding="utf-8")
        )
        records = [r for r in manifest["records"] if r.get("label") != -1]
        per_root[str(dataset_dir)] = len(records)

        for record in records:
            split_votes[record["granule_id"]].add(record["split"])
            if record["patch"] in by_patch:
                duplicate_patches += 1
                continue
            by_patch[record["patch"]] = {**record, "_root": dataset_dir}

    resolved = reconcile_splits(split_votes)
    reassigned = 0
    for record in by_patch.values():
        target = resolved[record["granule_id"]]
        if record["split"] != target:
            reassigned += 1
        record["split"] = target

    conflicted = sum(1 for splits in split_votes.values() if len(splits) > 1)
    merge_stats = {
        "roots": {k: v for k, v in per_root.items()},
        "records_read": sum(per_root.values()),
        "unique_patches": len(by_patch),
        "duplicate_patches_dropped": duplicate_patches,
        "granules": len(resolved),
        "granules_with_conflicting_splits": conflicted,
        "patches_reassigned_by_reconciliation": reassigned,
        "split_precedence": list(_SPLIT_PRECEDENCE),
    }

    records = sorted(by_patch.values(), key=lambda r: r["patch"])
    logger.info(
        "Merged %d records from %d root(s) → %d unique patches / %d granules "
        "(%d duplicates dropped, %d granules had conflicting splits, "
        "%d patches reassigned)",
        merge_stats["records_read"],
        len(per_root),
        len(records),
        len(resolved),
        duplicate_patches,
        conflicted,
        reassigned,
    )
    return records, merge_stats


# ── Image expansion ───────────────────────────────────────────────────────────


def expand_to_samples(
    records: list[dict],
    profiles: list[str],
    agc_modes: list[str],
) -> list[dict]:
    """Expand patch records into one training sample per exported image.

    The datagen export names images ``{patch}_{agc_mode}_{profile}.png``.  A
    sample is emitted only if its file exists on disk, so a partially exported
    root degrades to fewer samples rather than crashing mid-epoch.

    Args:
        records: Merged patch records from :func:`load_merged_records`.
        profiles: Degradation profiles to include (``"A"``/``"B"``/``"C"``).
        agc_modes: AGC modes to include (``"fixed"`` and/or ``"global"``).

    Returns:
        Sample dicts with ``path``, ``label``, ``granule_id``, ``patch``,
        ``split``, ``profile``, ``agc_mode``, and ``surface_type``.

    Raises:
        RuntimeError: If no image file resolves for any record.
    """
    samples: list[dict] = []
    missing = 0
    for record in records:
        for agc_mode in agc_modes:
            for profile in profiles:
                name = image_filename(record["patch"], agc_mode, profile)
                path = record["_root"] / "images" / name
                if not path.is_file():
                    missing += 1
                    continue
                samples.append(
                    {
                        "path": path,
                        "label": int(record["label"]),
                        "granule_id": record["granule_id"],
                        "patch": record["patch"],
                        "split": record["split"],
                        "profile": profile,
                        "agc_mode": agc_mode,
                        "surface_type": record.get("surface_type", "unknown"),
                    }
                )

    if not samples:
        raise RuntimeError(
            f"No images found for profiles={profiles} agc_modes={agc_modes}.  "
            "Check that the datagen simulation stage exported these variants."
        )
    if missing:
        logger.warning(
            "%d expected image(s) absent from disk and skipped "
            "(profiles=%s, agc_modes=%s)",
            missing,
            profiles,
            agc_modes,
        )
    return samples


def split_samples(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by split and assert the splits share no granule.

    The assertion is the payoff of :func:`reconcile_splits` — it fails loudly
    if a future change to the merge logic reintroduces granule-level leakage.

    Args:
        samples: Expanded samples from :func:`expand_to_samples`.

    Returns:
        Maps each split name to its sample list.

    Raises:
        RuntimeError: If any granule appears in more than one split.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["split"]].append(sample)

    granules = {
        split: {s["granule_id"] for s in split_samples}
        for split, split_samples in grouped.items()
    }
    for i, left in enumerate(_SPLIT_PRECEDENCE):
        for right in _SPLIT_PRECEDENCE[i + 1 :]:
            overlap = granules.get(left, set()) & granules.get(right, set())
            if overlap:
                raise RuntimeError(
                    f"Granule leakage: {len(overlap)} granule(s) appear in both "
                    f"'{left}' and '{right}' (e.g. {sorted(overlap)[0]}).  "
                    "Split reconciliation is broken."
                )
    return dict(grouped)


def describe_samples(samples: list[dict]) -> dict:
    """Summarise a sample list for logging and the training report.

    Args:
        samples: Samples for one split.

    Returns:
        Counts of images, patches, granules, labels, profiles and surface types.
    """
    return {
        "images": len(samples),
        "patches": len({s["patch"] for s in samples}),
        "granules": len({s["granule_id"] for s in samples}),
        "fire": sum(1 for s in samples if s["label"] == 1),
        "no_fire": sum(1 for s in samples if s["label"] == 0),
        "by_profile": dict(Counter(s["profile"] for s in samples)),
        "by_surface_type": dict(Counter(s["surface_type"] for s in samples)),
    }


# ── Torch dataset ─────────────────────────────────────────────────────────────


class ThermalDataset(Dataset):
    """Fire/no-fire images at native Lepton resolution.

    The exported PNGs are single-channel (mode ``L``) 8-bit AGC products: under
    the flight ``fixed`` AGC each level is a fixed temperature step of
    ``FIXED_TEMP_RANGE_K / 255 ≈ 0.392 K`` over 258.15–358.15 K.  The transform
    replicates that single channel to three so ImageNet-pretrained weights load
    unchanged; the replication is exact, so no radiometric information is added
    or lost.

    Args:
        samples: Sample dicts from :func:`expand_to_samples`.
        transform: Torchvision transform applied to the PIL image.
    """

    def __init__(self, samples: list[dict], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample["path"]) as img:
            image = self.transform(img.convert("L"))
        return image, torch.tensor(float(sample["label"]), dtype=torch.float32)
