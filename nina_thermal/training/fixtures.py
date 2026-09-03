"""Synthetic dataset generator for training / quantization tests.

No real Sentinel-3 or FIRMS data exists on disk for the thermal payload, so
the training, evaluation and quantization suites run against a dataset tree
that is **schema-identical** to the real datagen output
(``run_simulation`` -> ``run_split`` -> ``run_balance``) but built from
synthetic brightness-temperature patches.

Every processing step reuses the REAL datagen functions (AGC, sensor
simulation, fire labelling, granule-level split, class balance); only the
source BT array and the acquisition metadata are synthetic, so a "fire"
patch produced here triggers
:func:`~nina_thermal.datagen.labeling.fire_label.compute_patch_label` the
same way a real one would.

Two entry points:

* :func:`generate_synthetic_dataset` — one ``04_dataset/`` tree, mirroring a
  single datagen root.
* :func:`generate_multiroot_fixture` — several overlapping roots plus a
  ``firms/`` CSV per root, purpose-built to exercise
  :func:`nina_thermal.training.dataset.load_merged_records` (duplicate-patch
  de-duplication, cross-root split reconciliation) and
  :func:`nina_thermal.training.frp_labels.apply_frp_labels` (the FRP relabel
  and its ambiguous-band drop).

Metrics computed against either fixture carry no scientific meaning — they
only prove the pipeline is wired together correctly.
"""

import csv
import json
import logging
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.datagen.dataset.balance import build_balanced_dataset
from nina_thermal.datagen.dataset.export import (
    build_image_metadata,
    save_label_manifest,
    save_patch_image,
    save_patch_metadata,
)
from nina_thermal.datagen.labeling.fire_label import (
    classify_patch_surface_type,
    compute_patch_label,
)
from nina_thermal.datagen.labeling.split import granule_level_split
from nina_thermal.datagen.processing.agc import apply_agc, compute_global_percentiles
from nina_thermal.datagen.simulation.noise import generate_lepton_fpn
from nina_thermal.datagen.simulation.sensor import simulate_lepton

logger = logging.getLogger(__name__)

# Deterministic profile -> FPN seed offset (mirrors run_simulation.py so the
# fixture reproduces its per-profile, not per-granule, FPN pattern).
_PROFILE_SEED_OFFSET = {"A": 0, "B": 1, "C": 2}

_PATCH_W, _PATCH_H = config.LEPTON_RESOLUTION  # (160, 120)
_PATCH_PX = _PATCH_H * _PATCH_W

# ── FIRMS geography for generate_multiroot_fixture ────────────────────────────
# "Peak" granules are anchored on this point; "off-season" granules sit far
# enough away (in latitude) to fall outside the FRP footprint box
# (frp_labels.HALF_HEIGHT_KM ~= 240 km ~= 2.2 deg), guaranteeing a clean
# zero-detection negative regardless of the FIRMS rows written.
_FIRMS_CLUSTER_LAT = -12.0
_FIRMS_CLUSTER_LON = -63.0
_PEAK_CENTROID = (_FIRMS_CLUSTER_LAT, _FIRMS_CLUSTER_LON)
_OFF_SEASON_CENTROID = (-19.0, -61.0)  # ~780 km south of the cluster


# ── Filename convention ──────────────────────────────────────────────────────
# TODO(PR-D): hoist these to nina_thermal.datagen.dataset.export and update
# run_simulation / run_balance / training.dataset / training.frp_labels to use
# them (single source of truth for the on-disk name).


def _patch_key(granule_id: str, patch_index: int) -> str:
    """Physical-patch key: ``"{granule_id}_p{patch_index:04d}"``."""
    return f"{granule_id}_p{patch_index:04d}"


def _image_stem(patch: str, agc_mode: str, profile: str) -> str:
    """Shared stem of a rendered variant's PNG and JSON: ``"{patch}_{agc_mode}_{profile}"``."""
    return f"{patch}_{agc_mode}_{profile}"


@dataclass(frozen=True)
class FixtureSpec:
    """Parameters controlling the synthetic dataset's composition.

    Attributes:
        n_granules: Number of synthetic granules (ignored when
            ``granule_ids`` is given).
        patches_per_granule: Patches generated per granule.
        peak_fire_ratio: Per-patch fire probability for "peak season" granules.
        off_season_fire_ratio: Per-patch fire probability for "off season"
            granules — genuine negatives.
        off_season_fraction: Fraction of granules treated as off-season
            (ignored when ``granule_ids`` is given — season then comes from
            the granule id month).
        warm_surface_ratio: Per-patch probability of a warm-but-not-fire
            surface (the primary false-positive hard negative).
        ambient_mean_k: Mean ambient brightness temperature [K].
        ambient_std_k: Ambient spatial texture amplitude [K]; kept so that
            ``ambient_mean_k + ambient_std_k`` stays below
            ``config.WARM_SURFACE_THRESH_K``.
        agc_modes: AGC modes to render (mirrors ``run_simulation --agc_modes``).
        profiles: Degradation profiles to render.
        seed: RNG seed for BT synthesis and sensor simulation.
        granule_ids: Explicit granule ids to generate. When set, the granule
            population is fixed (used by :func:`generate_multiroot_fixture` to
            create overlapping roots); centroids are then a deterministic
            function of the granule id so a shared granule is byte-identical
            across roots.
        split_seed: Seed for :func:`granule_level_split`. Defaults to ``seed``.
            :func:`generate_multiroot_fixture` varies it per root so shared
            granules get conflicting split assignments.
    """

    n_granules: int = 12
    patches_per_granule: int = 12
    peak_fire_ratio: float = 0.35
    off_season_fire_ratio: float = 0.02
    off_season_fraction: float = 0.40
    warm_surface_ratio: float = 0.20
    ambient_mean_k: float = 300.0
    ambient_std_k: float = 6.0
    agc_modes: tuple[str, ...] = ("fixed", "global")
    profiles: tuple[str, ...] = ("A", "B", "C")
    seed: int = config.DEFAULT_SEED
    granule_ids: tuple[str, ...] | None = None
    split_seed: int | None = None

    def __post_init__(self) -> None:
        if self.ambient_mean_k + self.ambient_std_k >= config.WARM_SURFACE_THRESH_K:
            raise ValueError(
                "ambient_mean_k + ambient_std_k must stay below "
                f"WARM_SURFACE_THRESH_K ({config.WARM_SURFACE_THRESH_K} K) or "
                "'nominal' patches will spuriously classify as 'warm_surface'."
            )


# ── BT patch synthesis ───────────────────────────────────────────────────────


def _smooth_field(rng: np.random.Generator, low_res: int = 8) -> np.ndarray:
    """Low-frequency random field at patch resolution, normalized to [-1, 1]."""
    coarse = rng.standard_normal((low_res, low_res)).astype(np.float32)
    reps_h = -(-_PATCH_H // low_res)
    reps_w = -(-_PATCH_W // low_res)
    up = np.repeat(np.repeat(coarse, reps_h, axis=0), reps_w, axis=1)
    up = up[:_PATCH_H, :_PATCH_W]
    peak = float(np.abs(up).max())
    return up / peak if peak > 0 else up


def _make_bt_patch(
    rng: np.random.Generator,
    surface_target: str,
    ambient_mean_k: float,
    ambient_std_k: float,
) -> np.ndarray:
    """Synthesize a ``(H, W)`` float32 BT patch [K] targeting *surface_target*.

    Pixel counts are chosen against ``config.MIN_FIRE_FRACTION`` so the
    intended category reproduces under ``compute_patch_label`` /
    ``classify_patch_surface_type``.
    """
    field_ = _smooth_field(rng) * ambient_std_k
    bt = (ambient_mean_k + field_).astype(np.float32)
    flat = bt.reshape(-1)

    if surface_target == "warm_surface":
        n_warm = max(1, int(0.4 * config.MIN_FIRE_FRACTION * _PATCH_PX))
        idx = rng.choice(_PATCH_PX, size=n_warm, replace=False)
        flat[idx] = config.WARM_SURFACE_THRESH_K + rng.uniform(1.0, 8.0, size=n_warm)
    elif surface_target == "fire":
        n_fire = max(12, int(2.0 * config.MIN_FIRE_FRACTION * _PATCH_PX))
        idx = rng.choice(_PATCH_PX, size=n_fire, replace=False)
        flat[idx] = config.FIRE_F2_ABS_THRESHOLD_K + rng.uniform(40.0, 90.0, size=n_fire)

    return flat.reshape(_PATCH_H, _PATCH_W).astype(np.float32)


# ── Granule identity ─────────────────────────────────────────────────────────


def _granule_id(index: int, off_season: bool) -> str:
    """Synthetic Sentinel-3-style granule id; the month encodes the season."""
    month = "02" if off_season else ("08" if index % 2 == 0 else "09")
    day = 1 + (index % 26)
    return f"SIM_S3A_SL_1_RBT____2023{month}{day:02d}T120000_g{index:03d}"


def _is_off_season(granule_id: str) -> bool:
    """True when the granule id's month token is outside the Aug/Sep peak."""
    token = granule_id.split("____")[1][:8]  # YYYYMMDD
    return token[4:6] not in {"08", "09"}


def granule_sensing_date(granule_id: str) -> datetime:
    """Parse the sensing date from a granule id (matches ``frp_labels._granule_date``)."""
    token = granule_id.split("____")[1][:8]
    return datetime(int(token[:4]), int(token[4:6]), int(token[6:8]), tzinfo=timezone.utc)


def _deterministic_centroid(granule_id: str) -> tuple[float, float]:
    """Stable (lat, lon) for a granule.

    Off-season granules are anchored far from the FIRMS cluster (clean
    negatives); peak granules sit on the cluster (candidate positives). A
    small id-hashed jitter keeps distinct granules distinguishable without
    crossing the footprint boundary.
    """
    base_lat, base_lon = (
        _OFF_SEASON_CENTROID if _is_off_season(granule_id) else _PEAK_CENTROID
    )
    h = zlib.crc32(granule_id.encode())
    jitter_lat = ((h & 0xFFFF) / 0xFFFF - 0.5) * 0.2  # +/- 0.1 deg (~11 km)
    jitter_lon = (((h >> 16) & 0xFFFF) / 0xFFFF - 0.5) * 0.2
    return base_lat + jitter_lat, base_lon + jitter_lon


# ── Single-root generation ───────────────────────────────────────────────────


def generate_synthetic_dataset(
    output_dir: Path, spec: FixtureSpec | None = None
) -> dict[str, Any]:
    """Write one schema-compatible synthetic ``04_dataset/`` tree.

    Reuses the real datagen functions end to end: per-patch labelling ->
    global-AGC percentile pass -> sensor simulation -> granule-level split ->
    per-split class balance.

    Args:
        output_dir: Destination ``04_dataset/`` directory (created if absent).
        spec: Fixture parameters. Defaults to ``FixtureSpec()``.

    Returns:
        Summary dict: ``{granules, patches, images, fire, warm_surface,
        nominal, splits}``.
    """
    if spec is None:
        spec = FixtureSpec()

    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    meta_dir = output_dir / "metadata"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)

    fpn_maps: dict[str, dict[str, np.ndarray]] = {}
    for prof in spec.profiles:
        prof_cfg = config.DEGRADATION_PROFILES[prof]
        fpn_maps[prof] = generate_lepton_fpn(
            column_strength=prof_cfg["fpn_column_strength"],
            pixel_strength=prof_cfg["fpn_pixel_strength"],
            seed=spec.seed + _PROFILE_SEED_OFFSET.get(prof, 0),
        )

    if spec.granule_ids is not None:
        granule_ids = list(spec.granule_ids)
    else:
        granule_ids = [
            _granule_id(g, bool(rng.random() < spec.off_season_fraction))
            for g in range(spec.n_granules)
        ]

    # ── Pass 1: BT patches + labels ─────────────────────────────────────────
    granule_labels: dict[str, list[int]] = {}
    patch_records: list[dict[str, Any]] = []
    all_bt_patches: list[np.ndarray] = []

    for granule_id in granule_ids:
        granule_labels[granule_id] = []
        off_season = _is_off_season(granule_id)
        fire_ratio = spec.off_season_fire_ratio if off_season else spec.peak_fire_ratio
        surface_p = np.array(
            [fire_ratio, spec.warm_surface_ratio, 1.0 - fire_ratio - spec.warm_surface_ratio]
        )
        if surface_p[-1] < 0:
            raise ValueError("fire_ratio + warm_surface_ratio must be <= 1.0")
        centroid_lat, centroid_lon = _deterministic_centroid(granule_id)

        for p_idx in range(spec.patches_per_granule):
            surface_target = str(
                rng.choice(["fire", "warm_surface", "nominal"], p=surface_p)
            )
            bt_patch = _make_bt_patch(
                rng, surface_target, spec.ambient_mean_k, spec.ambient_std_k
            )
            delta_t = (rng.standard_normal((_PATCH_H, _PATCH_W)) * 0.3).astype(np.float32)

            label = compute_patch_label(bt_patch, delta_t)
            surface_type = classify_patch_surface_type(bt_patch, delta_t, label)
            granule_labels[granule_id].append(label)
            all_bt_patches.append(bt_patch)

            patch_records.append(
                {
                    "granule_id": granule_id,
                    "patch_index": p_idx,
                    "bt_patch": bt_patch,
                    "label": label,
                    "surface_type": surface_type,
                    "centroid_lat": centroid_lat,
                    "centroid_lon": centroid_lon,
                }
            )

    global_t_low: float | None = None
    global_t_high: float | None = None
    if "global" in spec.agc_modes:
        global_t_low, global_t_high = compute_global_percentiles(
            all_bt_patches, seed=spec.seed
        )

    # ── Pass 2: render + save each AGC-mode x profile variant ───────────────
    label_records: list[dict[str, Any]] = []
    n_fire = n_warm = n_nominal = 0
    n_images = 0

    for rec in patch_records:
        granule_id = rec["granule_id"]
        p_idx = rec["patch_index"]
        bt_patch = rec["bt_patch"]
        label = rec["label"]
        surface_type = rec["surface_type"]

        if surface_type == "fire":
            n_fire += 1
        elif surface_type == "warm_surface":
            n_warm += 1
        else:
            n_nominal += 1

        base_key = _patch_key(granule_id, p_idx)
        label_records.append(
            {
                "patch": base_key,
                "label": label,
                "granule_id": granule_id,
                "surface_type": surface_type,
                "centroid_lat": rec["centroid_lat"],
                "centroid_lon": rec["centroid_lon"],
                "split": "",  # populated below
            }
        )

        for agc_mode in spec.agc_modes:
            if agc_mode == "fixed":
                t_min, t_max = config.FIXED_TEMP_MIN_K, config.FIXED_TEMP_MAX_K
            elif agc_mode == "global":
                assert global_t_low is not None and global_t_high is not None
                t_min, t_max = global_t_low, global_t_high
            else:  # "auto"
                finite = bt_patch[np.isfinite(bt_patch)]
                t_min = float(np.percentile(finite, 2.0))
                t_max = float(np.percentile(finite, 98.0))
                if t_max <= t_min:
                    t_max = t_min + 1.0

            agc_range_k = t_max - t_min
            bt_normalized = apply_agc(bt_patch, t_min, t_max)

            for prof in spec.profiles:
                stem = _image_stem(base_key, agc_mode, prof)
                img_u8 = simulate_lepton(
                    bt_normalized,
                    profile=prof,
                    agc_range_k=agc_range_k,
                    fpn=fpn_maps[prof],
                    rng=rng,
                )
                save_patch_image(img_u8, img_dir / f"{stem}.png")
                meta = build_image_metadata(
                    granule_id=granule_id,
                    patch_index=p_idx,
                    label=label,
                    surface_type=surface_type,
                    agc_mode=agc_mode,
                    profile=prof,
                    t_min=t_min,
                    t_max=t_max,
                    agc_range_k=agc_range_k,
                    nan_fraction=0.0,
                    centroid_lat=rec["centroid_lat"],
                    centroid_lon=rec["centroid_lon"],
                    split="",
                    seed=spec.seed,
                )
                save_patch_metadata(meta, meta_dir / f"{stem}.json")
                n_images += 1

    # ── Split (granule level, real function) ────────────────────────────────
    splits = granule_level_split(
        granule_labels, seed=spec.split_seed if spec.split_seed is not None else spec.seed
    )
    granule_to_split = {gid: name for name, gids in splits.items() for gid in gids}

    for lrec in label_records:
        lrec["split"] = granule_to_split.get(lrec["granule_id"], "train")

    for meta_file in meta_dir.glob("*.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["split"] = granule_to_split.get(meta["granule_id"], "train")
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    save_label_manifest(label_records, output_dir / "label_manifest.json")

    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_version": config.SCRIPT_VERSION,
                "seed": spec.split_seed if spec.split_seed is not None else spec.seed,
                "train": splits["train"],
                "val": splits["val"],
                "test": splits["test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── Balance each split independently (real function) ────────────────────
    all_retained: list[dict[str, Any]] = []
    for gids in splits.values():
        gid_set = set(gids)
        split_recs = [r for r in label_records if r["granule_id"] in gid_set]
        if split_recs:
            all_retained.extend(build_balanced_dataset(split_recs, seed=spec.seed))

    (output_dir / "balanced_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_version": config.SCRIPT_VERSION,
                "fire_ratio": config.FIRE_RATIO,
                "seed": spec.seed,
                "total_retained": len(all_retained),
                "total_excluded": len(label_records) - len(all_retained),
                "records": all_retained,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "granules": len(granule_ids),
        "patches": len(label_records),
        "images": n_images,
        "fire": n_fire,
        "warm_surface": n_warm,
        "nominal": n_nominal,
        "splits": {name: len(gids) for name, gids in splits.items()},
    }
    logger.info("Synthetic dataset written to %s: %s", output_dir, summary)
    return summary


# ── FIRMS CSV ────────────────────────────────────────────────────────────────


def frp_granule_class(granule_id: str) -> str:
    """Deterministic FRP class for a granule: ``"fire"``, ``"ambiguous"`` or ``"clean"``.

    A pure function of the granule id, so every root's FIRMS CSV agrees on a
    given granule's fire state (the same fire is the same fire, whichever
    acquisition window observed it). Off-season granules are always
    ``"clean"``.
    """
    if _is_off_season(granule_id):
        return "clean"
    bucket = zlib.crc32(b"frp:" + granule_id.encode()) % 100
    if bucket < 45:
        return "fire"
    if bucket < 60:
        return "ambiguous"
    return "clean"


def write_firms_csv(
    firms_dir: Path,
    granule_ids: Sequence[str],
    *,
    row_frp_mw: float = 8000.0,
    seed: int = config.DEFAULT_SEED,
) -> Path:
    """Write a ``firms/detections.csv`` sized to exercise the FRP relabel.

    Peak (non-off-season) granules fall into three classes by
    :func:`frp_granule_class`:

    * **fire** — three rows at ``row_frp_mw`` each on the granule's sensing
      date near its centroid: summed FRP ``3 * row_frp_mw`` clears the
      default 20 GW threshold. The three rows carry a small per-row position
      offset so ``pandas.drop_duplicates`` (applied by
      ``frp_labels.load_firms_detections``) keeps all three, as it would for
      three distinct 1 km FIRMS pixels.
    * **ambiguous** — a single row: a non-zero detection below threshold,
      which ``assign_frp_labels`` drops.
    * **clean negative** — no rows.

    Off-season granules never get rows and, anchored ~780 km from the
    cluster, sit outside the footprint box regardless.

    Args:
        firms_dir: ``<root>/firms`` directory (created if absent).
        granule_ids: Granule ids present in the companion dataset root.
        row_frp_mw: FRP [MW] per detection row.
        seed: Unused; kept for call-site symmetry.

    Returns:
        Path to the written CSV.
    """
    firms_dir = Path(firms_dir)
    firms_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for granule_id in granule_ids:
        cls = frp_granule_class(granule_id)
        n_rows = {"fire": 3, "ambiguous": 1, "clean": 0}[cls]
        if n_rows == 0:
            continue
        lat, lon = _deterministic_centroid(granule_id)
        date = granule_sensing_date(granule_id).strftime("%Y-%m-%d")
        for k in range(n_rows):
            rows.append(
                {
                    "latitude": round(lat + k * 0.01, 5),  # distinct 1 km pixels
                    "longitude": round(lon + k * 0.01, 5),
                    "acq_date": date,
                    "confidence": "nominal",
                    "frp": row_frp_mw,
                }
            )

    csv_path = firms_dir / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["latitude", "longitude", "acq_date", "confidence", "frp"]
        )
        writer.writeheader()
        writer.writerows(rows)
    classes = [frp_granule_class(g) for g in granule_ids]
    logger.info(
        "FIRMS CSV -> %s (%d rows; %d fire granules, %d ambiguous)",
        csv_path,
        len(rows),
        classes.count("fire"),
        classes.count("ambiguous"),
    )
    return csv_path


# ── Multi-root fixture ───────────────────────────────────────────────────────


@dataclass
class MultiRootFixture:
    """Result of :func:`generate_multiroot_fixture`.

    Attributes:
        roots: One ``<root>`` path per acquisition window (each holds
            ``04_dataset/`` and ``firms/``), in priority order.
        shared_granules: Granule ids present in more than one root.
        conflicting_granules: Shared granules whose per-root split assignment
            disagrees — the population :func:`reconcile_splits` must resolve.
        summaries: Per-root :func:`generate_synthetic_dataset` summary dicts.
    """

    roots: list[Path]
    shared_granules: list[str]
    conflicting_granules: list[str]
    summaries: list[dict[str, Any]] = field(default_factory=list)


def generate_multiroot_fixture(
    base_dir: Path,
    *,
    n_roots: int = 3,
    granules_per_root: int = 9,
    overlap: int = 4,
    patches_per_granule: int = 8,
    profiles: tuple[str, ...] = ("A", "B", "C"),
    agc_modes: tuple[str, ...] = ("fixed",),
    seed: int = config.DEFAULT_SEED,
) -> MultiRootFixture:
    """Write ``n_roots`` overlapping dataset roots plus a FIRMS CSV each.

    Consecutive roots share ``overlap`` granules. Every root runs its own
    :func:`granule_level_split` with a distinct seed over its own granule
    population, so shared granules routinely land in different splits — the
    exact condition :func:`nina_thermal.training.dataset.load_merged_records`
    reconciles (``test`` > ``val`` > ``train``) and
    :func:`~nina_thermal.training.dataset.split_samples` asserts against.

    Args:
        base_dir: Directory to create the roots under.
        n_roots: Number of acquisition-window roots.
        granules_per_root: Granules materialised per root.
        overlap: Granules shared between each consecutive pair of roots.
        patches_per_granule: Patches per granule.
        profiles: Degradation profiles to render.
        agc_modes: AGC modes to render.
        seed: Base seed.

    Returns:
        A :class:`MultiRootFixture`.
    """
    base_dir = Path(base_dir)
    step = granules_per_root - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than granules_per_root")

    total_granules = step * (n_roots - 1) + granules_per_root
    all_ids = [
        _granule_id(i, off_season=(i % 3 == 0)) for i in range(total_granules)
    ]

    roots: list[Path] = []
    summaries: list[dict[str, Any]] = []
    per_root_split: dict[str, dict[str, str]] = {}

    for r in range(n_roots):
        start = r * step
        root_ids = tuple(all_ids[start : start + granules_per_root])
        root_dir = base_dir / f"root{r}"
        spec = FixtureSpec(
            patches_per_granule=patches_per_granule,
            agc_modes=agc_modes,
            profiles=profiles,
            seed=seed + r,
            split_seed=1000 + 7 * r,  # distinct split per root -> conflicts
            granule_ids=root_ids,
        )
        summary = generate_synthetic_dataset(root_dir / "04_dataset", spec)
        write_firms_csv(root_dir / "firms", root_ids, seed=seed + r)

        split_manifest = json.loads(
            (root_dir / "04_dataset" / "split_manifest.json").read_text(encoding="utf-8")
        )
        for split_name in ("train", "val", "test"):
            for gid in split_manifest[split_name]:
                per_root_split.setdefault(gid, {})[f"root{r}"] = split_name

        roots.append(root_dir)
        summaries.append(summary)

    shared = sorted(g for g, m in per_root_split.items() if len(m) > 1)
    conflicting = sorted(
        g for g in shared if len(set(per_root_split[g].values())) > 1
    )

    logger.info(
        "Multi-root fixture: %d roots, %d shared granules, %d with conflicting splits",
        n_roots,
        len(shared),
        len(conflicting),
    )
    return MultiRootFixture(
        roots=roots,
        shared_granules=shared,
        conflicting_granules=conflicting,
        summaries=summaries,
    )


__all__ = [
    "FixtureSpec",
    "MultiRootFixture",
    "frp_granule_class",
    "generate_multiroot_fixture",
    "generate_synthetic_dataset",
    "granule_sensing_date",
    "write_firms_csv",
]
