"""Feasibility gate: BT-statistics + logistic-regression baseline.

Answers, BEFORE any CNN training time is spent, whether the fire label is
detectable at all from the rendered imagery. Torch-free (numpy + scikit-learn),
so it always runs in CI.

Why this exists
---------------
A single 160x120 patch covers a ~640x480 km footprint at the 4 km synthetic
GSD. A FIRMS-scale fire is ~0.1% of one pixel, so after aggregation it raises
the patch by a few tenths of a Kelvin against a 40-60 level natural scene
spread under the fixed AGC (0.392 K / level). This module measures whether that
signal survives, using the exact information a CNN would see (the rendered
uint8 image, inverted back to Kelvin).

If the CV'd AUC is ~0.5 across every feature and fold, the honest conclusion is
"not detectable at this GSD and AGC configuration" — a legitimate, publishable
negative result. ``frp_labels.py`` exists because that is exactly what the
**datagen** label produced; run this module with ``--label datagen`` to
reproduce that finding and with ``--label frp`` to show the FRP reformulation
recovers a learnable signal.

The ``test`` split is never read here — this is a pre-training diagnostic, so
the held-out granules stay untouched for the final CNN evaluation.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nina_thermal.datagen import config
from nina_thermal.training.dataset import load_merged_records
from nina_thermal.training.frp_labels import (
    DEFAULT_FRP_THRESHOLD_MW,
    LABEL_AMBIGUOUS,
    apply_frp_labels,
)

logger = logging.getLogger(__name__)

# Grouped-CV folds for the pre-CNN baseline. Small because the expected granule
# count (~50) limits how many folds can carry both classes.
DEFAULT_CV_FOLDS: int = 5

# Column order is significant: build_feature_matrix returns columns in this
# order and per_feature_auc keys off these names.
FEATURE_NAMES: tuple[str, ...] = (
    "bt_mean",
    "bt_std",
    "bt_p50",
    "bt_p90",
    "bt_p99",
    "bt_p999",
    "bt_max",
    "bt_p99_minus_p50",
    "bt_max_minus_p90",
    "warm_frac",
    "hot_frac",
    "grad_mean",
    "grad_p99",
    "top_pixel_excess",
)


def _image_stem(patch: str, agc_mode: str, profile: str) -> str:
    # TODO(PR-D): shared with training.fixtures — hoist to datagen.dataset.export.
    return f"{patch}_{agc_mode}_{profile}"


# ── BT recovery and per-patch statistics ────────────────────────────────────


def bt_from_uint8(img_u8: np.ndarray, *, t_min_k: float, t_max_k: float) -> np.ndarray:
    """Invert the AGC linear mapping: recover BT [K] from a rendered uint8 image.

    Inverse of ``apply_agc`` up to the +/-0.5-level quantization. It does NOT
    undo the sensor chain (noise, blur, FPN) — it recovers exactly the
    information a CNN trained on the same PNG would see, in physical units.
    """
    normalized = img_u8.astype(np.float32) / 255.0
    return (t_min_k + normalized * (t_max_k - t_min_k)).astype(np.float32)


def patch_statistics(
    img_u8: np.ndarray,
    *,
    t_min_k: float,
    t_max_k: float,
    warm_thresh_k: float = config.WARM_SURFACE_THRESH_K,
    hot_thresh_k: float = config.FIRE_F2_ABS_THRESHOLD_K,
) -> dict[str, float]:
    """Feature vector (keyed by ``FEATURE_NAMES``) for one patch.

    ``top_pixel_excess`` (``bt_max - bt_p99``) is the most physically direct
    fire discriminant at this resolution: a sub-pixel hotspot is a small-area
    excess over the bulk scene, not a shift in the bulk distribution.
    """
    bt = bt_from_uint8(img_u8, t_min_k=t_min_k, t_max_k=t_max_k)
    # float64: summing ~19k float32 pixels (each ~0.39 K quantized) accumulates
    # enough rounding error to register as spurious texture on a flat patch.
    flat = bt.reshape(-1).astype(np.float64)

    p50, p90, p99, p999 = np.percentile(flat, [50.0, 90.0, 99.0, 99.9])
    bt_max = float(flat.max())

    bt64 = flat.reshape(bt.shape)
    grad = np.concatenate(
        [np.abs(np.diff(bt64, axis=0)).ravel(), np.abs(np.diff(bt64, axis=1)).ravel()]
    )

    return {
        "bt_mean": float(flat.mean()),
        "bt_std": float(flat.std()),
        "bt_p50": float(p50),
        "bt_p90": float(p90),
        "bt_p99": float(p99),
        "bt_p999": float(p999),
        "bt_max": bt_max,
        "bt_p99_minus_p50": float(p99 - p50),
        "bt_max_minus_p90": float(bt_max - p90),
        "warm_frac": float((flat > warm_thresh_k).mean()),
        "hot_frac": float((flat > hot_thresh_k).mean()),
        "grad_mean": float(grad.mean()),
        "grad_p99": float(np.percentile(grad, 99.0)),
        "top_pixel_excess": float(bt_max - p99),
    }


def build_feature_matrix(
    records: list[dict[str, Any]],
    *,
    agc_mode: str,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build ``(X, y, granule_ids)`` from merged records.

    Reads one rendered variant per patch. Each record must carry ``_root``
    (the ``04_dataset`` Path, as :func:`load_merged_records` attaches),
    ``patch``, ``label`` and ``granule_id``. AGC bounds come from each image's
    own metadata JSON, falling back to the fixed-AGC config.

    Returns:
        ``X`` ``(n, len(FEATURE_NAMES))`` float64, ``y`` ``(n,)`` int,
        ``granule_ids`` parallel list for grouped CV.
    """
    from PIL import Image

    rows: list[list[float]] = []
    labels: list[int] = []
    granule_ids: list[str] = []
    n_missing = 0

    for rec in records:
        root = Path(rec["_root"])
        stem = _image_stem(rec["patch"], agc_mode, profile)
        img_path = root / "images" / f"{stem}.png"
        meta_path = root / "metadata" / f"{stem}.json"
        if not img_path.is_file():
            n_missing += 1
            continue

        t_min_k, t_max_k = config.FIXED_TEMP_MIN_K, config.FIXED_TEMP_MAX_K
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            t_min_k = float(meta.get("t_min", t_min_k))
            t_max_k = float(meta.get("t_max", t_max_k))
        else:
            logger.warning("Metadata missing for %s — using config AGC bounds.", stem)

        with Image.open(img_path) as im:
            img_u8 = np.asarray(im.convert("L"), dtype=np.uint8)

        stats = patch_statistics(img_u8, t_min_k=t_min_k, t_max_k=t_max_k)
        rows.append([stats[name] for name in FEATURE_NAMES])
        labels.append(int(rec["label"]))
        granule_ids.append(rec["granule_id"])

    if n_missing:
        logger.warning(
            "build_feature_matrix: %d/%d records had no %s/%s image — skipped.",
            n_missing,
            len(records),
            agc_mode,
            profile,
        )

    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        granule_ids,
    )


# ── Baseline model ─────────────────────────────────────────────────────────


def fit_logistic_baseline(X: np.ndarray, y: np.ndarray, *, seed: int = config.DEFAULT_SEED):
    """Fit a standardized logistic-regression baseline.

    ``class_weight="balanced"`` compensates for label imbalance so the baseline
    is not merely rewarded for predicting the majority class.

    Returns:
        A fitted ``sklearn.pipeline.Pipeline`` (StandardScaler -> LogisticRegression).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=seed
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def grouped_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    *,
    n_splits: int = DEFAULT_CV_FOLDS,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, Any]:
    """Grouped-by-granule cross-validated AUC of the logistic baseline.

    Grouping by granule prevents leakage: patches from one granule share
    atmospheric and land-surface state, so a patch-level split biases the AUC
    optimistically. Folds are clamped to the available granule count; clamping
    is logged. A fold's AUC is ``None`` when its validation split is
    single-class.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(set(groups))
    effective_splits = max(2, min(n_splits, n_groups))
    if effective_splits < n_splits:
        logger.warning(
            "grouped_cv_auc: n_splits=%d but only %d granules — clamped to %d folds.",
            n_splits,
            n_groups,
            effective_splits,
        )

    result: dict[str, Any] = {
        "n_splits": effective_splits,
        "auc_per_fold": [],
        "auc_mean": None,
        "auc_std": None,
        "error": None,
    }
    if len(set(y.tolist())) < 2:
        result["error"] = "Only one class present overall — AUC is undefined."
        logger.warning(result["error"])
        return result

    try:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits, shuffle=True, random_state=seed
        )
        fold_aucs: list[float | None] = []
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            y_val = y[val_idx]
            if len(set(y_val.tolist())) < 2:
                fold_aucs.append(None)
                continue
            model = fit_logistic_baseline(X[train_idx], y[train_idx], seed=seed)
            scores = model.predict_proba(X[val_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y_val, scores)))

        result["auc_per_fold"] = fold_aucs
        valid = [a for a in fold_aucs if a is not None]
        if valid:
            result["auc_mean"] = float(np.mean(valid))
            result["auc_std"] = float(np.std(valid))
        else:
            result["error"] = "Every fold's validation split was single-class."
            logger.warning(result["error"])
    except ValueError as exc:
        result["error"] = str(exc)
        logger.warning("grouped_cv_auc failed: %s", exc)

    return result


def per_feature_auc(X: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """Univariate AUC of each raw feature vs the label (no CV).

    Values below 0.5 mean an anti-correlated but still informative feature —
    take ``max(auc, 1 - auc)`` for magnitude alone.
    """
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return {name: None for name in FEATURE_NAMES}
    return {
        name: float(roc_auc_score(y, X[:, i])) for i, name in enumerate(FEATURE_NAMES)
    }


# ── Degeneracy diagnostics ─────────────────────────────────────────────────


def detect_degenerate_labels(
    y: np.ndarray, groups: list[str], *, min_class_fraction: float = 0.02
) -> dict[str, Any]:
    """Flag a near-constant label distribution before it wastes GPU time.

    Over a peak-season, ~640x480 km footprint the fire label may be positive
    for nearly every patch while ``config.FIRE_RATIO`` undersampling assumes
    the opposite.

    Returns:
        ``{degenerate, fire_fraction, n_granules, all_positive_granules,
        all_negative_granules, warnings}``.
    """
    n = len(y)
    fire_fraction = float(y.mean()) if n else 0.0
    warns: list[str] = []

    if n == 0:
        return {
            "degenerate": True,
            "fire_fraction": 0.0,
            "n_granules": 0,
            "all_positive_granules": 0,
            "all_negative_granules": 0,
            "warnings": ["No records available — cannot assess label distribution."],
        }

    if fire_fraction >= 1.0 - min_class_fraction:
        warns.append(
            f"fire_fraction={fire_fraction:.4f} is near-constant positive — the label "
            "may be degenerate. Acquire off-season granules and/or raise "
            "MIN_FIRE_FRACTION."
        )
    if fire_fraction <= min_class_fraction:
        warns.append(
            f"fire_fraction={fire_fraction:.4f} is near-constant negative — check "
            "FIRMS coverage and MIN_FIRE_FRACTION."
        )

    per_granule: dict[str, list[int]] = {}
    for label, gid in zip(y.tolist(), groups, strict=True):
        per_granule.setdefault(gid, []).append(label)
    all_pos = sum(1 for v in per_granule.values() if all(x == 1 for x in v))
    all_neg = sum(1 for v in per_granule.values() if all(x == 0 for x in v))
    n_granules = len(per_granule)

    if n_granules and all_pos == n_granules:
        warns.append("Every granule is 100% fire-labeled — no genuine negative granule.")
    if n_granules and all_neg == n_granules:
        warns.append("Every granule is 100% no-fire-labeled — no genuine positive granule.")

    for w in warns:
        logger.warning(w)
    return {
        "degenerate": bool(warns),
        "fire_fraction": fire_fraction,
        "n_granules": n_granules,
        "all_positive_granules": all_pos,
        "all_negative_granules": all_neg,
        "warnings": warns,
    }


def recalibrate_min_fire_fraction(
    patches_manifest_path: Path,
    *,
    target_positive_rate: float = config.FIRE_RATIO,
) -> dict[str, Any]:
    """Recommend ``MIN_FIRE_FRACTION`` from the empirical FIRMS fire-fraction distribution.

    Reads ``03_patches/patches_manifest.json`` — READ-ONLY, analysis only. Never
    on the training data path; a human decides whether to act on it before
    re-running ``run_simulation``.

    Returns:
        ``{current_min_fire_fraction, current_positive_rate,
        recommended_min_fire_fraction, target_positive_rate,
        fire_fraction_distribution}``.
    """
    manifest = json.loads(Path(patches_manifest_path).read_text(encoding="utf-8"))
    fractions = np.asarray(
        [float(p.get("fire_fraction", 0.0)) for p in manifest.get("patches", [])],
        dtype=np.float64,
    )
    if fractions.size == 0:
        raise ValueError(
            f"No patch records with 'fire_fraction' in {patches_manifest_path}"
        )

    current_positive_rate = float((fractions >= config.MIN_FIRE_FRACTION).mean())
    recommended = float(np.percentile(fractions, 100.0 * (1.0 - target_positive_rate)))
    return {
        "current_min_fire_fraction": config.MIN_FIRE_FRACTION,
        "current_positive_rate": current_positive_rate,
        "recommended_min_fire_fraction": recommended,
        "target_positive_rate": target_positive_rate,
        "fire_fraction_distribution": {
            "p50": float(np.percentile(fractions, 50.0)),
            "p90": float(np.percentile(fractions, 90.0)),
            "p99": float(np.percentile(fractions, 99.0)),
            "max": float(fractions.max()),
        },
    }


# ── Top-level report ───────────────────────────────────────────────────────


def _apply_label_scheme(
    records: list[dict[str, Any]],
    roots: list[Path],
    label_scheme: str,
    agc_mode: str,
    profile: str,
    frp_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return ``(records, frp_summary)`` with labels per *label_scheme*.

    ``"datagen"`` keeps each record's manifest label. ``"frp"`` relabels by
    summed FRP over the footprint and drops the ambiguous band.
    """
    if label_scheme == "datagen":
        return records, None

    from nina_thermal.training.dataset import expand_to_samples

    samples = expand_to_samples(records, [profile], [agc_mode])
    samples, summary = apply_frp_labels(samples, roots, frp_threshold)
    frp_label = {
        s["patch"]: int(s["label"]) for s in samples if s["label"] != LABEL_AMBIGUOUS
    }
    kept = [
        {**r, "label": frp_label[r["patch"]]} for r in records if r["patch"] in frp_label
    ]
    return kept, summary


def build_baseline_report(
    roots: list[str | Path],
    *,
    label_scheme: str = "frp",
    agc_mode: str = "fixed",
    profile: str = "B",
    frp_threshold: float = DEFAULT_FRP_THRESHOLD_MW,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = config.DEFAULT_SEED,
    patches_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build the feasibility-gate report on the train+val granules.

    Args:
        roots: Data roots (same set training merges).
        label_scheme: ``"datagen"`` (manifest label) or ``"frp"`` (FRP relabel).
        agc_mode: AGC variant to read.
        profile: Degradation profile to read.
        frp_threshold: Summed-FRP threshold for ``label_scheme="frp"``.
        cv_folds: Requested grouped-CV fold count.
        seed: RNG seed.
        patches_manifest_path: Optional ``03_patches/patches_manifest.json`` for
            :func:`recalibrate_min_fire_fraction` (real data only).

    Returns:
        The report dict. Callers write it to ``baseline_report.json``.
    """
    roots = [Path(r) for r in roots]
    records, _ = load_merged_records(roots)
    records = [r for r in records if r.get("split") in ("train", "val")]
    if not records:
        raise ValueError("No train/val records after merge — run the datagen split.")

    records, frp_summary = _apply_label_scheme(
        records, roots, label_scheme, agc_mode, profile, frp_threshold
    )
    if not records:
        raise ValueError(
            "No records left after the FRP relabel dropped the ambiguous band."
        )

    X, y, groups = build_feature_matrix(records, agc_mode=agc_mode, profile=profile)
    if X.size == 0:
        raise ValueError(
            f"No {agc_mode}/{profile} images found for any record under {roots}."
        )

    degeneracy = detect_degenerate_labels(y, groups)
    cv_result = grouped_cv_auc(X, y, groups, n_splits=cv_folds, seed=seed)
    feature_auc = per_feature_auc(X, y)

    recalibration = None
    if patches_manifest_path is not None:
        recalibration = recalibrate_min_fire_fraction(Path(patches_manifest_path))

    n_fire = int(y.sum())
    return {
        "stage": "baseline_feasibility_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_version": config.SCRIPT_VERSION,
        "label_scheme": label_scheme,
        "frp_threshold_mw": frp_threshold if label_scheme == "frp" else None,
        "agc_mode": agc_mode,
        "profile": profile,
        "seed": seed,
        "n_records_used": int(y.size),
        "class_balance": {
            "n_fire": n_fire,
            "n_nofire": int(y.size - n_fire),
            "fire_fraction": float(y.mean()) if y.size else 0.0,
        },
        "degenerate": degeneracy["degenerate"],
        "degenerate_warnings": degeneracy["warnings"],
        "n_granules_used": degeneracy["n_granules"],
        "frp_relabel": frp_summary,
        "grouped_cv": cv_result,
        "per_feature_auc": feature_auc,
        "min_fire_fraction_recalibration": recalibration,
        "notes": (
            "grouped_cv.auc_mean is the number a CNN must beat to justify its "
            "complexity. An AUC near 0.5 across every feature and fold means the "
            "fire signal is not detectable at this GSD/AGC with this label. Run "
            "with label_scheme='datagen' to reproduce the negative result that "
            "motivated frp_labels.py, and label_scheme='frp' to show the "
            "reformulated target is learnable. The test split is never read here."
        ),
    }
