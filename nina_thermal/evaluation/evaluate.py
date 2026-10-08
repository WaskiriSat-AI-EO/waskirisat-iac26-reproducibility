"""Granule-disjoint test-split evaluation for the thermal fire classifier.

``train_thermal.py`` reports validation metrics only; ``run_quantize.py`` also
touches val only. Nothing committed evaluates the held-out **test** split, so
the "test ROC-AUC 0.787" in ``TRAINING_REPORT.md`` came from an uncommitted
script. This module is that evaluation, made reproducible.

It reuses PR #7's data path exactly — :func:`load_merged_records` ->
:func:`expand_to_samples` -> :func:`apply_frp_labels` -> :func:`split_samples`
— so the test set is the same physical granules the training run held out, with
the same FRP relabel. The model is rebuilt with
:func:`nina_thermal.training.train_thermal.build_model` (the single
architecture definition) and the operating threshold comes from
``training_report.json``.

Report structure
----------------
* ``nominal`` — profile-B metrics (the deployment headline) with granule-level
  cluster-bootstrap CIs.
* ``by_profile`` + ``deltas_vs_B`` — A/B/C robustness table. Profiles A and C
  are out-of-distribution for a B-trained model, so the signed drop is the
  reportable robustness bound.
* ``by_surface_type`` — ``warm_surface`` (hot desert / bare rock / urban heat,
  label 0 but warm) isolated from ordinary ``nominal`` no-fire patches, since
  it is the primary false-positive risk.
"""

import json
import logging
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from nina_thermal.datagen import config as datagen_config
from nina_thermal.evaluation import metrics as M
from nina_thermal.training.dataset import (
    ThermalDataset,
    expand_to_samples,
    load_merged_records,
    split_samples,
)
from nina_thermal.training.frp_labels import DEFAULT_FRP_THRESHOLD_MW, apply_frp_labels
from nina_thermal.training.threshold import select_threshold
from nina_thermal.training.train_thermal import (
    CHECKPOINT_NAME,
    build_model,
    build_transforms,
)

logger = logging.getLogger(__name__)

# Test-set variant policy (PR #5): all degradation profiles, flight AGC only.
TEST_PROFILES: list[str] = datagen_config.TEST_PROFILES
EVAL_AGC_MODES: list[str] = datagen_config.EVAL_AGC_MODES
_DEFAULT_BETA = math.sqrt(3.0)  # sqrt(POS_WEIGHT); overridden by training_report


# ── Model + inference ───────────────────────────────────────────────────────


def load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    """Rebuild the architecture and load a bare state-dict checkpoint.

    Args:
        checkpoint_path: Path to ``mobilenetv3s_fire_best.pt``.
        device: Target device.

    Returns:
        Model in ``eval()`` mode on *device*.
    """
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def run_inference(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
    threshold: float,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Score every sample and attach ``score`` (sigmoid) and ``pred`` (hard).

    The loader is unshuffled, so the returned list aligns 1:1 with *samples*.
    """
    loader = DataLoader(
        ThermalDataset(samples, build_transforms(augment=False)),
        batch_size=batch_size,
        shuffle=False,
    )
    scores: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device)).squeeze(1)
            scores.extend(logits.sigmoid().cpu().tolist())

    out = []
    for sample, score in zip(samples, scores, strict=True):
        out.append(
            {
                "patch": sample["patch"],
                "granule_id": sample["granule_id"],
                "profile": sample["profile"],
                "agc_mode": sample["agc_mode"],
                "surface_type": sample.get("surface_type", "unknown"),
                "label": int(sample["label"]),
                "score": round(float(score), 6),
                "pred": int(score >= threshold),
            }
        )
    return out


def _resolve_threshold_and_beta(
    training_report: Path | None, explicit_threshold: float | None
) -> tuple[float | None, float]:
    """Return ``(threshold_or_None, beta)``.

    ``threshold`` is ``None`` when neither an explicit value nor a training
    report is available — the caller then sweeps it on the test scores and
    flags the report.
    """
    if explicit_threshold is not None:
        return explicit_threshold, _DEFAULT_BETA
    if training_report and training_report.is_file():
        tr = json.loads(training_report.read_text(encoding="utf-8"))
        beta = float(tr.get("hyperparameters", {}).get("beta", _DEFAULT_BETA))
        thr = tr.get("selected_threshold")
        return (float(thr) if thr is not None else None), beta
    return None, _DEFAULT_BETA


# ── Report ──────────────────────────────────────────────────────────────────


def build_evaluation_report(
    roots: list[str | Path],
    checkpoint_path: Path,
    out_dir: Path,
    device: torch.device,
    *,
    frp_threshold: float = DEFAULT_FRP_THRESHOLD_MW,
    frp_negative_max: float = 0.0,
    explicit_threshold: float | None = None,
    training_report: Path | None = None,
    batch_size: int = 32,
    seed: int = datagen_config.DEFAULT_SEED,
    tflite_path: Path | None = None,
    tflite_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate the test split and write ``evaluation_summary.json``.

    Args:
        roots: Data roots (same set training merged).
        checkpoint_path: Trained fp32 checkpoint.
        out_dir: Directory for ``evaluation_summary.json`` + ``predictions.json``.
        device: Inference device.
        frp_threshold: Summed-FRP threshold for the relabel (must match training).
        frp_negative_max: Summed-FRP ceiling for the negative class (must
            match training).
        explicit_threshold: Override the operating threshold.
        training_report: ``training_report.json`` to read
            ``selected_threshold`` / ``beta`` from. Defaults to
            ``out_dir / "training_report.json"`` then
            ``<checkpoint dir> / "training_report.json"``.
        batch_size: DataLoader batch size.
        seed: Bootstrap resampling seed.
        tflite_path: Optional int8 artifact to score on the same test images;
            adds an ``int8`` block and per-record ``int8_score`` / ``int8_pred``.
        tflite_threshold: int8 operating threshold.  Defaults to
            ``int8.threshold`` in the ``quantization_report.json`` beside the
            artifact.

    Returns:
        The report dict (also written to disk).

    Raises:
        ValueError: If the merged manifest has no ``test`` split.
    """
    roots = [Path(r) for r in roots]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if training_report is None:
        for candidate in (
            out_dir / "training_report.json",
            checkpoint_path.parent / "training_report.json",
        ):
            if candidate.is_file():
                training_report = candidate
                break
    threshold, beta = _resolve_threshold_and_beta(training_report, explicit_threshold)

    records, merge_stats = load_merged_records(roots)
    samples = expand_to_samples(records, TEST_PROFILES, EVAL_AGC_MODES)
    samples, frp_summary = apply_frp_labels(
        samples, roots, frp_threshold, frp_negative_max
    )
    by_split = split_samples(samples)
    test_samples = by_split.get("test", [])
    if not test_samples:
        raise ValueError(
            "No 'test' samples after merge + FRP relabel + split. Check the "
            "roots and that run_split assigned a test partition."
        )

    threshold_swept = threshold is None
    if threshold_swept:
        # No training report — sweep on the test scores at the default recall
        # floor, and flag it. This is not a validated operating point.
        provisional = run_inference(
            load_checkpoint_model(checkpoint_path, device),
            test_samples,
            device,
            0.5,
            batch_size,
        )
        threshold, degenerate = select_threshold(
            [r["label"] for r in provisional],
            [r["score"] for r in provisional],
            recall_target=0.95,
        )
        logger.warning(
            "No training_report.json — swept threshold=%.4f on the TEST scores "
            "(degenerate=%s). Not a validated operating point.",
            threshold,
            degenerate,
        )

    assert threshold is not None  # set by _resolve or the swept block above

    model = load_checkpoint_model(checkpoint_path, device)
    predictions = run_inference(model, test_samples, device, threshold, batch_size)

    nominal = [r for r in predictions if r["profile"] == M.NOMINAL_PROFILE]
    if not nominal:
        logger.warning(
            "No profile-%s records in the test split — using all profiles for "
            "the nominal block.",
            M.NOMINAL_PROFILE,
        )
        nominal = predictions

    n_fire_nominal = sum(1 for r in nominal if r["label"] == 1)
    degenerate_eval = n_fire_nominal < M.MIN_FIRE_FOR_EVAL
    if degenerate_eval:
        warnings.warn(
            f"Nominal test slice has {n_fire_nominal} fire patches "
            f"(< MIN_FIRE_FOR_EVAL={M.MIN_FIRE_FOR_EVAL}); recall / PR-AUC / "
            "bootstrap CIs are unreliable.",
            stacklevel=2,
        )

    nominal_metrics = M.compute_classification_metrics(
        [r["label"] for r in nominal],
        [r["pred"] for r in nominal],
        [r["score"] for r in nominal],
        beta=beta,
        prevalences=M.EVAL_PREVALENCES,
    )
    nominal_metrics["bootstrap_ci"] = _bootstrap_block(nominal, threshold, beta, seed)

    by_profile = M.group_metrics(
        predictions, "profile", beta=beta, prevalences=M.EVAL_PREVALENCES
    )
    by_surface_type = M.group_metrics(
        nominal, "surface_type", beta=beta, prevalences=M.EVAL_PREVALENCES
    )

    report: dict[str, Any] = {
        "stage": "run_evaluate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_version": datagen_config.SCRIPT_VERSION,
        "checkpoint": Path(checkpoint_path).name,
        "device": str(device),
        "frp_threshold_mw": frp_threshold,
        "operating_threshold": round(float(threshold), 6),
        "threshold_source": (
            "swept_on_test"
            if threshold_swept
            else ("explicit" if explicit_threshold is not None else "training_report")
        ),
        "beta": round(beta, 6),
        "degenerate_eval": degenerate_eval,
        "test_composition": {
            "images": len(predictions),
            "patches": len({r["patch"] for r in predictions}),
            "granules": len({r["granule_id"] for r in predictions}),
            "nominal_images": len(nominal),
            "nominal_fire": n_fire_nominal,
        },
        "merge_stats": merge_stats,
        "frp_relabel": frp_summary,
        "nominal": nominal_metrics,
        "by_profile": by_profile,
        "deltas_vs_B": M.profile_deltas(by_profile),
        "by_surface_type": by_surface_type,
        **(
            {"int8": _int8_block(tflite_path, tflite_threshold, test_samples,
                                 predictions, nominal_metrics, beta, seed)}
            if tflite_path is not None
            else {}
        ),
        "notes": (
            f"'nominal' = profile {M.NOMINAL_PROFILE} only (the training / "
            "deployment configuration). Profiles A/C are out-of-distribution "
            "robustness bounds. 'bootstrap_ci' is a granule-level cluster "
            "bootstrap — patch-level resampling understates interval width at "
            "50% tile overlap. 'precision' is measured-as-sampled; "
            "'precision_at_prevalence' is the Bayes projection for the stated "
            "operational fire prevalences."
        ),
    }

    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "predictions.json").write_text(
        json.dumps({"total": len(predictions), "records": predictions}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Evaluation report -> %s  (test recall=%.3f roc_auc=%s)",
        out_dir / "evaluation_summary.json",
        nominal_metrics["recall"],
        nominal_metrics["roc_auc"],
    )
    return report


def _int8_block(
    tflite_path: Path,
    threshold: float | None,
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    fp32_nominal: dict[str, Any],
    beta: float,
    seed: int,
) -> dict[str, Any]:
    """Score an int8 TFLite on the test samples and return its metrics block.

    Mutates *predictions* in place, adding ``int8_score`` / ``int8_pred`` so
    ``predictions.json`` carries both models side by side.
    """
    from nina_thermal.quantization import qat as qat_lib
    from nina_thermal.quantization.pipelines.run_quantize import qat_transform

    tflite_path = Path(tflite_path)
    if threshold is None:
        qreport = tflite_path.parent / "quantization_report.json"
        if not qreport.is_file():
            raise ValueError(
                f"No int8 threshold: pass --tflite_threshold or provide {qreport}."
            )
        threshold = float(json.loads(qreport.read_text(encoding="utf-8"))["int8"]["threshold"])

    # The artifact folds normalization into the graph: feed [0, 1] pixels.
    loader = DataLoader(
        ThermalDataset(samples, qat_transform(augment=False)), batch_size=32, shuffle=False
    )
    scores = qat_lib.tflite_scores(tflite_path, loader)
    for record, score in zip(predictions, scores, strict=True):
        record["int8_score"] = round(float(score), 6)
        record["int8_pred"] = int(score >= threshold)
    int8_records = [
        {**r, "score": r["int8_score"], "pred": r["int8_pred"]} for r in predictions
    ]
    nominal = [r for r in int8_records if r["profile"] == M.NOMINAL_PROFILE] or int8_records
    nominal_metrics = M.compute_classification_metrics(
        [r["label"] for r in nominal],
        [r["pred"] for r in nominal],
        [r["score"] for r in nominal],
        beta=beta,
        prevalences=M.EVAL_PREVALENCES,
    )
    nominal_metrics["bootstrap_ci"] = _bootstrap_block(nominal, threshold, beta, seed)
    deltas = {
        k: round(nominal_metrics[k] - fp32_nominal[k], 6)
        for k in ("accuracy", "recall", "specificity", "precision", "f1", "roc_auc")
        if nominal_metrics.get(k) is not None and fp32_nominal.get(k) is not None
    }
    logger.info(
        "int8 test (profile %s): accuracy=%.3f recall=%.3f roc_auc=%s",
        M.NOMINAL_PROFILE,
        nominal_metrics["accuracy"],
        nominal_metrics["recall"],
        nominal_metrics["roc_auc"],
    )
    return {
        "artifact": str(tflite_path),
        "operating_threshold": round(threshold, 6),
        "nominal": nominal_metrics,
        "by_profile": M.group_metrics(
            int8_records, "profile", beta=beta, prevalences=M.EVAL_PREVALENCES
        ),
        "deltas_vs_fp32_nominal": deltas,
    }


def _bootstrap_block(
    nominal: list[dict[str, Any]], threshold: float, beta: float, seed: int
) -> dict[str, Any]:
    """Granule-level cluster-bootstrap CIs for the nominal headline metrics."""
    y_true = [r["label"] for r in nominal]
    y_score = [r["score"] for r in nominal]
    groups = [r["granule_id"] for r in nominal]
    return {
        "recall": M.cluster_bootstrap_ci(
            y_true, y_score, groups, M.recall_fn(threshold), seed=seed
        ),
        "precision": M.cluster_bootstrap_ci(
            y_true, y_score, groups, M.precision_fn(threshold), seed=seed
        ),
        "fbeta": M.cluster_bootstrap_ci(
            y_true, y_score, groups, M.fbeta_fn(threshold, beta), seed=seed
        ),
        "roc_auc": M.cluster_bootstrap_ci(
            y_true, y_score, groups, M.roc_auc_fn, seed=seed
        ),
    }


# ── Figures ─────────────────────────────────────────────────────────────────


def save_figures(
    report: dict[str, Any], predictions: list[dict[str, Any]], out_dir: Path
) -> None:
    """Write one 3-panel PNG (confusion | ROC | PR) per slice to ``<out_dir>/figures/``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _plot(
        group_name: str, metrics: dict[str, Any], recs: list[dict[str, Any]]
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Evaluation — {group_name}", fontsize=13)

        cm = metrics["confusion_matrix"]
        ax = axes[0]
        ax.axis("off")
        ax.set_title("Confusion Matrix", pad=14)
        tbl = ax.table(
            cellText=[[str(cm["tn"]), str(cm["fp"])], [str(cm["fn"]), str(cm["tp"])]],
            rowLabels=["True No-fire", "True Fire"],
            colLabels=["Pred No-fire", "Pred Fire"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.4, 2.0)

        y_true = np.array([r["label"] for r in recs])
        y_score = np.array([r["score"] for r in recs])
        ax2 = axes[1]
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax2.plot(fpr, tpr, lw=2, label=f"AUC = {metrics.get('roc_auc') or 0.0:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", lw=1)
            ax2.legend(loc="lower right")
        else:
            ax2.text(0.5, 0.5, "Single class — ROC N/A", ha="center", va="center")
        ax2.set_title("ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")

        ax3 = axes[2]
        pr = M.compute_pr_curve(y_true.tolist(), y_score.tolist())
        if pr is not None:
            p_vals, r_vals, _ = pr
            ax3.plot(
                r_vals,
                p_vals,
                lw=2,
                label=f"AP = {metrics.get('average_precision') or 0.0:.3f}",
            )
            base = metrics.get("n_fire", 0) / max(metrics.get("n", 1), 1)
            ax3.axhline(
                base, color="k", linestyle="--", lw=1, label=f"baseline (pi={base:.2f})"
            )
            ax3.legend(loc="upper right")
        else:
            ax3.text(0.5, 0.5, "Single class — PR N/A", ha="center", va="center")
        ax3.set_title("Precision-Recall Curve")
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")

        fig.subplots_adjust(top=0.88, wspace=0.4, left=0.08, right=0.97)
        path = figures_dir / f"eval_{group_name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    nominal = [
        r for r in predictions if r["profile"] == M.NOMINAL_PROFILE
    ] or predictions
    _plot(f"nominal_profile_{M.NOMINAL_PROFILE}", report["nominal"], nominal)

    by_profile_recs: dict[str, list[dict[str, Any]]] = {}
    for rec in predictions:
        by_profile_recs.setdefault(rec["profile"], []).append(rec)
    for profile, recs in sorted(by_profile_recs.items()):
        if report["by_profile"].get(profile):
            _plot(f"profile_{profile}", report["by_profile"][profile], recs)

    warm_recs = [r for r in nominal if r["surface_type"] == "warm_surface"]
    warm_metrics = report.get("by_surface_type", {}).get("warm_surface")
    if warm_recs and warm_metrics:
        _plot("warm_surface", warm_metrics, warm_recs)


DEFAULT_CHECKPOINT_NAME = CHECKPOINT_NAME
