"""Parity analysis: fp32 vs int8 score comparison, SQNR, KL, threshold re-sweep.

This module is the quantization analogue of ``evaluation/metrics.py``.  It is
deliberately kept free of heavy ML framework imports at the top level (only
``numpy`` and stdlib) so that the metric-computation helpers can be unit-tested
in CI without requiring ``torch`` or ``ai-edge-torch``.  TFLite-dependent
functions (``collect_tflite_scores``) import ``ai_edge_litert`` lazily.

IAC paper outputs produced here
---------------------------------
- **SQNR** (Signal-to-Quantization-Noise Ratio, dB): measures per-output-neuron
  quantization distortion in the sigmoid score space.  SQNR ≥ 30 dB is
  generally considered acceptable for classification.
- **KL divergence** KL(fp32 ‖ int8): measures the shift in the score
  distribution.  A large KL indicates the int8 model uses a different region
  of the sigmoid curve, which can invalidate the float32 operating threshold
  even if top-1 accuracy is unchanged.
- **Per-metric deltas** (Δ recall, Δ precision, Δ specificity, Δ F-beta,
  Δ ROC-AUC, Δ PR-AUC): the primary quantization-accuracy budget.
- **Acceptance verdict**: pass/fail against stated tolerances.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Acceptance gate defaults ──────────────────────────────────────────────────
# These are the maximum tolerated degradation (int8 minus fp32) for deployment.
# Values are in absolute percentage points for proportion-valued metrics.
DEFAULT_RECALL_TOL: float = 0.005  # ≤ 0.5 pp recall drop
DEFAULT_FBETA_TOL: float = 0.010  # ≤ 1.0 pp F-beta drop


# ── Pure-math helpers (torch-free, CI-safe) ───────────────────────────────────


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_sqnr(fp32_scores: list, int8_scores: list) -> float:
    """Signal-to-Quantization-Noise Ratio between fp32 and int8 score vectors.

    Computes SQNR in the sigmoid score space, treating the float32 scores as
    the reference signal and (fp32 - int8) as the quantization noise::

        SQNR = 10 · log10(E[fp32²] / E[(fp32 - int8)²])

    Higher SQNR indicates less distortion.  For INT8 quantization of well-
    trained CNNs, SQNR ≥ 30 dB in the output layer is typically sufficient
    for accuracy preservation.

    Args:
        fp32_scores: Float32 sigmoid scores in [0, 1].
        int8_scores: Int8-derived sigmoid scores in [0, 1].

    Returns:
        SQNR in dB.  Returns ``inf`` if the noise is exactly zero (lossless).
    """
    s = np.asarray(fp32_scores, dtype=np.float64)
    q = np.asarray(int8_scores, dtype=np.float64)
    noise = s - q
    signal_power = float(np.mean(s**2))
    noise_power = float(np.mean(noise**2))
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_kl_divergence(
    fp32_scores: list,
    int8_scores: list,
    n_bins: int = 64,
) -> float:
    """KL divergence KL(fp32 ‖ int8) via histogram approximation.

    Approximates the distributions using ``n_bins`` equal-width histogram bins
    over [0, 1].  A small epsilon is added before normalisation to avoid log(0).

    Lower KL divergence indicates the int8 model preserves the score
    distribution shape — important because the operating threshold was tuned on
    the float32 distribution.  A high KL (> 0.05 nats) signals that the
    threshold should be re-swept rather than carried over.

    Args:
        fp32_scores: Float32 sigmoid scores in [0, 1].
        int8_scores: Int8-derived sigmoid scores in [0, 1].
        n_bins: Histogram resolution (default: 64).

    Returns:
        KL divergence in nats.  Returns 0.0 if distributions are identical.
    """
    eps = 1e-9
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    p, _ = np.histogram(fp32_scores, bins=bins, density=True)
    q, _ = np.histogram(int8_scores, bins=bins, density=True)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


def select_threshold_from_scores(
    y_true: list,
    y_score: list,
    recall_target: float = 0.99,
) -> tuple[float, bool]:
    """Re-sweep the val-set operating threshold on a new score distribution.

    Identical algorithm to ``train_cloud._select_threshold``: sweeps all unique
    score values (descending) and returns the highest threshold whose cloud
    recall ≥ ``recall_target``, maximising precision under the recall floor.

    This must be re-run on the **int8** model's scores because quantization
    shifts the sigmoid distribution; the float32 threshold is no longer valid.

    Args:
        y_true: Ground-truth binary labels (1=cloud, 0=clear).
        y_score: Sigmoid scores from the int8 model in [0, 1].
        recall_target: Minimum cloud recall required (default: 0.99).

    Returns:
        ``(threshold, degenerate)``.  ``degenerate`` is True when only one
        class is present in ``y_true`` or no candidate achieves the target —
        in both cases the threshold falls back to 0.5.
    """
    arr_true = np.asarray(y_true, dtype=int)
    arr_score = np.asarray(y_score, dtype=float)

    if len(np.unique(arr_true)) < 2:
        logger.warning(
            "Threshold sweep skipped — val set contains only one class; "
            "falling back to 0.5."
        )
        return 0.5, True

    n_pos = int((arr_true == 1).sum())
    candidates = np.sort(np.unique(arr_score))[::-1]
    best_thr: Optional[float] = None

    for thr in candidates:
        tp = int(((arr_score >= thr) & (arr_true == 1)).sum())
        if tp / n_pos >= recall_target:
            if best_thr is None or thr > best_thr:
                best_thr = float(thr)

    if best_thr is None:
        logger.warning(
            "No threshold achieves val recall >= %.2f on int8 scores; "
            "falling back to 0.5.  Consider lowering --recall_target.",
            recall_target,
        )
        return 0.5, True

    return best_thr, False


def compute_metric_deltas(
    fp32_metrics: dict,
    int8_metrics: dict,
    keys: Optional[list] = None,
) -> dict:
    """Compute per-metric absolute deltas (int8 − fp32).

    Args:
        fp32_metrics: Metrics dict from ``metrics.compute_classification_metrics``.
        int8_metrics: Same structure, computed on the int8 model.
        keys: Metric names to compare.  Defaults to the primary classification
            metrics reported in the IAC paper.

    Returns:
        Dict mapping ``metric_name → {fp32, int8, delta}`` for each key present
        in both dicts.
    """
    if keys is None:
        keys = [
            "recall",
            "precision",
            "specificity",
            "f_beta",
            "roc_auc",
            "average_precision",
        ]
    result: dict = {}
    for k in keys:
        v_fp = fp32_metrics.get(k)
        v_int = int8_metrics.get(k)
        if v_fp is not None and v_int is not None:
            result[k] = {
                "fp32": round(float(v_fp), 6),
                "int8": round(float(v_int), 6),
                "delta": round(float(v_int) - float(v_fp), 6),
            }
    return result


def check_acceptance(
    metric_deltas: dict,
    recall_tol: float = DEFAULT_RECALL_TOL,
    fbeta_tol: float = DEFAULT_FBETA_TOL,
) -> tuple[bool, list]:
    """Evaluate the quantization acceptance gate.

    Checks whether the int8 model's recall and F-beta degradation are within
    the stated tolerances.  Returns a pass/fail verdict and a list of
    violated constraints for the report.

    The asymmetric cost function (pos_weight=3) makes recall the primary gate:
    a cloud recall drop > 0.5 pp could translate to missed-cloud FNs that waste
    irreplaceable downlink bandwidth on a mission with no second chance.

    Args:
        metric_deltas: Output of ``compute_metric_deltas``.
        recall_tol: Maximum acceptable recall *drop* (fp32 − int8).  Positive
            means int8 recall is lower (degraded).
        fbeta_tol: Maximum acceptable F-beta *drop*.

    Returns:
        ``(passed, violations)`` where ``violations`` is a list of human-readable
        strings describing each failed constraint.
    """
    violations: list = []

    recall_entry = metric_deltas.get("recall")
    if recall_entry is not None:
        drop = recall_entry["fp32"] - recall_entry["int8"]
        if drop > recall_tol:
            violations.append(
                f"recall drop {drop:.4f} exceeds tolerance {recall_tol:.4f} "
                f"(fp32={recall_entry['fp32']:.4f}, int8={recall_entry['int8']:.4f})"
            )

    fbeta_entry = metric_deltas.get("f_beta")
    if fbeta_entry is not None:
        drop = fbeta_entry["fp32"] - fbeta_entry["int8"]
        if drop > fbeta_tol:
            violations.append(
                f"f_beta drop {drop:.4f} exceeds tolerance {fbeta_tol:.4f} "
                f"(fp32={fbeta_entry['fp32']:.4f}, int8={fbeta_entry['int8']:.4f})"
            )

    return len(violations) == 0, violations


# ── TFLite score collection (ai-edge-litert required) ─────────────────────────


def collect_tflite_scores(
    tflite_path: Path,
    records: list,
    image_dir: Path,
    image_size: int = 224,
) -> tuple[list, list]:
    """Run the int8 TFLite model on host and return (y_true, y_score).

    Uses ``ai-edge-litert`` for host-side inference, identical to what the
    Neural-ART NPU will execute on-device.  Images are loaded as uint8 and
    fed directly (no float conversion or normalization) because normalization
    is baked into the TFLite graph via ``NormalizedMobileNetV2``.

    Args:
        tflite_path: Path to the int8 ``.tflite`` file.
        records: Manifest records with ``image`` and ``label`` keys.
        image_dir: Directory containing PNG images.
        image_size: Spatial resize dimension (must match model input, default: 224).

    Returns:
        ``(y_true, y_score)`` — ground-truth labels and sigmoid scores from the
        int8 model.

    Raises:
        ImportError: If ``ai-edge-litert`` is not installed.
    """
    from PIL import Image as PILImage

    from nina_cloud.quantization.export_tflite import get_interpreter, tflite_logit

    interp = get_interpreter(tflite_path)
    y_true: list = []
    y_score: list = []

    for rec in records:
        img = PILImage.open(image_dir / rec["image"]).convert("RGB")
        img = img.resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.uint8)
        logit = tflite_logit(interp, arr)
        y_true.append(int(rec["label"]))
        y_score.append(_sigmoid(logit))

    return y_true, y_score


# ── Full parity report ────────────────────────────────────────────────────────


def build_parity_report(
    fp32_y_true: list,
    fp32_y_score: list,
    int8_y_true: list,
    int8_y_score: list,
    int8_threshold: float,
    fp32_threshold: float,
    beta: float = math.sqrt(3.0),
    eval_prevalences: Optional[list] = None,
    recall_tol: float = DEFAULT_RECALL_TOL,
    fbeta_tol: float = DEFAULT_FBETA_TOL,
) -> dict:
    """Build the full fp32-vs-int8 parity comparison block for the report.

    Calls ``nina_cloud.evaluation.metrics.compute_classification_metrics``
    (torch-free) on both score sets and computes SQNR + KL for the paper.

    Args:
        fp32_y_true: Ground-truth labels for the fp32 evaluation.
        fp32_y_score: Float32 sigmoid scores.
        int8_y_true: Ground-truth labels for the int8 evaluation (same split).
        int8_y_score: Int8-derived sigmoid scores.
        int8_threshold: Re-swept operating threshold from ``select_threshold_from_scores``.
        fp32_threshold: Original float32 threshold from ``training_report.json``.
        beta: F-beta weighting (default: sqrt(3) to match pos_weight=3).
        eval_prevalences: Cloud prevalences for Bayes-projected precision.
        recall_tol: Acceptance gate for recall drop.
        fbeta_tol: Acceptance gate for F-beta drop.

    Returns:
        Dict with keys: ``sqnr_db``, ``kl_divergence``, ``fp32_threshold``,
        ``int8_threshold``, ``nominal_deltas``, ``acceptance``.
    """
    from nina_cloud.evaluation.metrics import compute_classification_metrics

    if eval_prevalences is None:
        from nina_cloud.datagen import config as datagen_config

        eval_prevalences = datagen_config.EVAL_PREVALENCES

    fp32_y_pred = [1 if s >= fp32_threshold else 0 for s in fp32_y_score]
    int8_y_pred = [1 if s >= int8_threshold else 0 for s in int8_y_score]

    fp32_metrics = compute_classification_metrics(
        fp32_y_true, fp32_y_pred, fp32_y_score, beta=beta, prevalences=eval_prevalences
    )
    int8_metrics = compute_classification_metrics(
        int8_y_true, int8_y_pred, int8_y_score, beta=beta, prevalences=eval_prevalences
    )

    sqnr = compute_sqnr(fp32_y_score, int8_y_score)
    kl = compute_kl_divergence(fp32_y_score, int8_y_score)
    deltas = compute_metric_deltas(fp32_metrics, int8_metrics)
    passed, violations = check_acceptance(deltas, recall_tol, fbeta_tol)

    if passed:
        logger.info("Acceptance gate PASSED — int8 model within tolerances.")
    else:
        logger.warning(
            "Acceptance gate FAILED — %d violation(s): %s",
            len(violations),
            "; ".join(violations),
        )

    return {
        "sqnr_db": round(sqnr, 2) if not math.isinf(sqnr) else None,
        "kl_divergence": round(kl, 6),
        "fp32_threshold": round(fp32_threshold, 6),
        "int8_threshold": round(int8_threshold, 6),
        "fp32_metrics": fp32_metrics,
        "int8_metrics": int8_metrics,
        "nominal_deltas": deltas,
        "acceptance": {
            "recall_drop_tolerance": recall_tol,
            "fbeta_drop_tolerance": fbeta_tol,
            "passed": passed,
            "violations": violations,
        },
    }
