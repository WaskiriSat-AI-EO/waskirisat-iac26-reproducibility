"""fp32-vs-int8 score-parity metrics and the deployment acceptance gate.

Torch-free (numpy + stdlib) so it unit-tests without the quantization stack.
``run_quantize`` collects the two sigmoid-score vectors and calls these to
decide whether the int8 artifact is fit to fly.

Why score-space, not weight-space
---------------------------------
SQNR and KL are computed on the sigmoid *scores*, because score-level
distortion is what moves the operating threshold — the number that actually
decides a fire call on orbit. A model can lose little top-1 accuracy yet shift
its score distribution enough to invalidate the fp32 threshold.
"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ── Acceptance-gate tolerances ─────────────────────────────────────────────
# The asymmetric FN:FP cost makes recall the primary gate: a fire-recall drop
# beyond tolerance is a missed-fire risk in a disaster-response context.
RECALL_DROP_TOL: float = 0.005  # <= 0.5 pp recall drop (fp32 - int8)
FBETA_DROP_TOL: float = 0.010  # <= 1.0 pp F-beta drop
SQNR_MIN_DB: float = 30.0  # >= 30 dB score-space SQNR
KL_MAX_NATS: float = 0.05  # <= 0.05 nats fp32->int8 score-distribution shift


def compute_sqnr(fp32_scores: list[float], int8_scores: list[float]) -> float:
    """Signal-to-quantization-noise ratio [dB] in sigmoid-score space.

    ``SQNR = 10 * log10(E[fp32^2] / E[(fp32 - int8)^2])``. Returns ``inf`` when
    the int8 scores are bit-identical to fp32.
    """
    s = np.asarray(fp32_scores, dtype=np.float64)
    q = np.asarray(int8_scores, dtype=np.float64)
    noise_power = float(np.mean((s - q) ** 2))
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(float(np.mean(s**2)) / noise_power))


def compute_kl_divergence(
    fp32_scores: list[float], int8_scores: list[float], n_bins: int = 64
) -> float:
    """KL(fp32 || int8) [nats] via an ``n_bins`` histogram over [0, 1].

    A large value means the int8 model works in a different region of the
    sigmoid curve, which can invalidate the fp32 operating threshold even when
    top-1 accuracy is unchanged.
    """
    eps = 1e-9
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    p, _ = np.histogram(fp32_scores, bins=bins, density=True)
    q, _ = np.histogram(int8_scores, bins=bins, density=True)
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


def check_acceptance(
    *,
    fp32_recall: float,
    int8_recall: float,
    fp32_fbeta: float,
    int8_fbeta: float,
    sqnr_db: float,
    kl_nats: float,
    recall_drop_tol: float = RECALL_DROP_TOL,
    fbeta_drop_tol: float = FBETA_DROP_TOL,
    sqnr_min_db: float = SQNR_MIN_DB,
    kl_max_nats: float = KL_MAX_NATS,
) -> dict[str, object]:
    """Evaluate the four-part deployment gate.

    Returns:
        ``{passed: bool, violations: [str, ...], tolerances: {...},
        measured: {...}}``.
    """
    recall_drop = fp32_recall - int8_recall
    fbeta_drop = fp32_fbeta - int8_fbeta
    violations: list[str] = []

    if recall_drop > recall_drop_tol:
        violations.append(
            f"recall drop {recall_drop:.4f} > tol {recall_drop_tol:.4f} "
            f"(fp32={fp32_recall:.4f}, int8={int8_recall:.4f})"
        )
    if fbeta_drop > fbeta_drop_tol:
        violations.append(
            f"F-beta drop {fbeta_drop:.4f} > tol {fbeta_drop_tol:.4f} "
            f"(fp32={fp32_fbeta:.4f}, int8={int8_fbeta:.4f})"
        )
    if not math.isinf(sqnr_db) and sqnr_db < sqnr_min_db:
        violations.append(f"SQNR {sqnr_db:.1f} dB < {sqnr_min_db:.0f} dB minimum")
    if kl_nats > kl_max_nats:
        violations.append(
            f"score-distribution KL {kl_nats:.4f} nats > {kl_max_nats:.4f} nats"
        )

    passed = not violations
    (logger.info if passed else logger.warning)(
        "Acceptance gate %s%s",
        "PASSED" if passed else "FAILED",
        "" if passed else " — " + "; ".join(violations),
    )
    return {
        "passed": passed,
        "violations": violations,
        "tolerances": {
            "recall_drop": recall_drop_tol,
            "fbeta_drop": fbeta_drop_tol,
            "sqnr_min_db": sqnr_min_db,
            "kl_max_nats": kl_max_nats,
        },
        "measured": {
            "recall_drop": round(recall_drop, 6),
            "fbeta_drop": round(fbeta_drop, 6),
            "sqnr_db": None if math.isinf(sqnr_db) else round(sqnr_db, 2),
            "kl_nats": round(kl_nats, 6),
        },
    }
