"""Unit tests for the quantization module's torch-free math helpers.

These tests cover ``parity.py`` utility functions that are pure numpy / stdlib
and run in CI without requiring ``torch``, ``ai-edge-torch``, or ``ai-edge-litert``.

Torch-dependent tests (model wrapping, QAT transforms, export) are grouped in a
separate section and skipped automatically when torch is unavailable.  This
mirrors the design of ``nina_cloud/evaluation/tests/test_metrics.py``.
"""

import math

import numpy as np
import pytest

from nina_cloud.quantization.parity import (
    _sigmoid,
    check_acceptance,
    compute_kl_divergence,
    compute_metric_deltas,
    compute_sqnr,
    select_threshold_from_scores,
)


# ── _sigmoid ──────────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero(self) -> None:
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_large_positive(self) -> None:
        assert abs(_sigmoid(100.0) - 1.0) < 1e-9

    def test_large_negative(self) -> None:
        assert abs(_sigmoid(-100.0) - 0.0) < 1e-9

    def test_known_value(self) -> None:
        # sigmoid(1) ≈ 0.7310586
        assert abs(_sigmoid(1.0) - 0.7310585975) < 1e-6

    def test_symmetry(self) -> None:
        assert abs(_sigmoid(2.0) + _sigmoid(-2.0) - 1.0) < 1e-9


# ── compute_sqnr ──────────────────────────────────────────────────────────────


class TestComputeSqnr:
    def test_identical_scores_returns_inf(self) -> None:
        s = [0.1, 0.5, 0.9]
        assert math.isinf(compute_sqnr(s, s))

    def test_high_sqnr_for_tiny_noise(self) -> None:
        rng = np.random.default_rng(0)
        fp32 = rng.uniform(0.1, 0.9, 200).tolist()
        int8 = [v + rng.normal(0, 1e-4) for v in fp32]
        sqnr = compute_sqnr(fp32, int8)
        assert sqnr > 30.0, f"Expected SQNR > 30 dB for tiny noise, got {sqnr:.1f}"

    def test_low_sqnr_for_large_noise(self) -> None:
        fp32 = [0.5] * 100
        int8 = [0.0] * 100  # maximum distortion
        sqnr = compute_sqnr(fp32, int8)
        assert (
            sqnr < 10.0
        ), f"Expected SQNR < 10 dB for maximum distortion, got {sqnr:.1f}"

    def test_single_element(self) -> None:
        assert compute_sqnr([0.8], [0.8]) == float("inf")
        # noise_power > signal_power → negative SQNR
        # fp32=0.1, int8=0.9: signal=0.01, noise=(0.1-0.9)²=0.64 → SQNR≈-18 dB
        assert compute_sqnr([0.1], [0.9]) < 0

    def test_result_is_float(self) -> None:
        result = compute_sqnr([0.3, 0.7], [0.31, 0.69])
        assert isinstance(result, float)


# ── compute_kl_divergence ─────────────────────────────────────────────────────


class TestComputeKlDivergence:
    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 1, 500).tolist()
        kl = compute_kl_divergence(scores, scores)
        assert kl < 1e-6, f"KL(P‖P) should be ~0, got {kl}"

    def test_different_distributions_positive(self) -> None:
        low = np.random.default_rng(2).uniform(0.0, 0.3, 200).tolist()
        high = np.random.default_rng(3).uniform(0.7, 1.0, 200).tolist()
        kl = compute_kl_divergence(low, high)
        assert kl > 0.0

    def test_result_is_float(self) -> None:
        assert isinstance(compute_kl_divergence([0.5], [0.5]), float)

    def test_asymmetry(self) -> None:
        # KL divergence is asymmetric: KL(P‖Q) ≠ KL(Q‖P) in general
        rng = np.random.default_rng(4)
        p = rng.beta(2, 5, 300).tolist()
        q = rng.beta(5, 2, 300).tolist()
        kl_pq = compute_kl_divergence(p, q)
        kl_qp = compute_kl_divergence(q, p)
        assert abs(kl_pq - kl_qp) > 1e-6


# ── select_threshold_from_scores ──────────────────────────────────────────────


class TestSelectThreshold:
    def _make_scores(self):
        # 10 cloud (label=1), 10 clear (label=0)
        y_true = [1] * 10 + [0] * 10
        # Cloud scores uniformly high, clear scores uniformly low
        y_score = [0.9 - i * 0.01 for i in range(10)] + [
            0.1 + i * 0.01 for i in range(10)
        ]
        return y_true, y_score

    def test_achieves_recall_target(self) -> None:
        y_true, y_score = self._make_scores()
        thr, degen = select_threshold_from_scores(y_true, y_score, recall_target=0.99)
        assert not degen
        arr_true = np.asarray(y_true)
        arr_score = np.asarray(y_score)
        tp = int(((arr_score >= thr) & (arr_true == 1)).sum())
        recall = tp / int((arr_true == 1).sum())
        assert (
            recall >= 0.99
        ), f"Recall {recall:.3f} below target at threshold {thr:.4f}"

    def test_maximises_threshold(self) -> None:
        # Two valid thresholds exist; should return the higher one (max precision)
        y_true = [1, 1, 0, 0]
        y_score = [0.95, 0.85, 0.3, 0.2]
        thr, degen = select_threshold_from_scores(y_true, y_score, recall_target=1.0)
        assert not degen
        assert abs(thr - 0.85) < 1e-9

    def test_single_class_returns_degenerate(self) -> None:
        thr, degen = select_threshold_from_scores([1, 1, 1], [0.9, 0.8, 0.7])
        assert degen
        assert thr == 0.5

    def test_no_feasible_threshold_returns_degenerate(self) -> None:
        # recall_target > 1.0 is unreachable (max recall = 1.0) → degenerate
        _, degen = select_threshold_from_scores([1, 0], [0.3, 0.9], recall_target=1.01)
        assert degen

    def test_fallback_is_0_5(self) -> None:
        thr, _ = select_threshold_from_scores([1, 1, 1], [0.9, 0.8, 0.7])
        assert thr == 0.5


# ── compute_metric_deltas ─────────────────────────────────────────────────────


class TestComputeMetricDeltas:
    def test_delta_direction(self) -> None:
        fp32 = {"recall": 0.99, "f_beta": 0.95, "roc_auc": 0.98}
        int8 = {"recall": 0.985, "f_beta": 0.94, "roc_auc": 0.977}
        deltas = compute_metric_deltas(fp32, int8, keys=["recall", "f_beta", "roc_auc"])
        assert deltas["recall"]["delta"] == pytest.approx(-0.005, abs=1e-6)
        assert deltas["f_beta"]["delta"] == pytest.approx(-0.010, abs=1e-6)

    def test_missing_key_skipped(self) -> None:
        fp32 = {"recall": 0.99}
        int8 = {"precision": 0.8}
        deltas = compute_metric_deltas(fp32, int8, keys=["recall", "precision"])
        assert "recall" not in deltas  # missing from int8
        assert "precision" not in deltas  # missing from fp32

    def test_none_values_skipped(self) -> None:
        fp32 = {"roc_auc": None}
        int8 = {"roc_auc": 0.95}
        deltas = compute_metric_deltas(fp32, int8, keys=["roc_auc"])
        assert "roc_auc" not in deltas


# ── check_acceptance ──────────────────────────────────────────────────────────


class TestCheckAcceptance:
    def _deltas(self, recall_drop: float, fbeta_drop: float) -> dict:
        return {
            "recall": {"fp32": 0.99, "int8": 0.99 - recall_drop, "delta": -recall_drop},
            "f_beta": {"fp32": 0.95, "int8": 0.95 - fbeta_drop, "delta": -fbeta_drop},
        }

    def test_passes_within_tolerance(self) -> None:
        passed, violations = check_acceptance(
            self._deltas(0.003, 0.008), recall_tol=0.005, fbeta_tol=0.010
        )
        assert passed
        assert violations == []

    def test_fails_on_recall_drop(self) -> None:
        passed, violations = check_acceptance(
            self._deltas(0.010, 0.005), recall_tol=0.005, fbeta_tol=0.010
        )
        assert not passed
        assert any("recall" in v for v in violations)

    def test_fails_on_fbeta_drop(self) -> None:
        passed, violations = check_acceptance(
            self._deltas(0.002, 0.015), recall_tol=0.005, fbeta_tol=0.010
        )
        assert not passed
        assert any("f_beta" in v for v in violations)

    def test_both_violations_reported(self) -> None:
        passed, violations = check_acceptance(
            self._deltas(0.010, 0.020), recall_tol=0.005, fbeta_tol=0.010
        )
        assert not passed
        assert len(violations) == 2

    def test_empty_deltas_passes(self) -> None:
        passed, violations = check_acceptance({})
        assert passed
        assert violations == []


# ── Torch-dependent tests (skipped in CI if torch unavailable) ────────────────

torch = pytest.importorskip("torch", reason="torch not installed — skipping QAT tests")


class TestNormalizedMobileNetV2:
    """Verify that NormalizedMobileNetV2 is numerically equivalent to
    ToTensor + transforms.Normalize applied externally."""

    def test_normalization_equivalence(self) -> None:
        from torchvision import transforms

        from nina_cloud.quantization.qat import (
            NormalizedMobileNetV2,
            _IMAGENET_MEAN,
            _IMAGENET_STD,
        )

        # Build a trivial identity backbone to isolate normalization
        class _Identity(torch.nn.Module):
            def forward(self, x):
                return x

        wrapped = NormalizedMobileNetV2(_Identity())
        normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

        rng = torch.Generator().manual_seed(0)
        img = torch.rand(1, 3, 224, 224, generator=rng)  # [0, 1] float

        out_wrapped = wrapped(img)
        out_external = normalize(img.squeeze(0)).unsqueeze(0)

        assert torch.allclose(
            out_wrapped, out_external, atol=1e-6
        ), "NormalizedMobileNetV2 output must match external transforms.Normalize"

    def test_qat_transform_no_normalize(self) -> None:
        """QAT transform must NOT apply Normalize — that's the model's job."""
        from torchvision import transforms

        from nina_cloud.quantization.qat import qat_eval_transform, qat_train_transform

        for tf in [qat_eval_transform(), qat_train_transform()]:
            has_normalize = any(
                isinstance(t, transforms.Normalize) for t in tf.transforms
            )
            assert not has_normalize, (
                "qat_*_transform must not include Normalize — "
                "NormalizedMobileNetV2 handles it internally."
            )
