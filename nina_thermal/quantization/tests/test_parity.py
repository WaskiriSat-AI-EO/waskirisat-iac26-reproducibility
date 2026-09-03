"""Tests for nina_thermal.quantization.parity — SQNR, KL, acceptance gate."""

import numpy as np
import pytest

from nina_thermal.quantization import parity


class TestComputeSqnr:
    def test_identical_scores_are_lossless(self):
        s = [0.1, 0.5, 0.9, 0.3]
        assert parity.compute_sqnr(s, s) == float("inf")

    def test_more_noise_lowers_sqnr(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 500).tolist()
        small = (np.array(s) + rng.normal(0, 0.002, 500)).tolist()
        large = (np.array(s) + rng.normal(0, 0.02, 500)).tolist()
        assert parity.compute_sqnr(s, small) > parity.compute_sqnr(s, large)

    def test_30db_corresponds_to_small_relative_noise(self):
        rng = np.random.default_rng(1)
        s = np.clip(0.5 + rng.normal(0, 0.2, 2000), 0, 1)
        q = s + rng.normal(0, 0.003, 2000)  # ~1.5% relative -> comfortably > 30 dB
        assert parity.compute_sqnr(s.tolist(), q.tolist()) > 30.0


class TestComputeKlDivergence:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 1000).tolist()
        assert parity.compute_kl_divergence(s, s) == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_has_positive_kl(self):
        rng = np.random.default_rng(0)
        s = np.clip(rng.normal(0.3, 0.1, 2000), 0, 1).tolist()
        q = np.clip(rng.normal(0.7, 0.1, 2000), 0, 1).tolist()
        assert parity.compute_kl_divergence(s, q) > 0.5


class TestCheckAcceptance:
    def _kw(self, **over):
        base = {
            "fp32_recall": 0.99,
            "int8_recall": 0.99,
            "fp32_fbeta": 0.95,
            "int8_fbeta": 0.95,
            "sqnr_db": 40.0,
            "kl_nats": 0.01,
        }
        base.update(over)
        return base

    def test_passes_within_all_tolerances(self):
        out = parity.check_acceptance(**self._kw())
        assert out["passed"] is True and out["violations"] == []

    def test_recall_drop_beyond_tolerance_fails(self):
        out = parity.check_acceptance(**self._kw(int8_recall=0.97))  # 2 pp drop
        assert out["passed"] is False
        assert any("recall drop" in v for v in out["violations"])

    def test_fbeta_drop_beyond_tolerance_fails(self):
        out = parity.check_acceptance(**self._kw(int8_fbeta=0.90))
        assert any("F-beta drop" in v for v in out["violations"])

    def test_low_sqnr_fails(self):
        out = parity.check_acceptance(**self._kw(sqnr_db=22.0))
        assert any("SQNR" in v for v in out["violations"])

    def test_infinite_sqnr_never_trips_the_gate(self):
        out = parity.check_acceptance(**self._kw(sqnr_db=float("inf")))
        assert out["passed"] is True
        assert out["measured"]["sqnr_db"] is None

    def test_high_kl_fails(self):
        out = parity.check_acceptance(**self._kw(kl_nats=0.2))
        assert any("KL" in v for v in out["violations"])

    def test_int8_better_than_fp32_still_passes(self):
        out = parity.check_acceptance(**self._kw(int8_recall=1.0, int8_fbeta=0.97))
        assert out["passed"] is True

    def test_report_block_shape(self):
        out = parity.check_acceptance(**self._kw())
        assert set(out) == {"passed", "violations", "tolerances", "measured"}
        assert set(out["tolerances"]) == {
            "recall_drop", "fbeta_drop", "sqnr_min_db", "kl_max_nats"
        }
