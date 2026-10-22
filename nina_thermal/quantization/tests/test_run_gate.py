"""Tests for nina_thermal.quantization.pipelines.run_gate.

The TFLite interpreter is replaced by stand-ins, so these check the data path
and that the gate is wired to ``parity.check_acceptance``, not real parity.
"""

import numpy as np
import torch

from nina_thermal.quantization import qat as qat_lib
from nina_thermal.quantization.pipelines import run_gate
from nina_thermal.training.train_thermal import collect_scores


def _gate(roots, ckpt, tmp_path):
    return run_gate.compute_gate(
        [str(r) for r in roots], ckpt, tmp_path / "fake.tflite", torch.device("cpu")
    )


def test_identical_scores_pass(eval_multiroot, random_checkpoint, tmp_path, monkeypatch):
    ckpt, _ = random_checkpoint
    # int8 == fp32 scores: no drop, infinite SQNR, zero KL
    monkeypatch.setattr(
        qat_lib,
        "tflite_scores",
        lambda path, loader: collect_scores(
            qat_lib.build_model_for_qat(ckpt, torch.device("cpu")),
            loader,
            torch.device("cpu"),
        )[1],
    )
    result = _gate(eval_multiroot.roots, ckpt, tmp_path)
    assert result["acceptance"]["passed"] is True
    assert result["val_composition"]["patches"] == len(result["val_scores"])


def test_unrelated_scores_fail(eval_multiroot, random_checkpoint, tmp_path, monkeypatch):
    ckpt, _ = random_checkpoint
    # int8 scores unrelated to fp32: SQNR and KL must both fail
    rng = np.random.default_rng(0)
    monkeypatch.setattr(
        qat_lib,
        "tflite_scores",
        lambda path, loader: rng.uniform(size=len(loader.dataset)).tolist(),
    )
    result = _gate(eval_multiroot.roots, ckpt, tmp_path)
    assert result["acceptance"]["passed"] is False
    assert set(result["acceptance"]["measured"]) == {
        "recall_drop",
        "fbeta_drop",
        "sqnr_db",
        "kl_nats",
    }
