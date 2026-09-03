"""Tests for nina_thermal.evaluation.evaluate — end-to-end on synthetic data.

Uses random weights, so the metric *values* are meaningless; these assert the
report shape, the data path (merge -> FRP relabel -> test split), threshold
resolution, the degenerate-eval guard, and figure rendering.
"""

import json
import warnings

import torch

from nina_thermal.evaluation import evaluate


def _report(roots, ckpt, out_dir, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate.build_evaluation_report(
            [str(r) for r in roots], ckpt, out_dir, torch.device("cpu"), seed=1, **kw
        )


class TestReportShape:
    def test_all_blocks_present(self, eval_multiroot, random_checkpoint, tmp_path):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        for key in (
            "nominal",
            "by_profile",
            "deltas_vs_B",
            "by_surface_type",
            "merge_stats",
            "frp_relabel",
            "test_composition",
            "operating_threshold",
        ):
            assert key in rep, key
        assert set(rep["by_profile"]) == {"A", "B", "C"}
        assert set(rep["deltas_vs_B"]) == {"A", "C"}
        assert "bootstrap_ci" in rep["nominal"]
        assert set(rep["nominal"]["bootstrap_ci"]) == {
            "recall",
            "precision",
            "fbeta",
            "roc_auc",
        }

    def test_predictions_file_aligns_with_report(
        self, eval_multiroot, random_checkpoint, tmp_path
    ):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        preds = json.loads((tmp_path / "predictions.json").read_text())
        assert preds["total"] == rep["test_composition"]["images"]
        assert {p["profile"] for p in preds["records"]} <= {"A", "B", "C"}
        assert all(p["agc_mode"] == "fixed" for p in preds["records"])


class TestDataPath:
    def test_evaluates_only_the_test_split(
        self, eval_multiroot, random_checkpoint, tmp_path
    ):
        from nina_thermal.training.dataset import (
            expand_to_samples,
            load_merged_records,
            split_samples,
        )
        from nina_thermal.training.frp_labels import apply_frp_labels

        ckpt, ckpt_dir = random_checkpoint
        _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        records, _ = load_merged_records(eval_multiroot.roots)
        samples = expand_to_samples(records, ["A", "B", "C"], ["fixed"])
        samples, _ = apply_frp_labels(samples, eval_multiroot.roots, 20000.0)
        test_granules = {
            s["granule_id"] for s in split_samples(samples).get("test", [])
        }
        preds = json.loads((tmp_path / "predictions.json").read_text())["records"]
        assert {p["granule_id"] for p in preds} == test_granules

    def test_non_degenerate_on_the_tuned_fixture(
        self, eval_multiroot, random_checkpoint, tmp_path
    ):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        assert rep["degenerate_eval"] is False
        assert rep["test_composition"]["nominal_fire"] >= 20
        assert rep["nominal"]["roc_auc"] is not None
        ci = rep["nominal"]["bootstrap_ci"]["roc_auc"]
        assert ci["ci_low"] is not None and ci["ci_low"] <= ci["ci_high"]


class TestThresholdResolution:
    def test_reads_selected_threshold_from_report(
        self, eval_multiroot, random_checkpoint, tmp_path
    ):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        assert rep["operating_threshold"] == 0.45
        assert rep["threshold_source"] == "training_report"
        assert rep["beta"] == round(3.0**0.5, 6)

    def test_explicit_threshold_wins(self, eval_multiroot, random_checkpoint, tmp_path):
        ckpt, _ = random_checkpoint
        rep = _report(eval_multiroot.roots, ckpt, tmp_path, explicit_threshold=0.8)
        assert rep["operating_threshold"] == 0.8
        assert rep["threshold_source"] == "explicit"

    def test_no_report_sweeps_and_flags(self, eval_multiroot, tmp_path):
        ckpt = tmp_path / "bare.pt"
        from nina_thermal.training import train_thermal

        torch.save(train_thermal.build_model(pretrained=False).state_dict(), ckpt)
        rep = _report(eval_multiroot.roots, ckpt, tmp_path / "out")
        assert rep["threshold_source"] == "swept_on_test"


class TestDegenerateGuard:
    def test_small_fixture_is_flagged(self, multiroot, random_checkpoint, tmp_path):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        # the small `multiroot` fixture cannot supply 20 fire patches in profile B
        assert rep["degenerate_eval"] is True


class TestFigures:
    def test_writes_one_png_per_slice(
        self, eval_multiroot, random_checkpoint, tmp_path
    ):
        ckpt, ckpt_dir = random_checkpoint
        rep = _report(
            eval_multiroot.roots,
            ckpt,
            tmp_path,
            training_report=ckpt_dir / "training_report.json",
        )
        preds = json.loads((tmp_path / "predictions.json").read_text())["records"]
        evaluate.save_figures(rep, preds, tmp_path)
        pngs = {p.name for p in (tmp_path / "figures").glob("*.png")}
        assert "eval_nominal_profile_b.png" in pngs
        assert {
            "eval_profile_a.png",
            "eval_profile_b.png",
            "eval_profile_c.png",
        } <= pngs
