"""End-to-end integration test: run_simulation → run_split → run_balance → run_report.

Creates synthetic tile .npy files in a tmp directory, runs the full pipeline,
and verifies:
  1. All pipeline stages return exit code 0 and produce expected outputs.
  2. Simulation output is bit-reproducible (determinism fix guard).
  3. Manifest chain validation fails fast when tile files are missing.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from nina_cloud.datagen.pipelines.run_simulation import main as sim_main
from nina_cloud.datagen.pipelines.run_split import main as split_main
from nina_cloud.datagen.pipelines.run_balance import main as balance_main
from nina_cloud.datagen.pipelines.run_report import main as report_main


# ── Synthetic dataset parameters ─────────────────────────────────────────────

# 7 cloud-heavy + 7 clear-heavy scenes → 5 train + 1 val + 1 test per tier ✓
N_CLOUD_SCENES = 7
N_CLEAR_SCENES = 7
TILES_PER_SCENE = 2  # 1 cloud + 1 clear per cloud-heavy; 2 clear per clear-heavy

# Small tile dimensions so the test runs quickly
TILE_H, TILE_W = 64, 64
SCL_H, SCL_W = TILE_H // 2, TILE_W // 2

# SCL pixel values used to force definitive labels
SCL_CLOUD_VALUE = 9  # "high probability cloud" — in config.SCL_CLOUD_CLASSES
SCL_CLEAR_VALUE = 4  # "vegetation" — not a cloud class

DATE_STR = "20230601T120000"


def _make_rgb(cloud: bool) -> np.ndarray:
    """Return a (H, W, 3) float32 RGB tile in [0, 1]."""
    fill = 0.8 if cloud else 0.2
    return np.full((TILE_H, TILE_W, 3), fill, dtype=np.float32)


def _make_scl(cloud: bool) -> np.ndarray:
    """Return a (H/2, W/2) uint8 SCL tile."""
    fill = SCL_CLOUD_VALUE if cloud else SCL_CLEAR_VALUE
    return np.full((SCL_H, SCL_W), fill, dtype=np.uint8)


def _build_tiles_dir(tmp_path: Path) -> Path:
    """Write synthetic tile .npy files and tiles_manifest.json."""
    tiles_dir = tmp_path / "03_tiles"
    tiles_dir.mkdir()

    tile_records: List[Dict] = []

    all_scenes = [f"S2A_MSIL2A_{DATE_STR}_CS{i:02d}" for i in range(N_CLOUD_SCENES)] + [
        f"S2A_MSIL2A_{DATE_STR}_CL{i:02d}" for i in range(N_CLEAR_SCENES)
    ]

    for scene_id in all_scenes:
        is_cloud_heavy = "_CS" in scene_id
        for tile_idx in range(TILES_PER_SCENE):
            # Cloud-heavy scenes: tile 0 = cloud, tile 1 = clear
            # Clear-heavy scenes: all tiles = clear
            is_cloud = is_cloud_heavy and (tile_idx == 0)

            rgb = _make_rgb(is_cloud)
            scl = _make_scl(is_cloud)

            stem = f"{scene_id}_tile{tile_idx:04d}"
            np.save(tiles_dir / f"{stem}_rgb.npy", rgb)
            np.save(tiles_dir / f"{stem}_scl.npy", scl)

            tile_records.append(
                {
                    "scene_id": scene_id,
                    "tile_idx": tile_idx,
                    "label": 1 if is_cloud else 0,
                    "tile_type": "cloud" if is_cloud else "clear",
                    "cloud_frac": 0.90 if is_cloud else 0.05,
                    "nodata_frac": 0.0,
                    "origin_row": tile_idx * TILE_H,
                    "origin_col": 0,
                }
            )

    manifest = {
        "stage": "run_processing",
        "generated_at": "2023-06-01T12:00:00+00:00",
        "n_cloud": sum(1 for r in tile_records if r["label"] == 1),
        "n_clear": sum(1 for r in tile_records if r["label"] == 0),
        "n_ambig_rejected": 0,
        "tiles": tile_records,
    }
    with open(tiles_dir / "tiles_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return tiles_dir


# ── Fixture: complete pipeline run ───────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the full pipeline and return (dataset_dir, summary_dict)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    log_dir = str(tmp / "logs")
    tiles_dir = _build_tiles_dir(tmp)
    dataset_dir = tmp / "04_dataset"

    rc = sim_main(
        [
            "--input_dir",
            str(tiles_dir),
            "--output_dir",
            str(dataset_dir),
            "--profiles",
            "A",
            "--seed",
            "42",
            "--log_dir",
            log_dir,
        ]
    )
    assert rc == 0, "run_simulation failed"

    rc = split_main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--seed",
            "42",
            "--log_dir",
            log_dir,
        ]
    )
    assert rc == 0, "run_split failed"

    rc = balance_main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--seed",
            "42",
            "--log_dir",
            log_dir,
        ]
    )
    assert rc == 0, "run_balance failed"

    rc = report_main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--log_dir",
            log_dir,
        ]
    )
    assert rc == 0, "run_report failed"

    summary_path = dataset_dir / "dataset_summary.json"
    assert summary_path.exists(), "dataset_summary.json not written"
    with open(summary_path) as fh:
        summary = json.load(fh)

    return dataset_dir, summary


# ── Pipeline integrity ────────────────────────────────────────────────────────


class TestPipelineIntegrity:
    def test_images_written(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        pngs = list((dataset_dir / "images").glob("*.png"))
        assert pngs, "No PNG images written"

    def test_metadata_written(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        jsons = list((dataset_dir / "metadata").glob("*.json"))
        assert jsons, "No metadata JSON files written"

    def test_label_manifest_has_splits(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        with open(dataset_dir / "label_manifest.json") as fh:
            records = json.load(fh)
        assert records, "label_manifest.json is empty"
        for rec in records:
            assert rec.get("split") in (
                "train",
                "val",
                "test",
            ), f"Invalid split value: {rec.get('split')!r}"

    def test_balanced_manifest_exists(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        assert (dataset_dir / "balanced_manifest.json").exists()

    def test_summary_top_level_fields(self, pipeline_output):
        _, s = pipeline_output
        for field in (
            "total_images",
            "splits",
            "script_version",
            "sensor",
            "label_thresholds",
            "tile_size_px",
        ):
            assert field in s, f"Missing field in dataset_summary.json: {field}"

    def test_summary_splits_have_data(self, pipeline_output):
        _, s = pipeline_output
        for split_name in ("train", "val", "test"):
            assert split_name in s["splits"], f"Missing split: {split_name}"
            assert (
                s["splits"][split_name]["n_total"] > 0
            ), f"Split {split_name!r} has 0 images"

    def test_cloud_labels_present(self, pipeline_output):
        dataset_dir, s = pipeline_output
        total_cloud = sum(s["splits"][sp]["n_cloud"] for sp in ("train", "val", "test"))
        assert total_cloud > 0, "No cloud-labelled images found"

    def test_split_disjoint_at_scene_level(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        with open(dataset_dir / "label_manifest.json") as fh:
            records = json.load(fh)
        scene_split: Dict[str, str] = {}
        for rec in records:
            sid = rec["scene_id"]
            sp = rec["split"]
            if sid in scene_split:
                assert scene_split[sid] == sp, (
                    f"Scene {sid} appears in multiple splits: "
                    f"{scene_split[sid]!r} and {sp!r}"
                )
            scene_split[sid] = sp


# ── Determinism guard (directly tests the FPN + noise seeding fix) ────────────


class TestDeterminism:
    def test_identical_output_same_seed(self, tmp_path_factory):
        """Two runs with the same seed on the same tiles produce identical PNGs."""
        tmp = tmp_path_factory.mktemp("determinism")
        log_dir = str(tmp / "logs")
        tiles_dir = _build_tiles_dir(tmp)

        out_a = tmp / "run_a"
        out_b = tmp / "run_b"

        for out in (out_a, out_b):
            rc = sim_main(
                [
                    "--input_dir",
                    str(tiles_dir),
                    "--output_dir",
                    str(out),
                    "--profiles",
                    "A,B",
                    "--seed",
                    "7",
                    "--log_dir",
                    log_dir,
                ]
            )
            assert rc == 0

        pngs_a = sorted((out_a / "images").glob("*.png"))
        pngs_b = sorted((out_b / "images").glob("*.png"))
        assert pngs_a, "No images produced in run A"
        assert [p.name for p in pngs_a] == [
            p.name for p in pngs_b
        ], "Output file names differ between runs"
        for pa, pb in zip(pngs_a, pngs_b):
            assert (
                pa.read_bytes() == pb.read_bytes()
            ), f"Image {pa.name} differs between runs — simulation is not reproducible"

    def test_order_independent(self, tmp_path_factory):
        """Reversing the profile list produces the same per-tile images."""
        tmp = tmp_path_factory.mktemp("order")
        log_dir = str(tmp / "logs")
        tiles_dir = _build_tiles_dir(tmp)

        out_ab = tmp / "profiles_AB"
        out_ba = tmp / "profiles_BA"

        for out, profiles in ((out_ab, "A,B"), (out_ba, "B,A")):
            rc = sim_main(
                [
                    "--input_dir",
                    str(tiles_dir),
                    "--output_dir",
                    str(out),
                    "--profiles",
                    profiles,
                    "--seed",
                    "7",
                    "--log_dir",
                    log_dir,
                ]
            )
            assert rc == 0

        for name in [p.name for p in (out_ab / "images").glob("*.png")]:
            img_ab = (out_ab / "images" / name).read_bytes()
            img_ba = (out_ba / "images" / name).read_bytes()
            assert img_ab == img_ba, (
                f"Image {name} differs when profile order is reversed — "
                "seeding is still order-dependent"
            )


# ── Chain fail-fast guard ─────────────────────────────────────────────────────


class TestChainValidation:
    def test_fails_fast_on_missing_tile_files(self, tmp_path_factory):
        """run_simulation exits 1 when a manifest references missing .npy files."""
        tmp = tmp_path_factory.mktemp("chain")
        log_dir = str(tmp / "logs")
        tiles_dir = _build_tiles_dir(tmp)

        # Remove one scene's tile files to simulate upstream data loss
        orphan_scene = f"S2A_MSIL2A_{DATE_STR}_CS00"
        for npy in tiles_dir.glob(f"{orphan_scene}_tile*_rgb.npy"):
            npy.unlink()

        dataset_dir = tmp / "04_dataset_broken"
        rc = sim_main(
            [
                "--input_dir",
                str(tiles_dir),
                "--output_dir",
                str(dataset_dir),
                "--profiles",
                "A",
                "--seed",
                "42",
                "--log_dir",
                log_dir,
            ]
        )
        assert (
            rc == 1
        ), "run_simulation should exit 1 when a scene's tile files are missing"
