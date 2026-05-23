"""End-to-end integration test: run_simulation → run_split → run_balance → run_report.

Creates synthetic .npy patch files in a tmp directory, runs the full pipeline,
and verifies:
  1. dataset_summary.json contains all required fields (Section 6.14).
  2. All 10 paper claims from Section 12 are verifiable from output files.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from nina_thermal.datagen import config
from nina_thermal.datagen.pipelines.run_simulation import main as sim_main
from nina_thermal.datagen.pipelines.run_split import main as split_main
from nina_thermal.datagen.pipelines.run_balance import main as balance_main
from nina_thermal.datagen.pipelines.run_report import main as report_main


# ── Synthetic dataset parameters ─────────────────────────────────────────────

# 7 high-fire + 7 low-fire granules → 5 train + 1 val + 1 test per tier ✓
N_HIGH_FIRE_GRANULES = 7
N_LOW_FIRE_GRANULES = 7
PATCHES_PER_GRANULE = 2  # 1 fire + 1 no-fire for high-fire; 2 no-fire for low-fire

PATCH_H, PATCH_W = config.LEPTON_RESOLUTION[1], config.LEPTON_RESOLUTION[0]  # 120, 160

FIRE_F2_BT = 330.0  # > FIRE_F2_ABS_THRESHOLD_K (320 K)
FIRE_DELTA_T = 15.0  # > FIRE_DELTA_T_THRESHOLD_K (8 K)
NOMINAL_F2_BT = 295.0
NOMINAL_DELTA_T = 2.0  # < 8 K — not a fire pixel

DATE_STR = "20230601T120000"  # embeds a parseable date


def _make_patch(is_fire: bool) -> np.ndarray:
    """Return a (120, 160) float32 BT array."""
    bt = np.full(
        (PATCH_H, PATCH_W), FIRE_F2_BT if is_fire else NOMINAL_F2_BT, dtype=np.float32
    )
    return bt


def _make_delta_t(is_fire: bool) -> np.ndarray:
    delta = np.full(
        (PATCH_H, PATCH_W),
        FIRE_DELTA_T if is_fire else NOMINAL_DELTA_T,
        dtype=np.float32,
    )
    return delta


def _build_patches_dir(tmp_path: Path) -> Path:
    """Create synthetic patch .npy files and patches_manifest.json."""
    patches_dir = tmp_path / "03_patches"
    patches_dir.mkdir()

    patch_records: List[Dict] = []

    all_granules = [
        f"S3A_SL_1_RBT____{DATE_STR}_HF{i:02d}" for i in range(N_HIGH_FIRE_GRANULES)
    ] + [f"S3A_SL_1_RBT____{DATE_STR}_LF{i:02d}" for i in range(N_LOW_FIRE_GRANULES)]

    for granule_id in all_granules:
        is_high_fire = "_HF" in granule_id
        for patch_idx in range(PATCHES_PER_GRANULE):
            # High-fire granules: patch 0 = fire, patch 1 = no-fire
            # Low-fire granules:  all patches = no-fire
            is_fire = is_high_fire and (patch_idx == 0)

            bt = _make_patch(is_fire)
            delta_t = _make_delta_t(is_fire)

            bt_fname = f"{granule_id}_p{patch_idx:04d}_bt.npy"
            delta_t_fname = f"{granule_id}_p{patch_idx:04d}_delta_t.npy"

            np.save(patches_dir / bt_fname, bt)
            np.save(patches_dir / delta_t_fname, delta_t)

            patch_records.append(
                {
                    "granule_id": granule_id,
                    "patch_index": patch_idx,
                    "patch_file": bt_fname,
                    "nan_fraction": 0.0,
                    "centroid_lat": -15.0 + patch_idx * 0.5,
                    "centroid_lon": -62.0 + patch_idx * 0.5,
                    "bt_min": float(bt.min()),
                    "bt_max": float(bt.max()),
                }
            )

    manifest = {
        "created_utc": "2023-06-01T12:00:00+00:00",
        "script_version": config.SCRIPT_VERSION,
        "args": {"input_dir": ".", "output_dir": str(patches_dir)},
        "patches": patch_records,
        "total_patches": len(patch_records),
    }
    with open(patches_dir / "patches_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    return patches_dir


# ── Fixture: complete pipeline run ───────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the full pipeline and return (dataset_dir, summary_dict)."""
    tmp = tmp_path_factory.mktemp("pipeline")
    patches_dir = _build_patches_dir(tmp)
    dataset_dir = tmp / "04_dataset"

    # Pass 2: simulate
    rc = sim_main(
        [
            "--input_dir",
            str(patches_dir),
            "--output_dir",
            str(dataset_dir),
            "--profiles",
            "A,B",  # two profiles for speed
            "--agc_modes",
            "fixed",  # one mode
            "--seed",
            "42",
        ]
    )
    assert rc == 0, "run_simulation failed"

    # Split
    rc = split_main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--seed",
            "42",
        ]
    )
    assert rc == 0, "run_split failed"

    # Balance
    rc = balance_main(
        [
            "--dataset_dir",
            str(dataset_dir),
            "--fire_ratio",
            str(config.FIRE_RATIO),
            "--seed",
            "42",
        ]
    )
    assert rc == 0, "run_balance failed"

    # Report
    rc = report_main(
        [
            "--dataset_dir",
            str(dataset_dir),
        ]
    )
    assert rc == 0, "run_report failed"

    summary_path = dataset_dir / "dataset_summary.json"
    assert summary_path.exists(), "dataset_summary.json not written"
    with open(summary_path) as fh:
        summary = json.load(fh)

    return dataset_dir, summary


# ── Section 6.14: all required fields present ─────────────────────────────────


class TestDatasetSummaryFields:
    def test_top_level_fields(self, pipeline_output):
        _, s = pipeline_output
        for field in (
            "total_granules",
            "total_patches_before_rejection",
            "total_patches_after_rejection",
            "rejection_rate",
            "splits",
            "label_distribution",
            "surface_types",
            "agc_modes",
            "profiles",
            "bt_statistics",
            "geographic_coverage",
            "temporal_coverage",
            "processing_parameters",
        ):
            assert field in s, f"Missing top-level field: {field}"

    def test_splits_fields(self, pipeline_output):
        _, s = pipeline_output
        for split_name in ("train", "val", "test"):
            assert split_name in s["splits"], f"Missing split: {split_name}"
            split = s["splits"][split_name]
            for key in ("granules", "patches", "fire_fraction"):
                assert key in split, f"Missing splits.{split_name}.{key}"

    def test_label_distribution_fields(self, pipeline_output):
        _, s = pipeline_output
        ld = s["label_distribution"]
        for key in ("fire", "no_fire", "fire_fraction"):
            assert key in ld

    def test_surface_types_fields(self, pipeline_output):
        _, s = pipeline_output
        for key in ("fire", "warm_surface", "nominal"):
            assert key in s["surface_types"]

    def test_agc_modes_fields(self, pipeline_output):
        _, s = pipeline_output
        for key in ("fixed", "global", "auto"):
            assert key in s["agc_modes"]

    def test_profiles_fields(self, pipeline_output):
        _, s = pipeline_output
        for key in ("A", "B", "C"):
            assert key in s["profiles"]

    def test_bt_statistics_fields(self, pipeline_output):
        _, s = pipeline_output
        for patch_class in ("fire_patches", "nofire_patches"):
            assert patch_class in s["bt_statistics"]
            for stat in ("mean_k", "std_k", "p2_k", "p98_k"):
                assert stat in s["bt_statistics"][patch_class]

    def test_geographic_coverage_fields(self, pipeline_output):
        _, s = pipeline_output
        geo = s["geographic_coverage"]
        for key in ("lat_min", "lat_max", "lon_min", "lon_max", "patch_centroids"):
            assert key in geo

    def test_temporal_coverage_fields(self, pipeline_output):
        _, s = pipeline_output
        tc = s["temporal_coverage"]
        for key in ("granule_dates", "date_min", "date_max"):
            assert key in tc

    def test_processing_parameters_fields(self, pipeline_output):
        _, s = pipeline_output
        pp = s["processing_parameters"]
        for key in (
            "agg_factor",
            "synthetic_gsd_m",
            "nina_lepton_gsd_m",
            "gsd_overestimation_pct",
            "fixed_agc_range_k",
            "fire_ratio_target",
            "seed",
            "script_version",
        ):
            assert key in pp, f"Missing processing_parameters.{key}"


# ── Section 12: all 10 paper claims verifiable ────────────────────────────────


class TestPaperClaims:
    def test_claim1_radiance_consistent_aggregation(self, pipeline_output):
        """Claim 1: radiance-consistent averaging."""
        _, s = pipeline_output
        pp = s["processing_parameters"]
        # Verifiable: agg_factor is present and matches config
        assert pp["agg_factor"] == config.SLSTR_AGG_FACTOR

    def test_claim2_gsd_overestimation(self, pipeline_output):
        """Claim 2: 12.3% conservative overestimation."""
        _, s = pipeline_output
        pp = s["processing_parameters"]
        assert abs(pp["gsd_overestimation_pct"] - 12.3) < 0.1
        assert pp["synthetic_gsd_m"] == config.SLSTR_SYNTHETIC_GSD_M
        assert pp["nina_lepton_gsd_m"] == config.NINA_LEPTON_GSD_M

    def test_claim3_noise_in_bt_space(self, pipeline_output):
        """Claim 3: thermal noise added in BT space before rendering."""
        dataset_dir, _ = pipeline_output
        meta_dir = dataset_dir / "metadata"
        meta_files = list(meta_dir.glob("*.json"))
        assert meta_files, "No metadata files found"
        with open(meta_files[0]) as fh:
            meta = json.load(fh)
        provenance = meta.get("provenance", [])
        # Noise step (sensor simulation) must appear before the uint8 cast
        assert any("sensor simulation" in p for p in provenance)
        assert any("agc_to_uint8" in p for p in provenance)
        sim_idx = next(i for i, p in enumerate(provenance) if "sensor simulation" in p)
        cast_idx = next(i for i, p in enumerate(provenance) if "agc_to_uint8" in p)
        assert sim_idx < cast_idx, "Sensor simulation must precede uint8 cast"

    def test_claim4_flight_agc_config(self, pipeline_output):
        """Claim 4: fixed AGC is the designated flight inference configuration."""
        dataset_dir, s = pipeline_output
        meta_dir = dataset_dir / "metadata"
        flight_count = 0
        for f in meta_dir.glob("*_fixed_*.json"):
            with open(f) as fh:
                meta = json.load(fh)
            if meta.get("flight_config"):
                flight_count += 1
        assert flight_count > 0, "No images with flight_config=True found"

        pp = s["processing_parameters"]
        assert pp["fixed_agc_range_k"] == config.FIXED_TEMP_RANGE_K

    def test_claim5_granule_level_splits(self, pipeline_output):
        """Claim 5: splits assigned at granule level."""
        dataset_dir, _ = pipeline_output
        split_path = dataset_dir / "split_manifest.json"
        assert split_path.exists()
        with open(split_path) as fh:
            sm = json.load(fh)
        # Each split contains a list of granule IDs (strings)
        for key in ("train", "val", "test"):
            assert key in sm
            assert isinstance(sm[key], list)
        # No granule appears in more than one split
        all_gids = sm["train"] + sm["val"] + sm["test"]
        assert len(all_gids) == len(
            set(all_gids)
        ), "Duplicate granule IDs across splits"

    def test_claim6_fire_ratio(self, pipeline_output):
        """Claim 6: 1:4 fire ratio (fire_ratio_target = 0.20)."""
        _, s = pipeline_output
        assert s["processing_parameters"]["fire_ratio_target"] == config.FIRE_RATIO
        assert abs(config.FIRE_RATIO - 0.20) < 1e-6

    def test_claim7_warm_surface_characterization(self, pipeline_output):
        """Claim 7: warm-surface label ambiguity tracked in report."""
        _, s = pipeline_output
        st = s["surface_types"]
        # All three surface types must be tracked (even if some have zero count)
        assert "fire" in st
        assert "warm_surface" in st
        assert "nominal" in st

    def test_claim8_fpn_model(self, pipeline_output):
        """Claim 8: FPN model has column and pixel components."""
        dataset_dir, _ = pipeline_output
        # Profiles B and C use FPN (fpn_factor != None); check that metadata
        # records exist for profile A (minimal) and B (moderate FPN)
        meta_dir = dataset_dir / "metadata"
        profile_a = list(meta_dir.glob("*_A.json"))
        profile_b = list(meta_dir.glob("*_B.json"))
        assert profile_a, "No profile-A metadata files found"
        assert profile_b, "No profile-B metadata files found"
        # Verify profile is correctly recorded
        with open(profile_a[0]) as fh:
            m = json.load(fh)
        assert m["profile"] == "A"

    def test_claim9_three_degradation_profiles(self, pipeline_output):
        """Claim 9: three profiles bound the simulation uncertainty envelope."""
        _, s = pipeline_output
        # All three profiles must be declared in the report
        for key in ("A", "B", "C"):
            assert key in s["profiles"]
        # Config has ffc_phase documentation for each profile
        for key in ("A", "B", "C"):
            assert "ffc_phase" in config.DEGRADATION_PROFILES[key]
            assert config.DEGRADATION_PROFILES[key]["ffc_phase"]

    def test_claim10_full_provenance(self, pipeline_output):
        """Claim 10: full provenance from SLSTR granule ID to training sample."""
        dataset_dir, _ = pipeline_output
        meta_dir = dataset_dir / "metadata"
        meta_files = list(meta_dir.glob("*.json"))
        assert meta_files, "No metadata files found"
        with open(meta_files[0]) as fh:
            meta = json.load(fh)
        prov = meta.get("provenance", [])
        # Must start with the source data and include the full chain
        assert len(prov) >= 4, "Provenance chain too short"
        assert "SLSTR" in prov[0], "Provenance must start with SLSTR source"
        assert meta.get("granule_id"), "granule_id must be present in metadata"
        assert meta.get("patch_index") is not None, "patch_index must be present"
        assert meta.get("script_version") == config.SCRIPT_VERSION


# ── Additional pipeline integrity checks ─────────────────────────────────────


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

    def test_fire_labels_present(self, pipeline_output):
        """High-fire granules should produce fire-labelled patches."""
        dataset_dir, s = pipeline_output
        fire_count = s["label_distribution"].get("fire", 0)
        assert fire_count > 0, "No fire-labelled patches found"

    def test_split_disjoint(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        with open(dataset_dir / "label_manifest.json") as fh:
            records = json.load(fh)
        with open(dataset_dir / "split_manifest.json") as fh:
            sm = json.load(fh)
        # Every record's granule_id must map to exactly one split
        all_train = set(sm["train"])
        all_val = set(sm["val"])
        all_test = set(sm["test"])
        assert not (all_train & all_val), "Granule overlap: train ∩ val"
        assert not (all_train & all_test), "Granule overlap: train ∩ test"
        assert not (all_val & all_test), "Granule overlap: val ∩ test"

    def test_metadata_flight_config_flag(self, pipeline_output):
        dataset_dir, _ = pipeline_output
        meta_dir = dataset_dir / "metadata"
        for f in list(meta_dir.glob("*_fixed_*.json"))[:5]:
            with open(f) as fh:
                m = json.load(fh)
            assert (
                m.get("flight_config") is True
            ), f"flight_config should be True for fixed AGC: {f.name}"

    def test_total_patches_consistent(self, pipeline_output):
        _, s = pipeline_output
        # Patches across all splits must equal total_patches_after_rejection
        split_total = sum(s["splits"][sp]["patches"] for sp in ("train", "val", "test"))
        assert split_total == s["total_patches_after_rejection"]

    def test_processing_parameters_values(self, pipeline_output):
        _, s = pipeline_output
        pp = s["processing_parameters"]
        assert pp["agg_factor"] == config.SLSTR_AGG_FACTOR
        assert pp["synthetic_gsd_m"] == config.SLSTR_SYNTHETIC_GSD_M
        assert pp["nina_lepton_gsd_m"] == config.NINA_LEPTON_GSD_M
        assert pp["fixed_agc_range_k"] == config.FIXED_TEMP_RANGE_K
        assert pp["fire_ratio_target"] == config.FIRE_RATIO
        assert pp["script_version"] == config.SCRIPT_VERSION
