"""Tests for run_split.py's split_manifest.json reuse/staleness handling.

Covers the fix for two bugs found in review: a stale split_manifest.json
(one whose granule set no longer matches the current label_manifest.json)
used to either silently mislabel new granules as split="train" (no error),
or raise KeyError when a granule referenced by the stale manifest had been
removed. Both cases must now fall back to a fresh, correct split instead.
"""

import json
from pathlib import Path

import pytest

from nina_thermal.datagen.pipelines.run_split import main as split_main


def _write_label_manifest(dataset_dir: Path, granule_ids) -> None:
    records = [
        {
            "patch": f"{gid}_p0000",
            "label": 0,
            "granule_id": gid,
            "surface_type": "nofire",
            "split": "",
        }
        for gid in granule_ids
    ]
    with open(dataset_dir / "label_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh)


def _write_split_manifest(dataset_dir: Path, train, val, test) -> None:
    with open(dataset_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"train": train, "val": val, "test": test}, fh)


def _read_label_splits(dataset_dir: Path) -> dict:
    with open(dataset_dir / "label_manifest.json", encoding="utf-8") as fh:
        records = json.load(fh)
    return {rec["granule_id"]: rec["split"] for rec in records}


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    (tmp_path / "metadata").mkdir()
    return tmp_path


class TestReuseWhenManifestMatches:
    def test_matching_split_manifest_is_reused_verbatim(self, dataset_dir):
        _write_label_manifest(dataset_dir, ["g1", "g2", "g3"])
        _write_split_manifest(dataset_dir, train=["g1", "g2"], val=["g3"], test=[])

        # A different seed must have NO effect when the manifest is reused —
        # if the split were recomputed instead, a different seed could
        # reassign granules.
        rc = split_main(["--dataset_dir", str(dataset_dir), "--seed", "999"])
        assert rc == 0

        splits = _read_label_splits(dataset_dir)
        assert splits == {"g1": "train", "g2": "train", "g3": "val"}


class TestStaleManifestFallback:
    def test_new_granule_not_silently_defaulted_to_train(self, dataset_dir):
        """A granule added since split_manifest.json was written must not be
        silently stamped split='train' — the whole split must be recomputed
        so the granule_level_split coverage/disjoint guarantee still holds."""
        _write_label_manifest(dataset_dir, ["g1", "g2", "g3"])
        # Stale manifest only knows about g1, g2 (g3 is new)
        _write_split_manifest(dataset_dir, train=["g1"], val=["g2"], test=[])

        rc = split_main(["--dataset_dir", str(dataset_dir), "--seed", "42"])
        assert rc == 0

        with open(dataset_dir / "split_manifest.json", encoding="utf-8") as fh:
            rewritten = json.load(fh)
        all_granules = set(rewritten["train"]) | set(rewritten["val"]) | set(
            rewritten["test"]
        )
        assert all_granules == {"g1", "g2", "g3"}, (
            "g3 must appear in the rewritten split_manifest.json, not just "
            "get silently stamped 'train' in label_manifest.json"
        )

    def test_removed_granule_does_not_crash(self, dataset_dir):
        """A granule removed since split_manifest.json was written used to
        raise KeyError (granule_labels[gid] lookup on a granule no longer in
        label_manifest.json). It must now trigger a safe recompute instead."""
        _write_label_manifest(dataset_dir, ["g1", "g2"])
        # Stale manifest still references g3, which no longer exists
        _write_split_manifest(dataset_dir, train=["g1", "g3"], val=["g2"], test=[])

        rc = split_main(["--dataset_dir", str(dataset_dir), "--seed", "42"])
        assert rc == 0

        splits = _read_label_splits(dataset_dir)
        assert set(splits) == {"g1", "g2"}

    def test_stale_manifest_recompute_still_covers_all_granules(self, dataset_dir):
        _write_label_manifest(dataset_dir, ["g1", "g2", "g3", "g4"])
        _write_split_manifest(dataset_dir, train=["g1"], val=[], test=[])

        rc = split_main(["--dataset_dir", str(dataset_dir), "--seed", "42"])
        assert rc == 0

        splits = _read_label_splits(dataset_dir)
        assert set(splits) == {"g1", "g2", "g3", "g4"}
        assert all(v in ("train", "val", "test") for v in splits.values())
