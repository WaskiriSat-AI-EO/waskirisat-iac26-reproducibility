"""Tests for the on-disk naming helpers in dataset/export.py.

These are the single source of truth for how a
(granule_id, patch_index, agc_mode, profile) tuple maps to a file, so every
pipeline stage and the training/eval code depends on them agreeing.
"""

from pathlib import Path

from nina_thermal.datagen.dataset import export


def test_patch_key_zero_pads_the_index():
    assert export.patch_key("S3A_g001", 7) == "S3A_g001_p0007"
    assert export.patch_key("g", 12345) == "g_p12345"


def test_image_stem_and_filenames_compose():
    key = export.patch_key("g001", 3)
    stem = export.image_stem(key, "fixed", "B")
    assert stem == "g001_p0003_fixed_B"
    assert export.image_filename(key, "fixed", "B") == f"{stem}.png"
    assert export.metadata_filename(key, "fixed", "B") == f"{stem}.json"


def test_metadata_path_for_image_swaps_images_for_metadata():
    img = Path("/data/root/04_dataset/images/g001_p0003_fixed_B.png")
    meta = export.metadata_path_for_image(img)
    assert meta == Path("/data/root/04_dataset/metadata/g001_p0003_fixed_B.json")


def test_metadata_path_round_trips_the_stem():
    key = export.patch_key("SIM_g042", 1)
    fname = export.image_filename(key, "global", "C")
    img = Path("x/y/04_dataset/images") / fname
    meta = export.metadata_path_for_image(img)
    assert meta.name == export.metadata_filename(key, "global", "C")
    assert meta.parent.name == "metadata"
