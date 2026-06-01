"""Tests for shared/scene_registry.py."""

import json

from shared.scene_registry import (
    assign_split,
    is_registered,
    load_registry,
    operator_name,
    register_scene,
    save_registry,
    split_for,
)


# ── load / save ───────────────────────────────────────────────────────────────


class TestLoadSave:
    def test_load_missing_returns_empty(self, tmp_path):
        registry = load_registry(tmp_path / "nonexistent.json")
        assert registry == {}

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "registry.json"
        data = {"SCENE_001": {"split": "train", "status": "acquired"}}
        save_registry(path, data)
        loaded = load_registry(path)
        assert loaded == data

    def test_atomic_write_creates_parent(self, tmp_path):
        path = tmp_path / "subdir" / "registry.json"
        save_registry(path, {"X": {"split": "test"}})
        assert path.exists()

    def test_save_overwrites(self, tmp_path):
        path = tmp_path / "registry.json"
        save_registry(path, {"A": {"split": "train"}})
        save_registry(path, {"B": {"split": "val"}})
        loaded = load_registry(path)
        assert "A" not in loaded
        assert loaded["B"]["split"] == "val"

    def test_valid_json_on_disk(self, tmp_path):
        path = tmp_path / "registry.json"
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="tester")
        save_registry(path, registry)
        with open(path) as fh:
            raw = json.load(fh)
        assert "S001" in raw


# ── register_scene ────────────────────────────────────────────────────────────


class TestRegisterScene:
    def test_new_scene_returns_true(self):
        registry: dict = {}
        result = register_scene(registry, "S001", cloud_mode="clear", operator="op1")
        assert result is True

    def test_existing_scene_returns_false(self):
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="op1")
        result = register_scene(registry, "S001", cloud_mode="clear", operator="op1")
        assert result is False

    def test_split_assigned_on_first_call(self):
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="op1")
        assert registry["S001"]["split"] in ("train", "val", "test")

    def test_split_never_overwritten(self):
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="op1")
        original_split = registry["S001"]["split"]
        # Second call with different operator
        register_scene(registry, "S001", cloud_mode="cloudy", operator="op2")
        assert registry["S001"]["split"] == original_split

    def test_mutable_fields_updated_on_second_call(self):
        registry: dict = {}
        register_scene(
            registry, "S001", cloud_mode="clear", operator="op1", status="acquired"
        )
        register_scene(
            registry, "S001", cloud_mode="cloudy", operator="op2", status="complete"
        )
        assert registry["S001"]["operator"] == "op2"
        assert registry["S001"]["status"] == "complete"
        assert registry["S001"]["cloud_mode"] == "cloudy"

    def test_provenance_fields_present(self):
        registry: dict = {}
        register_scene(registry, "S002", cloud_mode="cloudy", operator="alice")
        rec = registry["S002"]
        assert "registered_at" in rec
        assert "last_seen_at" in rec
        assert "operator" in rec
        assert "cloud_mode" in rec
        assert "status" in rec

    def test_default_status_is_acquired(self):
        registry: dict = {}
        register_scene(registry, "S003", cloud_mode="clear", operator="bob")
        assert registry["S003"]["status"] == "acquired"


# ── is_registered ─────────────────────────────────────────────────────────────


class TestIsRegistered:
    def test_false_for_unknown_scene(self):
        assert is_registered({}, "S001") is False

    def test_true_after_register(self):
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="op")
        assert is_registered(registry, "S001") is True


# ── split_for ─────────────────────────────────────────────────────────────────


class TestSplitFor:
    def test_uses_registry_value(self):
        registry: dict = {}
        register_scene(registry, "S001", cloud_mode="clear", operator="op")
        expected = registry["S001"]["split"]
        assert split_for(registry, "S001") == expected

    def test_fallback_equals_assign_split(self):
        """Scenes absent from the registry fall back to assign_split."""
        scene_id = "UNREGISTERED_SCENE_XYZ"
        assert split_for({}, scene_id) == assign_split(scene_id)

    def test_registry_and_fallback_agree(self):
        """Registry value must equal what assign_split would return."""
        registry: dict = {}
        scene_id = "S2A_MSIL2A_20230601_T19LGJ_UNIQUE"
        register_scene(registry, scene_id, cloud_mode="clear", operator="op")
        assert registry[scene_id]["split"] == assign_split(scene_id)


# ── operator_name ─────────────────────────────────────────────────────────────


class TestOperatorName:
    def test_returns_string(self):
        name = operator_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NINA_OPERATOR", "test_operator")
        assert operator_name() == "test_operator"

    def test_fallback_without_env(self, monkeypatch):
        monkeypatch.delenv("NINA_OPERATOR", raising=False)
        name = operator_name()
        assert isinstance(name, str)
        assert len(name) > 0
