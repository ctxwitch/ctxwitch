"""Tests for core context operations."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from ctxwitch.core.context import (
    ContextSnapshot,
    bump_version,
    diff_context,
    load_context,
    save_context,
    validate_context,
)


@pytest.fixture
def sample_context_data():
    return {
        "version": "v1.0.0",
        "name": "test-agent",
        "components": {
            "system_prompt": "You are a test agent.",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
        },
    }


@pytest.fixture
def context_file(sample_context_data, tmp_path):
    path = tmp_path / "witch.yaml"
    with open(path, "w") as f:
        yaml.dump(sample_context_data, f)
    return path


def test_load_context(context_file):
    snapshot = load_context(context_file)
    assert snapshot.version == "v1.0.0"
    assert snapshot.name == "test-agent"
    assert snapshot.sha


def test_validate_context(sample_context_data):
    validate_context(sample_context_data)


def test_validate_context_invalid():
    with pytest.raises(Exception):
        validate_context({"version": "v1.0.0"})


def test_save_and_reload(sample_context_data, tmp_path):
    snapshot = ContextSnapshot(
        version="v1.0.0",
        name="test-agent",
        data=sample_context_data,
    )
    path = tmp_path / "out.yaml"
    save_context(snapshot, path)

    loaded = load_context(path)
    assert loaded.version == snapshot.version
    assert loaded.sha == snapshot.sha


def test_diff_context():
    old = ContextSnapshot(
        version="v1.0.0",
        name="agent",
        data={
            "version": "v1.0.0",
            "name": "agent",
            "components": {
                "system_prompt": "Be helpful.",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
            },
        },
    )
    new = ContextSnapshot(
        version="v1.0.1",
        name="agent",
        data={
            "version": "v1.0.1",
            "name": "agent",
            "components": {
                "system_prompt": "Be strict.",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.7,
            },
        },
    )
    d = diff_context(old, new)
    assert d.has_changes
    assert len(d.entries) >= 2

    paths = [e.path for e in d.entries]
    assert "components.system_prompt" in paths
    assert "components.temperature" in paths


def test_diff_added_key():
    old = ContextSnapshot(version="v1", name="a", data={"version": "v1", "name": "a", "components": {"system_prompt": "x", "model": "m"}})
    new = ContextSnapshot(version="v2", name="a", data={"version": "v2", "name": "a", "components": {"system_prompt": "x", "model": "m"}, "metadata": {"key": "val"}})
    d = diff_context(old, new)
    assert any(e.change_type == "added" for e in d.entries)


def test_bump_version():
    assert bump_version("v1.0.0", "patch") == "v1.0.1"
    assert bump_version("v1.0.0", "minor") == "v1.1.0"
    assert bump_version("v1.0.0", "major") == "v2.0.0"
    assert bump_version("v0.9.9", "patch") == "v0.9.10"


def test_context_snapshot_sha_deterministic(sample_context_data):
    s1 = ContextSnapshot(version="v1.0.0", name="test", data=sample_context_data)
    s2 = ContextSnapshot(version="v1.0.0", name="test", data=sample_context_data)
    assert s1.sha == s2.sha
