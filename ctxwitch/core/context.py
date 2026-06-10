"""Context model — load, validate, diff, and serialize witch.yaml files."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jsonschema import ValidationError, validate

from ctxwitch.core.schema import CONTEXT_SCHEMA


@dataclass
class ContextSnapshot:
    """Immutable snapshot of a witch.yaml at a point in time."""

    version: str
    name: str
    data: Dict[str, Any]
    sha: str = ""
    author: str = ""
    timestamp: str = ""
    message: str = ""

    def __post_init__(self):
        if not self.sha:
            self.sha = self._compute_sha()
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def _compute_sha(self) -> str:
        canonical = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass
class ContextDiffEntry:
    """A single change within a context diff."""

    path: str
    old_value: Any
    new_value: Any
    change_type: str  # "added", "removed", "modified"


@dataclass
class ContextDiff:
    """Semantic diff between two context snapshots."""

    old_version: str
    new_version: str
    old_sha: str
    new_sha: str
    entries: List[ContextDiffEntry] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return len(self.entries) > 0

    @property
    def summary(self) -> str:
        added = sum(1 for e in self.entries if e.change_type == "added")
        removed = sum(1 for e in self.entries if e.change_type == "removed")
        modified = sum(1 for e in self.entries if e.change_type == "modified")
        parts = []
        if added:
            parts.append(f"{added} added")
        if removed:
            parts.append(f"{removed} removed")
        if modified:
            parts.append(f"{modified} modified")
        return ", ".join(parts) if parts else "no changes"


def load_context(path: Path) -> ContextSnapshot:
    """Load and validate a witch.yaml file."""
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty context file: {path}")

    validate_context(data)

    return ContextSnapshot(
        version=data.get("version", "v0.0.0"),
        name=data.get("name", "unnamed"),
        data=data,
    )


def validate_context(data: Dict[str, Any]) -> None:
    """Validate context data against the schema. Raises ValidationError."""
    try:
        validate(instance=data, schema=CONTEXT_SCHEMA)
    except ValidationError as e:
        raise ValidationError(
            f"Invalid context: {e.message} at path {'.'.join(str(p) for p in e.absolute_path)}"
        ) from e


def save_context(snapshot: ContextSnapshot, path: Path) -> None:
    """Write a context snapshot to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(snapshot.data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def diff_context(old: ContextSnapshot, new: ContextSnapshot) -> ContextDiff:
    """Compute semantic diff between two context snapshots."""
    entries: List[ContextDiffEntry] = []
    _diff_recursive(old.data, new.data, "", entries)
    return ContextDiff(
        old_version=old.version,
        new_version=new.version,
        old_sha=old.sha,
        new_sha=new.sha,
        entries=entries,
    )


def _diff_recursive(
    old: Any, new: Any, prefix: str, entries: List[ContextDiffEntry]
) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        all_keys = set(old.keys()) | set(new.keys())
        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old:
                entries.append(ContextDiffEntry(path, None, new[key], "added"))
            elif key not in new:
                entries.append(ContextDiffEntry(path, old[key], None, "removed"))
            else:
                _diff_recursive(old[key], new[key], path, entries)
    elif isinstance(old, list) and isinstance(new, list):
        if old != new:
            entries.append(ContextDiffEntry(prefix, old, new, "modified"))
    elif old != new:
        entries.append(ContextDiffEntry(prefix, old, new, "modified"))


def bump_version(version: str, bump_type: str = "patch") -> str:
    """Bump a semver string. bump_type: major, minor, patch."""
    v = version.lstrip("v")
    parts = v.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"v{major}.{minor}.{patch}"
