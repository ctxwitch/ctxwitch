"""Context store — git-backed versioning engine for witch.yaml files.

Wraps git operations to provide context-specific branching, commits,
and history tracking. Every context change is a git commit with
structured metadata.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ctxwitch.core.context import (
    ContextDiff,
    ContextSnapshot,
    bump_version,
    diff_context,
    load_context,
    save_context,
)

CTXWITCH_DIR = ".ctxwitch"
CONTEXT_FILE = "witch.yaml"
HISTORY_DIR = "history"
CONFIG_FILE = "config.yaml"


@dataclass
class CommitRecord:
    sha: str
    context_sha: str
    version: str
    author: str
    message: str
    timestamp: str
    branch: str


@dataclass
class BranchInfo:
    name: str
    is_current: bool
    last_commit: Optional[CommitRecord] = None


class ContextStore:
    """Git-backed context versioning store."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or Path.cwd()
        self.ctx_dir = self.root / CTXWITCH_DIR
        self.context_path = self.root / CONTEXT_FILE
        self.history_path = self.ctx_dir / HISTORY_DIR
        self.config_path = self.ctx_dir / CONFIG_FILE

    @property
    def is_initialized(self) -> bool:
        return self.ctx_dir.exists() and self.context_path.exists()

    def init(self, name: str, owner: str = "") -> ContextSnapshot:
        """Initialize a new ctxwitch project."""
        if self.is_initialized:
            raise RuntimeError("ctxwitch already initialized in this directory")

        if not owner:
            owner = self._git_user() or "unknown"

        self.ctx_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)
        (self.ctx_dir / "evals").mkdir(exist_ok=True)

        from ctxwitch.core.schema import SCAFFOLD_CONTEXT

        context_content = SCAFFOLD_CONTEXT.format(name=name, owner=owner)
        self.context_path.write_text(context_content)

        config = {
            "project": name,
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "settings": {
                "require_pr_for_main": True,
                "auto_eval": True,
                "canary_percentage": 10,
                "canary_duration_minutes": 30,
            },
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        evals_path = self.root / "evals"
        evals_path.mkdir(exist_ok=True)
        golden_path = evals_path / "golden.jsonl"
        if not golden_path.exists():
            sample = {
                "input": "What is your refund policy?",
                "expected_behavior": "Explains refund policy clearly and accurately",
                "tags": ["refund", "policy"],
            }
            golden_path.write_text(json.dumps(sample) + "\n")

        gitignore_path = self.root / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(
                "*.pyc\n__pycache__/\n.env\n*.egg-info/\ndist/\nbuild/\n"
            )

        self._ensure_git()
        self._git("add", ".")
        self._git("commit", "-m", f"witch init: {name}")

        return load_context(self.context_path)

    def checkout(self, branch: str, create: bool = False) -> str:
        """Switch to or create a context branch."""
        self._require_init()
        if create:
            self._git("checkout", "-b", branch)
        else:
            self._git("checkout", branch)
        return branch

    def commit(
        self, message: str, author: str = "", bump: str = "patch"
    ) -> CommitRecord:
        """Commit the current context state with version bump."""
        self._require_init()

        snapshot = load_context(self.context_path)
        new_version = bump_version(snapshot.version, bump)

        data = snapshot.data.copy()
        data["version"] = new_version
        with open(self.context_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        snapshot = load_context(self.context_path)

        if not author:
            author = self._git_user() or "unknown"

        history_entry = {
            "version": new_version,
            "context_sha": snapshot.sha,
            "author": author,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "branch": self.current_branch,
        }
        # Branch switches can prune the history dir when git leaves it empty.
        self.history_path.mkdir(parents=True, exist_ok=True)
        history_file = self.history_path / f"{new_version}.json"
        with open(history_file, "w") as f:
            json.dump(history_entry, f, indent=2)

        # Stage only what a context commit owns — never sweep up unrelated
        # working-tree changes with a blanket `git add .`.
        self._git("add", CONTEXT_FILE, str(history_file.relative_to(self.root)))
        commit_msg = f"witch: {message} [{new_version}]"
        self._git("commit", "-m", commit_msg)

        git_sha = self._git("rev-parse", "HEAD").strip()
        self._tag_version(new_version)

        return CommitRecord(
            sha=git_sha[:8],
            context_sha=snapshot.sha,
            version=new_version,
            author=author,
            message=message,
            timestamp=history_entry["timestamp"],
            branch=self.current_branch,
        )

    def diff(self, ref: str = "HEAD") -> ContextDiff:
        """Diff current context against a ref (branch, commit, or tag)."""
        self._require_init()

        current = load_context(self.context_path)

        try:
            old_content = self._git("show", f"{ref}:{CONTEXT_FILE}")
            old_data = yaml.safe_load(old_content)
            old = ContextSnapshot(
                version=old_data.get("version", "v0.0.0"),
                name=old_data.get("name", "unknown"),
                data=old_data,
            )
        except subprocess.CalledProcessError:
            old = ContextSnapshot(version="v0.0.0", name="(none)", data={})

        return diff_context(old, current)

    def log(self, count: int = 20, since_days: Optional[int] = None) -> List[CommitRecord]:
        """Get context change history."""
        self._require_init()

        records = []
        history_files = sorted(self.history_path.glob("*.json"), reverse=True)

        for hf in history_files[:count]:
            with open(hf) as f:
                entry = json.load(f)
            records.append(
                CommitRecord(
                    sha="",
                    context_sha=entry.get("context_sha", ""),
                    version=entry["version"],
                    author=entry.get("author", "unknown"),
                    message=entry.get("message", ""),
                    timestamp=entry.get("timestamp", ""),
                    branch=entry.get("branch", "main"),
                )
            )

        if since_days is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - (since_days * 86400)
            records = [
                r
                for r in records
                if datetime.fromisoformat(r.timestamp).timestamp() > cutoff
            ]

        return records

    def branches(self) -> List[BranchInfo]:
        """List all context branches."""
        self._require_init()
        output = self._git("branch", "--list")
        current = self.current_branch
        branches = []
        for line in output.strip().split("\n"):
            name = line.strip().lstrip("* ").strip()
            if name:
                branches.append(BranchInfo(name=name, is_current=(name == current)))
        return branches

    @property
    def current_branch(self) -> str:
        try:
            return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        except subprocess.CalledProcessError:
            return "main"

    def rollback(self, version: str) -> ContextSnapshot:
        """Rollback to a specific version."""
        self._require_init()

        history_file = self.history_path / f"{version}.json"
        if not history_file.exists():
            raise ValueError(f"Version {version} not found in history")

        commit_sha = self._resolve_version_commit(version)
        if not commit_sha:
            raise ValueError(f"Git commit for version {version} not found")

        self._git("checkout", commit_sha, "--", CONTEXT_FILE)
        self._git("add", CONTEXT_FILE)
        self._git("commit", "-m", f"witch rollback to {version}")

        return load_context(self.context_path)

    def merge(self, branch: str, message: str = "", no_ff: bool = True) -> str:
        """Merge a context branch into the current branch.

        Returns the merge commit sha. Raises subprocess.CalledProcessError
        on conflicts (the caller decides how to resolve or abort).
        """
        self._require_init()
        msg = message or f"witch merge: {branch} into {self.current_branch}"
        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        args.extend([branch, "-m", msg])
        try:
            self._git(*args)
        except subprocess.CalledProcessError:
            # Leave the tree clean rather than half-merged.
            self._git("merge", "--abort")
            raise
        return self._git("rev-parse", "HEAD").strip()

    def _tag_version(self, version: str) -> None:
        """Tag the current commit so versions resolve without log-grepping.

        Namespaced under witch/ to avoid colliding with the host repo's
        own release tags.
        """
        try:
            self._git("tag", f"witch/{version}")
        except subprocess.CalledProcessError:
            pass  # tag exists (e.g. re-init over old history) — keep the original

    def _resolve_version_commit(self, version: str) -> Optional[str]:
        """Resolve a context version to a commit sha: tag first, then the
        legacy commit-message grep for histories created before tagging."""
        try:
            return self._git("rev-list", "-n", "1", f"witch/{version}").strip()
        except subprocess.CalledProcessError:
            pass

        git_log = self._git("log", "--oneline", "--all", f"--grep=\\[{version}\\]")
        lines = [l for l in git_log.strip().split("\n") if l.strip()]
        if not lines:
            return None
        return lines[0].split()[0]

    def _require_init(self):
        if not self.is_initialized:
            raise RuntimeError(
                "Not a ctxwitch project. Run 'witch init <name>' first."
            )

    def _ensure_git(self):
        git_dir = self.root / ".git"
        if not git_dir.exists():
            self._git("init")

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, f"git {' '.join(args)}", result.stderr
            )
        return result.stdout

    def _git_user(self) -> Optional[str]:
        try:
            name = self._git("config", "user.name").strip()
            return name if name else None
        except subprocess.CalledProcessError:
            return None
