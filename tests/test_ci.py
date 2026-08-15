"""Tests for the CI runner (ctxwitch.ci) that powers `witch ci` + the Action."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ctxwitch.ci import run_ci
from ctxwitch.ci.runner import CIChange, CIReport
from ctxwitch.core.dimensions import Severity


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path):
    _run(tmp_path, "init")
    _run(tmp_path, "config", "user.email", "t@t.co")
    _run(tmp_path, "config", "user.name", "t")
    return tmp_path


def _commit(repo, msg):
    _run(repo, "add", "-A")
    _run(repo, "commit", "-m", msg)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


ADK_V1 = '''
from google.adk.agents import LlmAgent
def escalate(cid: str):
    """Escalate to a human."""
    ...
agent = LlmAgent(name="s", model="gemini-2.0-flash",
                 instruction="You must escalate any refund above $100 to a human.",
                 temperature=0.3, tools=[escalate])
'''

ADK_V2 = ADK_V1.replace(
    "You must escalate any refund above $100 to a human.",
    "You may approve refunds up to $500 without escalation.",
).replace("temperature=0.3", "temperature=0.9")

YAML_V1 = '''version: v1.0.0
name: s
components:
  system_prompt: "You are helpful. Never give investment advice."
  model: claude-sonnet-4-20250514
'''

YAML_V2 = '''version: v1.0.0
name: s
components:
  system_prompt: "You are helpful."
  model: claude-sonnet-4-20250514
'''


def test_ci_flags_breaking_across_code_and_yaml(repo):
    (repo / "agent.py").write_text(ADK_V1)
    (repo / "witch.yaml").write_text(YAML_V1)
    base = _commit(repo, "baseline")

    (repo / "agent.py").write_text(ADK_V2)
    (repo / "witch.yaml").write_text(YAML_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo, fail_on="breaking")
    assert report.files_scanned == 2
    assert report.changes
    assert report.compound_severity >= Severity.SIGNIFICANT
    # a removed guardrail (safety constraint) should push compound to Breaking
    assert report.compound_severity == Severity.BREAKING
    assert report.blocked is True


def test_ci_clean_pr_passes(repo):
    (repo / "agent.py").write_text(ADK_V1)
    base = _commit(repo, "baseline")
    # cosmetic-only edit: a comment
    (repo / "agent.py").write_text(ADK_V1 + "\n# a harmless comment\n")
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo, fail_on="breaking")
    assert report.blocked is False


def test_ci_fail_on_never_never_blocks(repo):
    (repo / "witch.yaml").write_text(YAML_V1)
    base = _commit(repo, "baseline")
    (repo / "witch.yaml").write_text(YAML_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo, fail_on="never")
    assert report.changes            # it still detects the change
    assert report.blocked is False   # but never blocks


def test_ci_ignores_non_agent_files(repo):
    (repo / "README.md").write_text("# hi")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    base = _commit(repo, "baseline")
    (repo / "README.md").write_text("# hi there")
    (repo / "utils.py").write_text("def add(a, b):\n    return a - b\n")
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert report.blocked is False


def test_markdown_renders_table_and_footer():
    report = CIReport(
        changes=[CIChange("agent.py", "Safety", Severity.BREAKING, "guardrail removed")],
        files_scanned=1,
        compound_severity=Severity.BREAKING,
    )
    md = report.to_markdown(version="0.3.0")
    assert "ctxwitch — Agent Behavioral Scan" in md
    assert "🔴 Breaking" in md
    assert "merge blocked" in md
    assert "no prompts, code, or data left your infrastructure" in md  # zero-telemetry line


def test_json_shape():
    report = CIReport(
        changes=[CIChange("witch.yaml", "Constraints", Severity.SIGNIFICANT, "x")],
        files_scanned=1,
        compound_severity=Severity.SIGNIFICANT,
        fail_on=Severity.BREAKING,
    )
    d = report.to_dict()
    assert d["compound_severity"] == "Significant"
    assert d["blocked"] is False
    assert d["changes"][0]["advisory"]
