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


# ── scope: only the product agent's surface is scored ───────────────────────


def _write(repo, rel, text):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _move(repo, old, new):
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    (repo / old).rename(repo / new)


def test_ci_skips_eval_vendored_generated_and_pipeline_files(repo):
    paths = [
        "evaluation/agent.py",
        "external_tools/ai_scientist/agent.py",
        "apps/collect-trace/agent.py",
        "generated/witch.yaml",
    ]
    for rel in paths:
        _write(repo, rel, YAML_V1 if rel.endswith(".yaml") else ADK_V1)
    base = _commit(repo, "baseline")
    for rel in paths:
        _write(repo, rel, YAML_V2 if rel.endswith(".yaml") else ADK_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert report.blocked is False
    assert {s.file for s in report.skipped} == set(paths)
    reasons = {s.file: s.reason for s in report.skipped}
    assert reasons["evaluation/agent.py"] == "evaluation / benchmark prompt"
    assert reasons["external_tools/ai_scientist/agent.py"] == "vendored third-party code"
    assert reasons["apps/collect-trace/agent.py"] == "offline data pipeline"
    assert reasons["generated/witch.yaml"] == "generated data / lockfile"


def test_ci_still_scores_the_product_agent_next_to_skipped_files(repo):
    _write(repo, "agent.py", ADK_V1)
    _write(repo, "evaluation/agent.py", ADK_V1)
    base = _commit(repo, "baseline")
    _write(repo, "agent.py", ADK_V2)
    _write(repo, "evaluation/agent.py", ADK_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert {c.file for c in report.changes} == {"agent.py"}
    assert report.blocked is True
    assert [s.file for s in report.skipped] == ["evaluation/agent.py"]


def test_ci_include_path_overrides_exclusion(repo):
    _write(repo, "evaluation/agent.py", ADK_V1)
    base = _commit(repo, "baseline")
    _write(repo, "evaluation/agent.py", ADK_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo, include_paths=["evaluation/*"])
    assert {c.file for c in report.changes} == {"evaluation/agent.py"}
    assert report.skipped == []


def test_ci_coding_assistant_config_is_opt_in_and_labeled(repo):
    rel = ".claude/skills/review/witch.yaml"
    _write(repo, rel, YAML_V1)
    base = _commit(repo, "baseline")
    _write(repo, rel, YAML_V2)
    _commit(repo, "pr")

    default = run_ci(base=base, repo_root=repo)
    assert default.changes == []
    assert default.skipped[0].surface == "coding-assistant"
    assert "--include-coding-assistant" in default.to_markdown()

    opted = run_ci(base=base, repo_root=repo, include_coding_assistant=True)
    assert opted.changes
    assert {c.surface for c in opted.changes} == {"coding-assistant"}
    assert "[coding-assistant]" in opted.to_markdown()
    assert opted.to_dict()["changes"][0]["surface"] == "coding-assistant"


def test_ci_reports_pure_move_as_move_not_delete_plus_add(repo):
    _write(repo, "crates/goose/agent.py", ADK_V1)
    base = _commit(repo, "baseline")
    _move(repo, "crates/goose/agent.py", "crates/goose-cm/agent.py")
    _commit(repo, "move")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert report.blocked is False
    assert [(m.old_file, m.new_file) for m in report.moves] == [
        ("crates/goose/agent.py", "crates/goose-cm/agent.py")
    ]
    md = report.to_markdown()
    assert "Moved without content change" in md
    assert "`crates/goose/agent.py` → `crates/goose-cm/agent.py`" in md
    assert report.to_dict()["moves"] == [
        {"from": "crates/goose/agent.py", "to": "crates/goose-cm/agent.py"}
    ]


def test_ci_move_in_working_tree_is_a_move(repo):
    (repo / "witch.yaml").write_text(YAML_V1)
    base = _commit(repo, "baseline")
    (repo / "witch.yaml").rename(repo / "agent.yaml")
    _run(repo, "add", "-A")   # staged, not committed: head=None reads the worktree

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert [(m.old_file, m.new_file) for m in report.moves] == [("witch.yaml", "agent.yaml")]


def test_ci_move_with_edit_is_still_analyzed(repo):
    _write(repo, "a/agent.py", ADK_V1)
    base = _commit(repo, "baseline")
    _move(repo, "a/agent.py", "b/agent.py")
    _write(repo, "b/agent.py", ADK_V2)
    _commit(repo, "move+edit")

    report = run_ci(base=base, repo_root=repo)
    assert report.moves == []
    assert report.changes


def test_ci_move_of_non_agent_file_is_not_reported(repo):
    _write(repo, "a/utils.py", "def add(a, b):\n    return a + b\n")
    base = _commit(repo, "baseline")
    _move(repo, "a/utils.py", "b/utils.py")
    _commit(repo, "move")

    report = run_ci(base=base, repo_root=repo)
    assert report.moves == []
    assert report.changes == []


def test_cli_ci_include_coding_assistant_flag(repo, monkeypatch):
    from click.testing import CliRunner

    from ctxwitch.cli.main import cli

    rel = ".claude/skills/review/witch.yaml"
    _write(repo, rel, YAML_V1)
    base = _commit(repo, "baseline")
    _write(repo, rel, YAML_V2)
    _commit(repo, "pr")
    monkeypatch.chdir(repo)

    runner = CliRunner()
    off = runner.invoke(cli, ["ci", "--base", base, "--format", "json"])
    assert off.exit_code == 0, off.output
    assert '"changes": []' in off.output

    on = runner.invoke(cli, ["ci", "--base", base, "--format", "json",
                             "--include-coding-assistant"])
    assert on.exit_code == 2, on.output   # guardrail removal → Breaking → blocked
    assert '"surface": "coding-assistant"' in on.output


# ── prompt files + agent config: same file kinds as `witch scan` ────────────

PROMPT_MD_V1 = "You must escalate any refund above $100 to a human.\n"
PROMPT_MD_V2 = "You may approve refunds up to $500 without escalation.\n"

AGENT_JSON_V1 = '{"model": "gpt-4o", "temperature": 0.2}\n'
AGENT_JSON_V2 = '{"model": "gpt-4o-mini", "temperature": 0.9}\n'


def _scan_diff(rel, old, new):
    from ctxwitch.extract.extractor import diff_code

    rep, _, _ = diff_code(old, new, source_file=rel)
    return {(i.dimension.display_name, i.severity) for i in rep.impacts
            if i.severity > Severity.NO_CHANGE}


def test_ci_scores_markdown_prompt_file(repo):
    _write(repo, "prompts/system.md", PROMPT_MD_V1)
    base = _commit(repo, "baseline")
    _write(repo, "prompts/system.md", PROMPT_MD_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.files_scanned == 1
    assert {c.file for c in report.changes} == {"prompts/system.md"}
    # agrees with `witch scan prompts/system.md --diff`
    assert {(c.dimension, c.severity) for c in report.changes} == _scan_diff(
        "prompts/system.md", PROMPT_MD_V1, PROMPT_MD_V2
    )


def test_ci_scores_json_agent_config(repo):
    _write(repo, "agent.json", AGENT_JSON_V1)
    base = _commit(repo, "baseline")
    _write(repo, "agent.json", AGENT_JSON_V2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.files_scanned == 1
    assert {c.file for c in report.changes} == {"agent.json"}
    assert {(c.dimension, c.severity) for c in report.changes} == _scan_diff(
        "agent.json", AGENT_JSON_V1, AGENT_JSON_V2
    )


def test_ci_scores_plain_yaml_agent_config(repo):
    # no `components:` block, so not a witch.yaml context — read as config
    _write(repo, "config/agent.yaml", "model: gpt-4o\ntemperature: 0.2\n")
    base = _commit(repo, "baseline")
    _write(repo, "config/agent.yaml", "model: gpt-4o-mini\ntemperature: 0.9\n")
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert {c.file for c in report.changes} == {"config/agent.yaml"}


def test_ci_readme_edit_stays_unscored(repo):
    _write(repo, "README.md", "# Support bot\n\nEscalates refunds above $100.\n")
    base = _commit(repo, "baseline")
    _write(repo, "README.md", "# Support bot\n\nApproves refunds up to $500.\n")
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert report.files_scanned == 0
    assert [(s.file, s.reason) for s in report.skipped] == [("README.md", "documentation")]


def test_ci_non_agent_text_and_json_stay_unscored(repo):
    files = {
        "requirements.txt": ("requests==2.31.0\n", "requests==2.32.0\n"),
        "docs/prompting.md": ("Old guide.\n", "New guide.\n"),
        "package.json": ('{"name": "a", "version": "1.0.0"}\n',
                         '{"name": "a", "version": "1.1.0"}\n'),
        "k8s/deploy.yaml": ("spec:\n  replicas: 1\n", "spec:\n  replicas: 2\n"),
        "Makefile": ("all:\n\techo a\n", "all:\n\techo b\n"),
    }
    for rel, (v1, _) in files.items():
        _write(repo, rel, v1)
    base = _commit(repo, "baseline")
    for rel, (_, v2) in files.items():
        _write(repo, rel, v2)
    _commit(repo, "pr")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert report.blocked is False
    reasons = {s.file: s.reason for s in report.skipped}
    assert reasons == {
        "requirements.txt": "packaging / build / CI manifest",
        "docs/prompting.md": "documentation",
        "package.json": "packaging / build / CI manifest",
    }  # k8s yaml is in scope but carries no agent fields; Makefile isn't read


def test_ci_move_of_prompt_file_is_a_move(repo):
    _write(repo, "prompts/system.md", PROMPT_MD_V1)
    base = _commit(repo, "baseline")
    _move(repo, "prompts/system.md", "agent/prompts/system.md")
    _commit(repo, "move")

    report = run_ci(base=base, repo_root=repo)
    assert report.changes == []
    assert [(m.old_file, m.new_file) for m in report.moves] == [
        ("prompts/system.md", "agent/prompts/system.md")
    ]


def test_ci_move_of_non_agent_json_is_not_reported(repo):
    _write(repo, "a/settings.json", '{"port": 8080, "debug": false}\n')
    base = _commit(repo, "baseline")
    _move(repo, "a/settings.json", "b/settings.json")
    _commit(repo, "move")

    report = run_ci(base=base, repo_root=repo)
    assert report.moves == []
    assert report.changes == []
