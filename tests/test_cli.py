"""Tests for CLI commands."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from ctxwitch.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temp directory with git init for testing."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, capture_output=True,
    )
    original = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original)


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    from ctxwitch import __version__

    assert __version__ in result.output


def test_init(runner, tmp_project):
    result = runner.invoke(cli, ["init", "test-project"])
    assert result.exit_code == 0
    assert "test-project" in result.output
    assert (tmp_project / "witch.yaml").exists()
    assert (tmp_project / ".ctxwitch").exists()


def test_init_already_exists(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["init", "test-project"])
    assert result.exit_code != 0


def test_status_after_init(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "test-project" in result.output


def test_commit(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["commit", "-m", "update prompt"])
    assert result.exit_code == 0
    assert "v0.1.1" in result.output


def test_log_after_commit(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    runner.invoke(cli, ["commit", "-m", "first change"])
    result = runner.invoke(cli, ["log"])
    assert result.exit_code == 0
    assert "first change" in result.output


def test_checkout_branch(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["checkout", "-b", "feature-test"])
    assert result.exit_code == 0
    assert "feature-test" in result.output


def test_branches(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    runner.invoke(cli, ["checkout", "-b", "feature-a"])
    result = runner.invoke(cli, ["branches"])
    assert result.exit_code == 0
    assert "feature-a" in result.output


def test_eval(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["eval"])
    assert result.exit_code == 0
    assert "PASSED" in result.output or "Verdict" in result.output


def test_spell_validate(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["spell", "validate"])
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_spell_set(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["spell", "set", "components.temperature", "0.7"])
    assert result.exit_code == 0
    assert "0.7" in result.output


def test_inspect_prompt(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["inspect", "prompt"])
    assert result.exit_code == 0
    assert "helpful" in result.output.lower() or "System Prompt" in result.output


def test_spell_export_json(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["spell", "export", "--format", "json"])
    assert result.exit_code == 0
    assert '"version"' in result.output


def test_eval_blocked_by_breaking_behavioral_change(runner, tmp_project):
    """Metrics passing must not green-light a Breaking behavioral change."""
    import yaml

    runner.invoke(cli, ["init", "test-project"])

    # second commit so behavioral analysis has a HEAD~1 to compare against
    data = yaml.safe_load(Path("witch.yaml").read_text())
    data["components"]["system_prompt"] = "You are a helpful support assistant."
    Path("witch.yaml").write_text(yaml.dump(data, sort_keys=False))
    runner.invoke(cli, ["commit", "-m", "baseline"])

    # breaking change in the working tree: enable RAG (Knowledge Scope: Breaking)
    data = yaml.safe_load(Path("witch.yaml").read_text())
    data["components"]["rag_config"]["enabled"] = True
    data["components"]["rag_config"]["source"] = "s3://docs/"
    Path("witch.yaml").write_text(yaml.dump(data, sort_keys=False))
    runner.invoke(cli, ["commit", "-m", "enable rag"])

    result = runner.invoke(cli, ["eval"])
    assert result.exit_code == 2
    assert "BLOCKED" in result.output

    result = runner.invoke(cli, ["eval", "--allow-breaking"])
    assert result.exit_code == 0
    assert "OVERRIDE" in result.output


def test_eval_passes_cleanly_without_breaking_changes(runner, tmp_project):
    runner.invoke(cli, ["init", "test-project"])
    result = runner.invoke(cli, ["eval"])
    assert result.exit_code == 0
    assert "PASSED" in result.output
