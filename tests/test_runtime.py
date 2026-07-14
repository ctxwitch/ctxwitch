"""Tests for the runtime integration API and witch pr merge."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ctxwitch.cli.main import cli
from ctxwitch.runtime import load_agent_context, load_components


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def project(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.dev"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmp_path, capture_output=True)
    original = os.getcwd()
    os.chdir(tmp_path)
    CliRunner().invoke(cli, ["init", "test-agent"])
    yield tmp_path
    os.chdir(original)


class TestRuntimeLoader:
    def test_load_components_base(self, project):
        components = load_components()
        assert components["model"] == "claude-sonnet-4-20250514"
        assert components["temperature"] == 0.3

    def test_env_override_deep_merges(self, project):
        components = load_components(env="dev")
        assert components["temperature"] == 0.7  # dev override
        assert components["model"] == "claude-sonnet-4-20250514"  # untouched

    def test_env_from_environment_variable(self, project, monkeypatch):
        monkeypatch.setenv("CTXWITCH_ENV", "dev")
        assert load_components()["temperature"] == 0.7

    def test_unknown_env_raises(self, project):
        with pytest.raises(KeyError):
            load_agent_context(env="does-not-exist")

    def test_explicit_path(self, project):
        components = load_components(path=project / "witch.yaml")
        assert "system_prompt" in components


class TestPRMerge:
    def _make_pr(self, runner, breaking: bool):
        runner.invoke(cli, ["checkout", "-b", "change"])
        data = yaml.safe_load(Path("witch.yaml").read_text())
        if breaking:
            data["components"]["rag_config"]["enabled"] = True
            data["components"]["rag_config"]["source"] = "s3://docs/"
        else:
            data["components"]["system_prompt"] += "\nBe concise."
        Path("witch.yaml").write_text(yaml.dump(data, sort_keys=False))
        runner.invoke(cli, ["commit", "-m", "change"])
        result = runner.invoke(cli, ["pr", "create", "-t", "Test change"])
        assert result.exit_code == 0

    def test_merge_clean_pr(self, runner, project):
        self._make_pr(runner, breaking=False)
        result = runner.invoke(cli, ["pr", "merge", "1"])
        assert result.exit_code == 0
        assert "Merged PR #1" in result.output
        # base branch received the change
        subprocess.run(["git", "checkout", "main"], capture_output=True)
        assert "Be concise." in Path("witch.yaml").read_text()

    def test_merge_blocked_on_breaking(self, runner, project):
        self._make_pr(runner, breaking=True)
        result = runner.invoke(cli, ["pr", "merge", "1"])
        assert result.exit_code == 2
        assert "blocked" in result.output.lower()

        result = runner.invoke(cli, ["pr", "merge", "1", "--allow-breaking"])
        assert result.exit_code == 0
        assert "overridden" in result.output.lower()

    def test_merge_twice_fails(self, runner, project):
        self._make_pr(runner, breaking=False)
        runner.invoke(cli, ["pr", "merge", "1"])
        result = runner.invoke(cli, ["pr", "merge", "1"])
        assert result.exit_code == 1
        assert "already merged" in result.output

    def test_merge_missing_pr(self, runner, project):
        result = runner.invoke(cli, ["pr", "merge", "99"])
        assert result.exit_code == 1
