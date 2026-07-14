"""Tests for the witch tour guided walkthrough."""

import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from ctxwitch.cli.main import cli
from ctxwitch.cli.tour import BRANCH, GUARDRAIL, NEW_RULE, SANDBOX_NAME


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workdir(tmp_path):
    original = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original)


def _enter_sandbox(workdir):
    os.chdir(workdir / SANDBOX_NAME)


def test_tour_creates_sandbox(runner, workdir):
    result = runner.invoke(cli, ["tour"])
    assert result.exit_code == 0
    assert "step 1" in result.output.lower()
    sandbox = workdir / SANDBOX_NAME
    assert (sandbox / "witch.yaml").exists()
    assert (sandbox / ".ctxwitch" / "tour.yaml").exists()
    assert (sandbox / "evals" / "golden.jsonl").exists()
    # sandbox is its own committed git repo
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=sandbox, capture_output=True, text=True
    )
    assert "witch tour" in log.stdout


def test_tour_refuses_foreign_directory(runner, workdir):
    (workdir / SANDBOX_NAME).mkdir()
    (workdir / SANDBOX_NAME / "precious.txt").write_text("user data")
    result = runner.invoke(cli, ["tour"])
    assert result.exit_code != 0
    assert (workdir / SANDBOX_NAME / "precious.txt").exists()


def test_tour_reset_only_removes_tour_sandbox(runner, workdir):
    runner.invoke(cli, ["tour"])
    result = runner.invoke(cli, ["tour", "--reset"])
    assert result.exit_code == 0
    assert not (workdir / SANDBOX_NAME).exists()
    # reset with nothing to remove is a no-op
    result = runner.invoke(cli, ["tour", "--reset"])
    assert result.exit_code == 0


def test_full_tour_flow(runner, workdir):
    runner.invoke(cli, ["tour"])
    _enter_sandbox(workdir)

    # step 1: edit not made yet, --do applies it
    assert "step 1" in runner.invoke(cli, ["tour"]).output.lower()
    runner.invoke(cli, ["tour", "--do"])
    assert NEW_RULE in Path("witch.yaml").read_text()

    # step 2: commit
    assert "step 2" in runner.invoke(cli, ["tour"]).output.lower()
    result = runner.invoke(cli, ["commit", "-m", "raise refund autonomy"])
    assert result.exit_code == 0

    # step 3: branch
    assert "step 3" in runner.invoke(cli, ["tour"]).output.lower()
    assert runner.invoke(cli, ["checkout", "-b", BRANCH]).exit_code == 0

    # step 4: remove guardrail via --do
    assert "step 4" in runner.invoke(cli, ["tour"]).output.lower()
    runner.invoke(cli, ["tour", "--do"])
    assert GUARDRAIL not in Path("witch.yaml").read_text()

    # step 5: commit the breaking change
    assert "step 5" in runner.invoke(cli, ["tour"]).output.lower()
    assert runner.invoke(cli, ["commit", "-m", "remove guardrail"]).exit_code == 0

    # step 6: open the Context PR
    assert "step 6" in runner.invoke(cli, ["tour"]).output.lower()
    assert runner.invoke(cli, ["pr", "create", "-t", "Remove guardrail"]).exit_code == 0

    # finished
    assert "finished" in runner.invoke(cli, ["tour"]).output.lower()


def test_do_on_command_step_does_not_edit(runner, workdir):
    runner.invoke(cli, ["tour"])
    _enter_sandbox(workdir)
    runner.invoke(cli, ["tour", "--do"])  # step 1 edit
    before = Path("witch.yaml").read_text()
    result = runner.invoke(cli, ["tour", "--do"])  # step 2 is a command
    assert "command" in result.output.lower()
    assert Path("witch.yaml").read_text() == before


def test_tour_resumes_from_parent_dir(runner, workdir):
    runner.invoke(cli, ["tour"])
    _enter_sandbox(workdir)
    runner.invoke(cli, ["tour", "--do"])
    os.chdir(workdir)  # back to parent — tour should still find the sandbox
    result = runner.invoke(cli, ["tour"])
    assert "step 2" in result.output.lower()
    assert "cd" in result.output.lower()
