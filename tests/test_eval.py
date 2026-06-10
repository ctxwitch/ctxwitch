"""Tests for eval gate framework."""

import json
import tempfile
from pathlib import Path

import pytest

from ctxwitch.eval.gate import EvalGate, GateVerdict, MetricResult, create_default_gate


@pytest.fixture
def basic_context():
    return {
        "version": "v1.0.0",
        "name": "test",
        "components": {
            "system_prompt": "You are a helpful assistant.",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "guardrails": {
                "blocked_topics": ["violence"],
                "output_filters": ["pii_filter"],
                "max_turns": 50,
            },
        },
    }


@pytest.fixture
def eval_config():
    return {
        "metrics": [
            {"name": "helpfulness", "threshold": 50, "direction": "higher_is_better"},
            {"name": "safety", "threshold": 50, "direction": "higher_is_better"},
        ],
        "block_on_failure": True,
    }


def test_eval_gate_passes(basic_context, eval_config):
    gate = create_default_gate()
    result = gate.run(basic_context, eval_config=eval_config)
    assert result.passed
    assert result.verdict == GateVerdict.PASSED


def test_eval_gate_skips_without_config(basic_context):
    gate = create_default_gate()
    result = gate.run(basic_context, eval_config=None)
    assert result.verdict == GateVerdict.SKIPPED


def test_eval_gate_fails_high_threshold(basic_context):
    config = {
        "metrics": [
            {"name": "helpfulness", "threshold": 99, "direction": "higher_is_better"},
        ],
        "block_on_failure": True,
    }
    gate = create_default_gate()
    result = gate.run(basic_context, eval_config=config)
    assert result.verdict == GateVerdict.FAILED


def test_custom_evaluator():
    gate = EvalGate()

    def always_pass(context, golden):
        return MetricResult(
            name="custom", score=100, threshold=50,
            direction="higher_is_better", passed=True,
        )

    gate.register(always_pass)
    result = gate.run(
        {"components": {}},
        eval_config={"metrics": [], "block_on_failure": True},
    )
    assert result.passed


def test_golden_dataset_loading(basic_context, eval_config, tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(json.dumps({"input": "test", "expected_behavior": "responds"}) + "\n")

    gate = create_default_gate()
    result = gate.run(basic_context, eval_config=eval_config, golden_path=golden)
    assert result.golden_count == 1
