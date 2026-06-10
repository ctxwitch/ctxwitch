"""Tests for Tier 6: LLM-as-judge behavioral analysis."""

from unittest.mock import MagicMock, patch

import pytest

from ctxwitch.core.dimensions import (
    BehavioralReport,
    Dimension,
    DimensionImpact,
    Severity,
)
from ctxwitch.core.judge import (
    JudgeResult,
    _format_segments_for_judge,
    _parse_judge_response,
    get_provider,
    judge_to_impacts,
    needs_judge,
    run_judge,
)


# ── needs_judge ──────────────────────────────────────────────────────────


def _make_report(impacts):
    report = BehavioralReport(impacts=impacts)
    report.compute_compound_severity()
    return report


def test_needs_judge_significant_low_confidence():
    report = _make_report([
        DimensionImpact(
            dimension=Dimension.TONE,
            severity=Severity.SIGNIFICANT,
            reason="tone changed",
            confidence=0.60,
        ),
    ])
    assert needs_judge(report) is True


def test_needs_judge_significant_high_confidence():
    report = _make_report([
        DimensionImpact(
            dimension=Dimension.TONE,
            severity=Severity.SIGNIFICANT,
            reason="tone changed",
            confidence=0.90,
        ),
    ])
    assert needs_judge(report) is False


def test_needs_judge_minor_low_confidence():
    report = _make_report([
        DimensionImpact(
            dimension=Dimension.TONE,
            severity=Severity.MINOR,
            reason="slight tone shift",
            confidence=0.50,
        ),
    ])
    assert needs_judge(report) is False


def test_needs_judge_non_subjective_dimension():
    report = _make_report([
        DimensionImpact(
            dimension=Dimension.TOOLS_CAPABILITY,
            severity=Severity.SIGNIFICANT,
            reason="tool changed",
            confidence=0.50,
        ),
    ])
    assert needs_judge(report) is False


def test_needs_judge_breaking_persona():
    report = _make_report([
        DimensionImpact(
            dimension=Dimension.PERSONA,
            severity=Severity.BREAKING,
            reason="persona removed",
            confidence=0.70,
        ),
    ])
    assert needs_judge(report) is True


# ── get_provider ─────────────────────────────────────────────────────────


def test_get_provider_anthropic():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        assert get_provider() == "anthropic"


def test_get_provider_openai():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
        assert get_provider() == "openai"


def test_get_provider_anthropic_preferred():
    with patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-o"},
    ):
        assert get_provider() == "anthropic"


def test_get_provider_none():
    with patch.dict("os.environ", {}, clear=True):
        assert get_provider() is None


# ── _parse_judge_response ────────────────────────────────────────────────


def test_parse_valid_json():
    text = '{"analysis": [{"dimension": "tone", "severity": "significant", "explanation": "More formal", "prediction": "Users notice formality"}], "overall_summary": "Tone shift"}'
    result = _parse_judge_response(text, "anthropic", 500)
    assert result.available is True
    assert result.provider == "anthropic"
    assert len(result.analysis) == 1
    assert result.analysis[0]["dimension"] == "tone"
    assert result.overall_summary == "Tone shift"
    assert result.token_cost == 500


def test_parse_json_with_code_fence():
    text = '```json\n{"analysis": [], "overall_summary": "No changes."}\n```'
    result = _parse_judge_response(text, "openai", 100)
    assert result.available is True
    assert result.analysis == []
    assert result.overall_summary == "No changes."


def test_parse_invalid_json():
    text = "This is not JSON at all"
    result = _parse_judge_response(text, "anthropic", 200)
    assert result.available is True
    assert result.error == "Failed to parse LLM response as JSON"
    assert "This is not JSON" in result.overall_summary


# ── judge_to_impacts ─────────────────────────────────────────────────────


def test_judge_to_impacts_basic():
    result = JudgeResult(
        available=True,
        provider="anthropic",
        analysis=[
            {
                "dimension": "tone",
                "severity": "significant",
                "explanation": "Shift from casual to formal",
                "prediction": "Responses become notably more professional",
            },
            {
                "dimension": "persona",
                "severity": "minor",
                "explanation": "Slight identity refinement",
                "prediction": "Subtle difference in self-identification",
            },
        ],
        overall_summary="Major tone shift detected",
    )
    impacts = judge_to_impacts(result)
    assert len(impacts) == 2
    assert impacts[0].dimension == Dimension.TONE
    assert impacts[0].severity == Severity.SIGNIFICANT
    assert "[LLM Judge]" in impacts[0].reason
    assert "Predicted impact:" in impacts[0].reason
    assert impacts[0].confidence == 0.90


def test_judge_to_impacts_unavailable():
    result = JudgeResult(available=False, error="No API key")
    assert judge_to_impacts(result) == []


def test_judge_to_impacts_unknown_dimension():
    result = JudgeResult(
        available=True,
        provider="openai",
        analysis=[{"dimension": "unknown_dim", "severity": "minor", "explanation": "test"}],
    )
    assert judge_to_impacts(result) == []


def test_judge_to_impacts_interaction_style_variations():
    for dim_name in ["interaction_style", "interaction style"]:
        result = JudgeResult(
            available=True,
            provider="anthropic",
            analysis=[{"dimension": dim_name, "severity": "breaking", "explanation": "test"}],
        )
        impacts = judge_to_impacts(result)
        assert len(impacts) == 1
        assert impacts[0].dimension == Dimension.INTERACTION_STYLE


# ── _format_segments_for_judge ───────────────────────────────────────────


def test_format_segments_removed():
    segments = [{"change_type": "removed", "segment_type": "constraint", "old_text": "Never lie"}]
    text = _format_segments_for_judge(segments)
    assert "REMOVED" in text
    assert "Never lie" in text


def test_format_segments_added():
    segments = [{"change_type": "added", "segment_type": "task", "new_text": "Help with billing"}]
    text = _format_segments_for_judge(segments)
    assert "ADDED" in text
    assert "Help with billing" in text


def test_format_segments_modified():
    segments = [
        {
            "change_type": "modified",
            "segment_type": "persona",
            "old_text": "You are friendly",
            "new_text": "You are strict",
            "similarity": "45%",
        }
    ]
    text = _format_segments_for_judge(segments)
    assert "CHANGED" in text
    assert "You are friendly" in text
    assert "You are strict" in text
    assert "45%" in text


def test_format_segments_empty():
    assert _format_segments_for_judge([]) == "(no specific segments)"


# ── run_judge no provider ────────────────────────────────────────────────


def test_run_judge_no_provider():
    with patch.dict("os.environ", {}, clear=True):
        result = run_judge("old prompt", "new prompt", [], provider=None)
        assert result.available is False
        assert "No LLM API key" in result.error


def test_run_judge_unknown_provider():
    result = run_judge("old", "new", [], provider="gemini")
    assert result.available is False
    assert "Unknown provider" in result.error


# ── full integration: analyze_behavioral_impact with judge ───────────────


def test_analyze_with_judge_skipped_no_key():
    """When use_judge=True and needs_judge triggers but no API key exists,
    Tier 6 appends a 'Skipped' note to the report."""
    from ctxwitch.core.behavioral import analyze_behavioral_impact, _run_judge_tier

    old = {
        "components": {
            "system_prompt": "You are a helpful friendly assistant.",
            "temperature": 0.3,
        }
    }
    new = {
        "components": {
            "system_prompt": "You are a strict compliance officer. Never be casual.",
            "temperature": 0.3,
        }
    }

    report = analyze_behavioral_impact(old, new, use_judge=False)

    for impact in report.impacts:
        if impact.dimension in (Dimension.TONE, Dimension.PERSONA):
            impact.severity = Severity.SIGNIFICANT
            impact.confidence = 0.60

    with patch.dict("os.environ", {}, clear=True):
        _run_judge_tier(report, old["components"]["system_prompt"], new["components"]["system_prompt"], None)

    tier6_notes = [d for d in report.details if "Tier 6" in d]
    assert any("Skipped" in n or "no API key" in n.lower() for n in tier6_notes)


def test_analyze_with_judge_not_needed():
    from ctxwitch.core.behavioral import analyze_behavioral_impact

    old = {"components": {"system_prompt": "Hello.", "temperature": 0.3}}
    new = {"components": {"system_prompt": "Hello.", "temperature": 0.3}}
    report = analyze_behavioral_impact(old, new, use_judge=True)
    tier6_notes = [d for d in report.details if "Tier 6" in d]
    assert len(tier6_notes) == 0
