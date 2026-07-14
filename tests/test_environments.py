"""Tests for Tier-1b environment-override analysis.

Production behavior is base ⊕ prod-override. A change confined to an
environment override moves deployed behavior while base `components` is
untouched — CBIA must score it instead of reporting "No Change".
"""

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.dimensions import Dimension, Severity
from ctxwitch.core.impact import analyze_environment_changes


def _ctx(temp=0.3, blocked=None, environments=None):
    data = {
        "components": {
            "system_prompt": "You are a helpful assistant.",
            "model": "claude-sonnet-4-20250514",
            "temperature": temp,
            "guardrails": {"blocked_topics": blocked if blocked is not None else ["violence"]},
        }
    }
    if environments is not None:
        data["environments"] = environments
    return data


def _dims(report):
    return {i.dimension: i.severity for i in report.changed_dimensions}


class TestEnvironmentOverrides:
    def test_prod_override_temperature_loosening_is_breaking(self):
        old = _ctx(environments={"prod": {"components": {"temperature": 0.3}}})
        new = _ctx(environments={"prod": {"components": {"temperature": 0.9}}})
        report = analyze_behavioral_impact(old, new)

        assert report.compound_severity == Severity.BREAKING
        assert _dims(report)[Dimension.INTERACTION_STYLE] == Severity.BREAKING
        # every environment impact is labeled so reviewers see the scope
        assert any("prod override" in i.reason for i in report.changed_dimensions)

    def test_prod_override_strips_guardrail_is_breaking_safety(self):
        old = _ctx()  # no environments block
        new = _ctx(environments={"prod": {"components": {"guardrails": {"blocked_topics": []}}}})
        report = analyze_behavioral_impact(old, new)

        assert _dims(report).get(Dimension.SAFETY) == Severity.BREAKING

    def test_non_prod_override_is_de_risked(self):
        old = _ctx(environments={"dev": {"components": {"temperature": 0.7}}})
        new = _ctx(environments={"dev": {"components": {"temperature": 1.2}}})
        report = analyze_behavioral_impact(old, new)

        # 0.7 -> 1.2 would be Breaking at prod, capped to Significant for dev
        assert _dims(report)[Dimension.INTERACTION_STYLE] == Severity.SIGNIFICANT
        assert report.compound_severity == Severity.SIGNIFICANT
        assert any("dev override" in i.reason for i in report.changed_dimensions)

    def test_unchanged_environments_produce_nothing(self):
        envs = {"prod": {"components": {"temperature": 0.3}}}
        assert analyze_environment_changes(_ctx(environments=envs), _ctx(environments=envs)) == []

    def test_no_double_count_when_base_and_prod_move_together(self):
        old = {"components": {"temperature": 0.3, "model": "m", "system_prompt": "x"},
               "environments": {"prod": {"components": {"temperature": 0.3}}}}
        new = {"components": {"temperature": 0.8, "model": "m", "system_prompt": "x"},
               "environments": {"prod": {"components": {"temperature": 0.8}}}}
        report = analyze_behavioral_impact(old, new)

        istyle = [i for i in report.changed_dimensions
                  if i.dimension == Dimension.INTERACTION_STYLE]
        assert len(istyle) == 1  # base pass reports it; env pass dedups

    def test_base_only_change_still_works(self):
        # regression: no environments block anywhere → identical to old behavior
        old = _ctx(temp=0.3)
        new = _ctx(temp=0.9)
        report = analyze_behavioral_impact(old, new)
        assert report.compound_severity == Severity.BREAKING
        assert not any("override" in i.reason for i in report.changed_dimensions)
