"""Tests for numeric threshold change detection (Tier 4).

Covers the CBIA blind spot found while building the tour: changing only a
number inside a directive ("$100" -> "$500") is textually near-identical,
so Tier 3 similarity scores it Cosmetic at best — but the number is the
behaviorally load-bearing part of the rule.
"""

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.dimensions import Dimension, Severity


def _ctx(prompt: str) -> dict:
    return {
        "components": {
            "system_prompt": prompt,
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
        }
    }

BASE_PROMPT = (
    "You are a customer support agent.\n"
    "Always verify identity before discussing account details.\n"
    "You must escalate any refund above $100 to a human agent.\n"
    "Respond clearly and concisely."
)


def _impact_for(report, dimension):
    return next(i for i in report.impacts if i.dimension == dimension)


class TestNumericThresholdDetection:
    def test_dollar_threshold_change_is_significant(self):
        new = _ctx(BASE_PROMPT.replace("$100", "$500"))
        report = analyze_behavioral_impact(_ctx(BASE_PROMPT), new)

        assert report.compound_severity >= Severity.SIGNIFICANT
        flagged = [
            i for i in report.impacts
            if i.severity >= Severity.SIGNIFICANT and "$100 → $500" in i.reason
        ]
        assert flagged, "threshold change was not flagged"

    def test_day_count_change_in_decision_rule(self):
        old = _ctx(BASE_PROMPT + "\nIf the order is older than 30 days, reject the refund.")
        new = _ctx(BASE_PROMPT + "\nIf the order is older than 90 days, reject the refund.")
        report = analyze_behavioral_impact(old, new)

        assert any(
            "30 → 90" in i.reason and i.severity >= Severity.SIGNIFICANT
            for i in report.impacts
        )

    def test_percentage_change_detected(self):
        old = _ctx(BASE_PROMPT + "\nNever offer discounts above 10%.")
        new = _ctx(BASE_PROMPT + "\nNever offer discounts above 25%.")
        report = analyze_behavioral_impact(old, new)

        assert any("10% → 25%" in i.reason for i in report.impacts)

    def test_identical_numbers_not_flagged(self):
        report = analyze_behavioral_impact(_ctx(BASE_PROMPT), _ctx(BASE_PROMPT))
        assert report.compound_severity == Severity.NO_CHANGE

    def test_rewrite_with_number_change_not_double_flagged(self):
        """A real rewrite that happens to change a number is Tier 3's job —
        the threshold detector must stay quiet to avoid double counting."""
        new = _ctx(BASE_PROMPT.replace(
            "You must escalate any refund above $100 to a human agent.",
            "Refund handling is delegated to the payments team beyond $500.",
        ))
        report = analyze_behavioral_impact(_ctx(BASE_PROMPT), new)

        # still flagged as changed (by Tier 3), but not as a pure threshold shift
        assert report.compound_severity >= Severity.MINOR
        assert not any("only its trigger value moved" in i.reason for i in report.impacts)

    def test_light_rewording_with_number_change_still_flagged(self):
        new = _ctx(BASE_PROMPT.replace(
            "You must escalate any refund above $100 to a human agent.",
            "You must escalate every refund above $250 to a human agent.",
        ))
        report = analyze_behavioral_impact(_ctx(BASE_PROMPT), new)

        flagged = [i for i in report.impacts if "$100 → $250" in i.reason]
        assert flagged
        assert flagged[0].confidence < 0.95  # reworded → lower confidence

    def test_number_added_to_directive(self):
        old = _ctx(BASE_PROMPT + "\nAlways limit conversations appropriately.")
        new = _ctx(BASE_PROMPT + "\nAlways limit conversations appropriately to 20 turns.")
        report = analyze_behavioral_impact(old, new)
        # accepted either as threshold addition or segment modification —
        # it must not be invisible
        assert report.compound_severity > Severity.NO_CHANGE
