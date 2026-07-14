"""Tests for typo/punctuation-robust negation detection (Tier 4).

Found by a user: "don't" triggered negation-insertion (Breaking) while the
typo "dont" didn't (Significant) — the same semantic change scored two
different verdicts depending on an apostrophe. Negation tokens are now
normalized (case, punctuation, straight/curly apostrophes), matched with
light morphology (refuses/rejected), and guarded edit-distance-1 typo
tolerance for long distinctive words.
"""

import pytest

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.similarity import _is_negation_token


def _ctx(prompt):
    return {"components": {
        "system_prompt": prompt,
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
    }}


BASE = "You are a support agent.\nAlways share promotional offers when relevant."


class TestSpellingParity:
    @pytest.mark.parametrize("negation", ["don't", "dont", "Don't", "don’t", "do not"])
    def test_all_negation_spellings_score_identically(self, negation):
        """The verdict must not depend on apostrophe style or typos."""
        reference = analyze_behavioral_impact(
            _ctx(BASE), _ctx(BASE.replace("Always share", "You don't share"))
        )
        variant = analyze_behavioral_impact(
            _ctx(BASE), _ctx(BASE.replace("Always share", f"You {negation} share"))
        )
        assert variant.compound_severity == reference.compound_severity


class TestNegationTokenLookup:
    @pytest.mark.parametrize("token", [
        "dont", "don't", "don’t", "Don't.", "DOESN'T", "cannot", "won't,",
        "refuses", "rejected", "forbiding",   # morphology + typo
        "canot",                              # typo, 1 edit from "cannot"
    ])
    def test_true_positives(self, token):
        assert _is_negation_token(token)

    @pytest.mark.parametrize("token", [
        "either", "reuse", "eject", "fever", "lever", "black",
        "want", "ever", "share", "now", "on",
    ])
    def test_false_positive_guards(self, token):
        assert not _is_negation_token(token)

    def test_trailing_punctuation_no_longer_hides_negation(self):
        # pre-fix: "never." failed the set lookup because of the period
        assert _is_negation_token("never.")
        assert _is_negation_token("prohibit;")
