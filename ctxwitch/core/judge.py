"""Tier 6: LLM-as-judge — optional behavioral impact analysis using an LLM.

Only invoked when Tiers 1-5 detect SIGNIFICANT+ changes in subjective
dimensions (tone, persona, interaction_style) where heuristic analysis
has limited accuracy.

Supports multiple providers (OpenAI, Anthropic) via API keys.
Falls back gracefully when no API key is configured.

This tier is surgical: it only sends the changed segments (not the full
prompt), saving ~80% of token cost vs. sending everything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ctxwitch.core.dimensions import (
    BehavioralReport,
    Dimension,
    DimensionImpact,
    Severity,
)

SUBJECTIVE_DIMENSIONS = {
    Dimension.TONE,
    Dimension.PERSONA,
    Dimension.INTERACTION_STYLE,
    Dimension.AUTONOMY,
    Dimension.ERROR_HANDLING,
}

JUDGE_RUBRIC = """\
You are an expert at analyzing how changes to AI system prompts affect agent behavior.

Given an OLD and NEW version of specific prompt segments, analyze the behavioral impact across these dimensions:

1. **Tone**: How does the agent's communication style change? (formal/informal, warm/cold, empathetic/strict)
2. **Persona**: How does the agent's identity or role change?
3. **Autonomy**: Does the agent make more or fewer independent decisions?
4. **Interaction Style**: How does the conversation flow change? (proactive/reactive, verbose/concise)
5. **Error Handling**: How does the agent handle ambiguous or difficult situations differently?

For each dimension that changed, provide:
- dimension: the dimension name
- severity: one of "cosmetic", "minor", "significant", "breaking"
- explanation: 1-2 sentences explaining the behavioral impact in plain language
- prediction: what specific behavioral difference a user would notice

Respond ONLY with valid JSON in this format:
{
  "analysis": [
    {
      "dimension": "tone",
      "severity": "significant",
      "explanation": "...",
      "prediction": "..."
    }
  ],
  "overall_summary": "One sentence summary of the compound behavioral shift."
}

If there are no meaningful behavioral changes, return {"analysis": [], "overall_summary": "No meaningful behavioral changes."}
"""


@dataclass
class JudgeResult:
    """Result from the LLM-as-judge analysis."""

    available: bool
    provider: str = ""
    analysis: List[Dict[str, str]] = field(default_factory=list)
    overall_summary: str = ""
    error: str = ""
    token_cost: int = 0


def needs_judge(report: BehavioralReport) -> bool:
    """Determine if the LLM judge should be invoked.

    Only triggers when subjective dimensions have SIGNIFICANT+ severity
    from Tiers 1-5, where heuristic confidence is lower.
    """
    for impact in report.impacts:
        if impact.dimension in SUBJECTIVE_DIMENSIONS:
            if impact.severity >= Severity.SIGNIFICANT:
                if impact.confidence < 0.85:
                    return True
    return False


def get_provider() -> Optional[str]:
    """Detect which LLM provider is available via environment variables."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def run_judge(
    old_prompt: str,
    new_prompt: str,
    changed_segments: List[Dict[str, str]],
    provider: Optional[str] = None,
) -> JudgeResult:
    """Run the LLM-as-judge on changed prompt segments.

    Only sends the segments that actually changed — not the full prompt.
    This is the surgical approach that saves ~80% of token cost.
    """
    provider = provider or get_provider()

    if not provider:
        return JudgeResult(
            available=False,
            error="No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for Tier 6 analysis.",
        )

    segments_text = _format_segments_for_judge(changed_segments)

    user_message = f"""Analyze the behavioral impact of these prompt changes:

OLD PROMPT:
{old_prompt}

NEW PROMPT:
{new_prompt}

SPECIFIC CHANGES DETECTED BY AUTOMATED ANALYSIS:
{segments_text}

Provide your behavioral impact analysis as JSON."""

    try:
        if provider == "anthropic":
            return _call_anthropic(user_message)
        elif provider == "openai":
            return _call_openai(user_message)
        else:
            return JudgeResult(available=False, error=f"Unknown provider: {provider}")
    except Exception as e:
        return JudgeResult(available=False, error=f"Judge call failed: {e}")


def judge_to_impacts(result: JudgeResult) -> List[DimensionImpact]:
    """Convert judge results into dimension impacts."""
    if not result.available or not result.analysis:
        return []

    severity_map = {
        "cosmetic": Severity.COSMETIC,
        "minor": Severity.MINOR,
        "significant": Severity.SIGNIFICANT,
        "breaking": Severity.BREAKING,
    }

    dimension_map = {
        "tone": Dimension.TONE,
        "persona": Dimension.PERSONA,
        "autonomy": Dimension.AUTONOMY,
        "interaction_style": Dimension.INTERACTION_STYLE,
        "interaction style": Dimension.INTERACTION_STYLE,
        "error_handling": Dimension.ERROR_HANDLING,
        "error handling": Dimension.ERROR_HANDLING,
    }

    impacts = []
    for entry in result.analysis:
        dim_name = entry.get("dimension", "").lower().strip()
        dimension = dimension_map.get(dim_name)
        if not dimension:
            continue

        severity = severity_map.get(entry.get("severity", "").lower(), Severity.MINOR)
        explanation = entry.get("explanation", "")
        prediction = entry.get("prediction", "")

        reason = explanation
        if prediction:
            reason = f"{explanation} Predicted impact: {prediction}"

        impacts.append(DimensionImpact(
            dimension=dimension,
            severity=severity,
            reason=f"[LLM Judge] {reason}",
            confidence=0.90,
        ))

    return impacts


def _format_segments_for_judge(segments: List[Dict[str, str]]) -> str:
    lines = []
    for seg in segments:
        change_type = seg.get("change_type", "modified")
        seg_type = seg.get("segment_type", "unknown")
        if change_type == "removed":
            lines.append(f"  REMOVED [{seg_type}]: {seg.get('old_text', '')}")
        elif change_type == "added":
            lines.append(f"  ADDED [{seg_type}]: {seg.get('new_text', '')}")
        else:
            lines.append(f"  CHANGED [{seg_type}]:")
            lines.append(f"    OLD: {seg.get('old_text', '')}")
            lines.append(f"    NEW: {seg.get('new_text', '')}")
            sim = seg.get("similarity", "")
            if sim:
                lines.append(f"    Similarity: {sim}")
    return "\n".join(lines) if lines else "(no specific segments)"


def _call_anthropic(user_message: str) -> JudgeResult:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=JUDGE_RUBRIC,
        messages=[{"role": "user", "content": user_message}],
    )

    text = response.content[0].text
    token_cost = response.usage.input_tokens + response.usage.output_tokens

    return _parse_judge_response(text, "anthropic", token_cost)


def _call_openai(user_message: str) -> JudgeResult:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": user_message},
        ],
    )

    text = response.choices[0].message.content
    token_cost = response.usage.total_tokens if response.usage else 0

    return _parse_judge_response(text, "openai", token_cost)


def _parse_judge_response(text: str, provider: str, token_cost: int) -> JudgeResult:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return JudgeResult(
            available=True,
            provider=provider,
            overall_summary=text[:200],
            token_cost=token_cost,
            error="Failed to parse LLM response as JSON",
        )

    return JudgeResult(
        available=True,
        provider=provider,
        analysis=data.get("analysis", []),
        overall_summary=data.get("overall_summary", ""),
        token_cost=token_cost,
    )
