"""Compound Behavioral Impact Analysis (CBIA) — the core ctxwitch algorithm.

Unique 6-tier pipeline that analyzes the full behavioral impact of any
context change across 12 dimensions:

  Tier 1: Structural decomposition — config parameter deltas (<10ms)
  Tier 2: Prompt decomposition — parse into typed segments (~20ms)
  Tier 3: Segment similarity — embedding or token-based matching (~50ms)
  Tier 4: Directive contradiction — detect reversed/negated rules (~10ms)
  Tier 5: Dimension scoring — aggregate into compound severity (<5ms)
  Tier 6: LLM-as-judge — subjective behavioral analysis (~3s, optional)

Tiers 1-5 are fully local and deterministic (no LLM calls required).
Tier 6 is optional and only invoked for subjective dimensions with
SIGNIFICANT+ severity where heuristic confidence is low.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ctxwitch.core.decompose import DecomposedPrompt, decompose_prompt
from ctxwitch.core.dimensions import (
    BehavioralReport,
    Dimension,
    DimensionImpact,
    Severity,
)
from ctxwitch.core.impact import (
    analyze_guardrail_changes,
    analyze_max_tokens_change,
    analyze_memory_changes,
    analyze_model_change,
    analyze_rag_changes,
    analyze_temperature_change,
    analyze_tool_changes,
)
from ctxwitch.core.similarity import (
    analyze_segment_changes,
    contradictions_to_impacts,
    detect_contradictions,
    match_segments,
)


def analyze_behavioral_impact(
    old_data: Dict[str, Any],
    new_data: Dict[str, Any],
    use_judge: bool = False,
) -> BehavioralReport:
    """Run the full CBIA pipeline on two context snapshots.

    Args:
        old_data: Previous witch.yaml data
        new_data: Current witch.yaml data
        use_judge: If True, invoke Tier 6 LLM-as-judge for subjective
                   dimensions (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)

    Returns a BehavioralReport with per-dimension impact scores,
    compound severity, and human-readable explanations.
    """
    all_impacts: List[DimensionImpact] = []

    # ── Tier 1: Config parameter deltas (deterministic, <10ms) ──────────

    old_comp = old_data.get("components", {})
    new_comp = new_data.get("components", {})

    old_temp = old_comp.get("temperature")
    new_temp = new_comp.get("temperature")
    if old_temp is not None and new_temp is not None:
        all_impacts.extend(analyze_temperature_change(old_temp, new_temp))

    old_model = old_comp.get("model", "")
    new_model = new_comp.get("model", "")
    if old_model or new_model:
        all_impacts.extend(analyze_model_change(old_model, new_model))

    old_tools = old_comp.get("tool_definitions", [])
    new_tools = new_comp.get("tool_definitions", [])
    all_impacts.extend(analyze_tool_changes(old_tools, new_tools))

    old_rag = old_comp.get("rag_config", {})
    new_rag = new_comp.get("rag_config", {})
    all_impacts.extend(analyze_rag_changes(old_rag, new_rag))

    old_guard = old_comp.get("guardrails", {})
    new_guard = new_comp.get("guardrails", {})
    all_impacts.extend(analyze_guardrail_changes(old_guard, new_guard))

    old_mem = old_comp.get("memory", {})
    new_mem = new_comp.get("memory", {})
    all_impacts.extend(analyze_memory_changes(old_mem, new_mem))

    old_max = old_comp.get("max_tokens", 0)
    new_max = new_comp.get("max_tokens", 0)
    if old_max and new_max:
        all_impacts.extend(analyze_max_tokens_change(old_max, new_max))

    # ── Tier 2+3: Prompt decomposition + segment similarity ─────────────

    old_prompt = old_comp.get("system_prompt", "")
    new_prompt = new_comp.get("system_prompt", "")
    segment_matches = None

    if old_prompt != new_prompt:
        old_decomposed = decompose_prompt(old_prompt)
        new_decomposed = decompose_prompt(new_prompt)

        segment_matches = match_segments(old_decomposed, new_decomposed)
        segment_impacts = analyze_segment_changes(segment_matches)
        all_impacts.extend(segment_impacts)

        # ── Tier 4: Directive contradiction detection ───────────────────

        contradictions = detect_contradictions(old_decomposed, new_decomposed)
        contradiction_impacts = contradictions_to_impacts(contradictions)
        all_impacts.extend(contradiction_impacts)

    # ── Tier 5: Dimension scoring + compound severity ───────────────────

    report = _build_report(all_impacts)

    # ── Tier 6: LLM-as-judge (optional, ~3s) ───────────────────────────

    if use_judge and old_prompt != new_prompt:
        _run_judge_tier(report, old_prompt, new_prompt, segment_matches)

    return report


def _run_judge_tier(
    report: BehavioralReport,
    old_prompt: str,
    new_prompt: str,
    segment_matches: Optional[list],
) -> None:
    """Invoke Tier 6 LLM-as-judge if needed and available.

    Only runs when subjective dimensions have SIGNIFICANT+ changes with
    low heuristic confidence. Merges judge results into the report,
    upgrading dimension severities if the judge finds higher impact.
    """
    from ctxwitch.core.judge import (
        get_provider,
        judge_to_impacts,
        needs_judge,
        run_judge,
    )

    if not needs_judge(report):
        return

    provider = get_provider()
    if not provider:
        report.details.append(
            "[Tier 6] Skipped: no API key. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )
        return

    changed_segments = []
    if segment_matches:
        for m in segment_matches:
            if m.match_type == "matched" and m.similarity > 0.85:
                continue
            seg_info = {"change_type": m.match_type}
            if m.old_segment:
                seg_info["old_text"] = m.old_segment.text
                seg_info["segment_type"] = m.old_segment.segment_type.value
            if m.new_segment:
                seg_info["new_text"] = m.new_segment.text
                seg_info["segment_type"] = m.new_segment.segment_type.value
            if m.similarity > 0:
                seg_info["similarity"] = f"{m.similarity:.0%}"
            changed_segments.append(seg_info)

    result = run_judge(old_prompt, new_prompt, changed_segments, provider)

    if not result.available:
        if result.error:
            report.details.append(f"[Tier 6] {result.error}")
        return

    judge_impacts = judge_to_impacts(result)

    for ji in judge_impacts:
        for i, existing in enumerate(report.impacts):
            if existing.dimension == ji.dimension and ji.severity > existing.severity:
                report.impacts[i] = ji
                break

    if result.overall_summary:
        report.details.append(f"[Tier 6 — {result.provider}] {result.overall_summary}")

    if result.token_cost:
        report.details.append(f"[Tier 6] Token cost: {result.token_cost}")

    report.compute_compound_severity()


def _build_report(impacts: List[DimensionImpact]) -> BehavioralReport:
    """Aggregate raw impacts into a per-dimension report.

    For each dimension, takes the highest severity impact and merges
    reasons from all impacts on the same dimension so nothing is lost.
    """
    from collections import defaultdict

    dim_impacts: Dict[Dimension, List[DimensionImpact]] = defaultdict(list)
    for impact in impacts:
        dim_impacts[impact.dimension].append(impact)

    final_impacts: List[DimensionImpact] = []
    for dim in Dimension:
        group = dim_impacts.get(dim)
        if not group:
            final_impacts.append(DimensionImpact(
                dimension=dim,
                severity=Severity.NO_CHANGE,
                reason="No change detected",
            ))
            continue

        group.sort(key=lambda i: i.severity, reverse=True)
        best = group[0]

        if len(group) > 1:
            extra_reasons = [
                g.reason for g in group[1:]
                if g.severity >= Severity.MINOR and g.reason != best.reason
            ]
            if extra_reasons:
                combined = best.reason + "; " + "; ".join(extra_reasons[:2])
                best = DimensionImpact(
                    dimension=best.dimension,
                    severity=best.severity,
                    reason=combined,
                    old_signal=best.old_signal,
                    new_signal=best.new_signal,
                    confidence=best.confidence,
                )

        final_impacts.append(best)

    report = BehavioralReport(impacts=final_impacts)
    report.compute_compound_severity()

    details = []
    for impact in impacts:
        if impact.severity >= Severity.MINOR:
            details.append(
                f"[{impact.severity.label}] {impact.dimension.display_name}: {impact.reason}"
            )
    report.details = details

    return report


def format_behavioral_report(report: BehavioralReport) -> str:
    """Format a behavioral report as a human-readable string."""
    lines = []
    lines.append(f"Compound Severity: {report.compound_severity.label}")
    lines.append(f"Summary: {report.summary}")
    lines.append("")

    changed = report.changed_dimensions
    if not changed:
        lines.append("No behavioral changes detected.")
        return "\n".join(lines)

    lines.append("Dimension Scorecard:")
    for impact in report.impacts:
        if impact.severity == Severity.NO_CHANGE:
            continue
        icon = {
            Severity.COSMETIC: ".",
            Severity.MINOR: "~",
            Severity.SIGNIFICANT: "!",
            Severity.BREAKING: "X",
        }.get(impact.severity, " ")

        lines.append(f"  [{icon}] {impact.dimension.display_name}: {impact.severity.label}")
        lines.append(f"      {impact.reason}")
        if impact.old_signal and impact.new_signal and impact.old_signal != "(none)":
            old_short = impact.old_signal[:60].replace("\n", " ")
            new_short = impact.new_signal[:60].replace("\n", " ")
            if impact.new_signal != "(removed)":
                lines.append(f"      - {old_short}")
                lines.append(f"      + {new_short}")

    return "\n".join(lines)
