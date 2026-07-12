"""Config impact analyzer — deterministic behavioral impact tables for
non-prompt context parameters.

Temperature, model, tools, RAG config, guardrails, and memory changes
have known, measurable behavioral impacts. This module classifies them
with near-100% accuracy because they're structural, not linguistic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ctxwitch.core.dimensions import (
    BehavioralReport,
    Dimension,
    DimensionImpact,
    Severity,
)


def analyze_temperature_change(old: float, new: float) -> List[DimensionImpact]:
    delta = abs(new - old)
    if delta == 0:
        return []

    impacts = []
    direction = "increased" if new > old else "decreased"

    if delta >= 0.5:
        severity = Severity.BREAKING
    elif delta >= 0.3:
        severity = Severity.SIGNIFICANT
    elif delta >= 0.1:
        severity = Severity.MINOR
    else:
        severity = Severity.COSMETIC

    consistency_desc = (
        f"Temperature {direction} from {old} to {new}. "
        f"{'Response variance increases significantly — outputs become less predictable.' if new > old else 'Responses become more deterministic and consistent.'}"
    )
    impacts.append(DimensionImpact(
        dimension=Dimension.INTERACTION_STYLE,
        severity=severity,
        reason=consistency_desc,
        old_signal=str(old),
        new_signal=str(new),
    ))

    if new > 0.8:
        impacts.append(DimensionImpact(
            dimension=Dimension.SAFETY,
            severity=Severity.SIGNIFICANT if new > 1.0 else Severity.MINOR,
            reason=f"High temperature ({new}) increases hallucination risk and unpredictable outputs.",
            old_signal=str(old),
            new_signal=str(new),
        ))

    return impacts


def analyze_model_change(old: str, new: str) -> List[DimensionImpact]:
    if old == new:
        return []

    old_tier = _model_tier(old)
    new_tier = _model_tier(new)
    tier_delta = abs(new_tier - old_tier)

    if tier_delta == 0:
        severity = Severity.MINOR
        reason = f"Model changed from {old} to {new} (same capability tier). Minor behavioral differences expected."
    elif tier_delta == 1:
        severity = Severity.SIGNIFICANT
        direction = "upgraded" if new_tier > old_tier else "downgraded"
        reason = f"Model {direction} from {old} to {new}. Capability and reasoning quality will change."
    else:
        severity = Severity.BREAKING
        direction = "upgraded" if new_tier > old_tier else "downgraded"
        reason = f"Model {direction} from {old} to {new}. Major capability shift across {tier_delta} tiers."

    impacts = [
        DimensionImpact(
            dimension=Dimension.TASK_SCOPE,
            severity=severity,
            reason=reason,
            old_signal=old,
            new_signal=new,
        ),
    ]

    if new_tier < old_tier:
        impacts.append(DimensionImpact(
            dimension=Dimension.AUTONOMY,
            severity=Severity.SIGNIFICANT if tier_delta > 1 else Severity.MINOR,
            reason=f"Downgraded model may handle complex decisions less reliably.",
            old_signal=old,
            new_signal=new,
        ))

    return impacts


def analyze_tool_changes(
    old_tools: List[Dict], new_tools: List[Dict]
) -> List[DimensionImpact]:
    old_names = {t.get("name", "") for t in old_tools}
    new_names = {t.get("name", "") for t in new_tools}

    added = new_names - old_names
    removed = old_names - new_names
    common = old_names & new_names

    impacts = []

    for name in removed:
        impacts.append(DimensionImpact(
            dimension=Dimension.TOOLS_CAPABILITY,
            severity=Severity.BREAKING,
            reason=f"Tool '{name}' removed. Agent can no longer perform actions that required this tool.",
            old_signal=name,
            new_signal="(removed)",
        ))

    for name in added:
        impacts.append(DimensionImpact(
            dimension=Dimension.TOOLS_CAPABILITY,
            severity=Severity.SIGNIFICANT,
            reason=f"Tool '{name}' added. Agent gains new capability.",
            old_signal="(none)",
            new_signal=name,
        ))

    old_by_name = {t.get("name", ""): t for t in old_tools}
    new_by_name = {t.get("name", ""): t for t in new_tools}

    for name in common:
        old_t = old_by_name.get(name, {})
        new_t = new_by_name.get(name, {})

        old_desc = old_t.get("description", "")
        new_desc = new_t.get("description", "")
        if old_desc != new_desc:
            impacts.append(DimensionImpact(
                dimension=Dimension.TOOLS_CAPABILITY,
                severity=Severity.SIGNIFICANT,
                reason=f"Tool '{name}' description changed. Model may route to this tool differently.",
                old_signal=old_desc[:80],
                new_signal=new_desc[:80],
            ))

        old_confirm = old_t.get("requires_confirmation", False)
        new_confirm = new_t.get("requires_confirmation", False)
        if old_confirm != new_confirm:
            impacts.append(DimensionImpact(
                dimension=Dimension.AUTONOMY,
                severity=Severity.SIGNIFICANT,
                reason=f"Tool '{name}' confirmation requirement {'added' if new_confirm else 'removed'}.",
                old_signal=str(old_confirm),
                new_signal=str(new_confirm),
            ))

        old_params = old_t.get("parameters", {})
        new_params = new_t.get("parameters", {})
        if old_params != new_params:
            impacts.append(DimensionImpact(
                dimension=Dimension.TOOLS_CAPABILITY,
                severity=Severity.MINOR,
                reason=f"Tool '{name}' parameters changed.",
                old_signal=str(old_params)[:80],
                new_signal=str(new_params)[:80],
            ))

    return impacts


def analyze_rag_changes(
    old_rag: Dict[str, Any], new_rag: Dict[str, Any]
) -> List[DimensionImpact]:
    if old_rag == new_rag:
        return []

    impacts = []
    old_enabled = old_rag.get("enabled", False)
    new_enabled = new_rag.get("enabled", False)

    if old_enabled != new_enabled:
        if new_enabled and not old_enabled:
            impacts.append(DimensionImpact(
                dimension=Dimension.KNOWLEDGE_SCOPE,
                severity=Severity.BREAKING,
                reason="RAG enabled. Agent now retrieves from external knowledge base — responses will be grounded in documents.",
                old_signal="disabled",
                new_signal="enabled",
            ))
        else:
            impacts.append(DimensionImpact(
                dimension=Dimension.KNOWLEDGE_SCOPE,
                severity=Severity.BREAKING,
                reason="RAG disabled. Agent loses access to external knowledge base — falls back to model knowledge only.",
                old_signal="enabled",
                new_signal="disabled",
            ))
        return impacts

    if not new_enabled:
        return []

    old_top_k = old_rag.get("top_k", 5)
    new_top_k = new_rag.get("top_k", 5)
    if old_top_k != new_top_k:
        delta = abs(new_top_k - old_top_k)
        severity = Severity.SIGNIFICANT if delta >= 3 else Severity.MINOR
        impacts.append(DimensionImpact(
            dimension=Dimension.KNOWLEDGE_SCOPE,
            severity=severity,
            reason=f"RAG top_k changed from {old_top_k} to {new_top_k}. {'More' if new_top_k > old_top_k else 'Fewer'} documents retrieved per query.",
            old_signal=str(old_top_k),
            new_signal=str(new_top_k),
        ))

    old_chunk = old_rag.get("chunk_size", 512)
    new_chunk = new_rag.get("chunk_size", 512)
    if old_chunk != new_chunk:
        ratio = max(old_chunk, new_chunk) / max(min(old_chunk, new_chunk), 1)
        severity = Severity.SIGNIFICANT if ratio > 2 else Severity.MINOR
        impacts.append(DimensionImpact(
            dimension=Dimension.KNOWLEDGE_SCOPE,
            severity=severity,
            reason=f"RAG chunk_size changed from {old_chunk} to {new_chunk}. Affects retrieval granularity and context window usage.",
            old_signal=str(old_chunk),
            new_signal=str(new_chunk),
        ))

    old_embed = old_rag.get("embedding_model", "")
    new_embed = new_rag.get("embedding_model", "")
    if old_embed and new_embed and old_embed != new_embed:
        impacts.append(DimensionImpact(
            dimension=Dimension.KNOWLEDGE_SCOPE,
            severity=Severity.SIGNIFICANT,
            reason=f"Embedding model changed from {old_embed} to {new_embed}. Retrieval relevance will differ.",
            old_signal=old_embed,
            new_signal=new_embed,
        ))

    old_source = old_rag.get("source", "")
    new_source = new_rag.get("source", "")
    if old_source != new_source:
        impacts.append(DimensionImpact(
            dimension=Dimension.KNOWLEDGE_SCOPE,
            severity=Severity.BREAKING,
            reason=f"RAG source changed. Agent now retrieves from a different knowledge base.",
            old_signal=old_source[:60] or "(none)",
            new_signal=new_source[:60] or "(none)",
        ))

    return impacts


def analyze_guardrail_changes(
    old_guard: Dict[str, Any], new_guard: Dict[str, Any]
) -> List[DimensionImpact]:
    if old_guard == new_guard:
        return []

    impacts = []

    old_blocked = set(old_guard.get("blocked_topics", []))
    new_blocked = set(new_guard.get("blocked_topics", []))

    removed_topics = old_blocked - new_blocked
    added_topics = new_blocked - old_blocked

    for topic in removed_topics:
        impacts.append(DimensionImpact(
            dimension=Dimension.SAFETY,
            severity=Severity.BREAKING,
            reason=f"Blocked topic '{topic}' removed. Agent may now engage with this topic.",
            old_signal=topic,
            new_signal="(unblocked)",
        ))

    for topic in added_topics:
        impacts.append(DimensionImpact(
            dimension=Dimension.SAFETY,
            severity=Severity.SIGNIFICANT,
            reason=f"Topic '{topic}' now blocked. Agent will refuse to engage.",
            old_signal="(allowed)",
            new_signal=topic,
        ))

    old_filters = set(old_guard.get("output_filters", []))
    new_filters = set(new_guard.get("output_filters", []))

    for f in old_filters - new_filters:
        impacts.append(DimensionImpact(
            dimension=Dimension.SAFETY,
            severity=Severity.BREAKING,
            reason=f"Output filter '{f}' removed. Outputs are no longer filtered.",
            old_signal=f,
            new_signal="(removed)",
        ))

    for f in new_filters - old_filters:
        impacts.append(DimensionImpact(
            dimension=Dimension.SAFETY,
            severity=Severity.SIGNIFICANT,
            reason=f"Output filter '{f}' added.",
            old_signal="(none)",
            new_signal=f,
        ))

    old_turns = old_guard.get("max_turns", 0)
    new_turns = new_guard.get("max_turns", 0)
    if old_turns != new_turns:
        severity = Severity.MINOR
        if new_turns == 0 and old_turns > 0:
            severity = Severity.SIGNIFICANT
        elif old_turns == 0 and new_turns > 0:
            severity = Severity.MINOR
        impacts.append(DimensionImpact(
            dimension=Dimension.INTERACTION_STYLE,
            severity=severity,
            reason=f"Max turns changed from {old_turns or 'unlimited'} to {new_turns or 'unlimited'}.",
            old_signal=str(old_turns),
            new_signal=str(new_turns),
        ))

    return impacts


def analyze_memory_changes(
    old_mem: Dict[str, Any], new_mem: Dict[str, Any]
) -> List[DimensionImpact]:
    if old_mem == new_mem:
        return []

    impacts = []
    old_enabled = old_mem.get("enabled", False)
    new_enabled = new_mem.get("enabled", False)

    if old_enabled != new_enabled:
        impacts.append(DimensionImpact(
            dimension=Dimension.MEMORY_POLICY,
            severity=Severity.BREAKING,
            reason=f"Memory {'enabled' if new_enabled else 'disabled'}. Agent {'now retains' if new_enabled else 'no longer retains'} information across sessions.",
            old_signal="enabled" if old_enabled else "disabled",
            new_signal="enabled" if new_enabled else "disabled",
        ))
        return impacts

    if not new_enabled:
        return []

    old_policy = old_mem.get("write_policy", "on_trigger")
    new_policy = new_mem.get("write_policy", "on_trigger")
    if old_policy != new_policy:
        impacts.append(DimensionImpact(
            dimension=Dimension.MEMORY_POLICY,
            severity=Severity.SIGNIFICANT,
            reason=f"Memory write policy changed from '{old_policy}' to '{new_policy}'.",
            old_signal=old_policy,
            new_signal=new_policy,
        ))

    old_retention = old_mem.get("retention_days", 30)
    new_retention = new_mem.get("retention_days", 30)
    if old_retention != new_retention:
        impacts.append(DimensionImpact(
            dimension=Dimension.MEMORY_POLICY,
            severity=Severity.MINOR,
            reason=f"Memory retention changed from {old_retention} to {new_retention} days.",
            old_signal=str(old_retention),
            new_signal=str(new_retention),
        ))

    return impacts


def analyze_max_tokens_change(old: int, new: int) -> List[DimensionImpact]:
    if old == new:
        return []

    ratio = new / max(old, 1)
    if ratio > 2 or ratio < 0.5:
        severity = Severity.SIGNIFICANT
    elif ratio > 1.5 or ratio < 0.67:
        severity = Severity.MINOR
    else:
        severity = Severity.COSMETIC

    direction = "increased" if new > old else "decreased"
    return [DimensionImpact(
        dimension=Dimension.OUTPUT_FORMAT,
        severity=severity,
        reason=f"Max tokens {direction} from {old} to {new}. {'Longer' if new > old else 'Shorter'} responses expected.",
        old_signal=str(old),
        new_signal=str(new),
    )]


_TIER_REGISTRY: Optional[Dict[str, Any]] = None


def _load_tier_registry() -> Dict[str, Any]:
    """Load the model tier registry.

    Models ship and rename faster than this library releases, so tiers live
    in data (ctxwitch/core/model_tiers.yaml), overridable per-deployment via
    the CTXWITCH_MODEL_TIERS environment variable.
    """
    global _TIER_REGISTRY
    if _TIER_REGISTRY is not None:
        return _TIER_REGISTRY

    import os

    import yaml

    override = os.environ.get("CTXWITCH_MODEL_TIERS")
    path = Path(override) if override else Path(__file__).parent / "model_tiers.yaml"
    with open(path) as f:
        _TIER_REGISTRY = yaml.safe_load(f)
    return _TIER_REGISTRY


def _model_tier(model_name: str) -> int:
    """Classify a model into a capability tier via the tier registry."""
    registry = _load_tier_registry()
    name = model_name.lower()

    for rule in registry.get("rules", []):
        if not any(p in name for p in rule.get("patterns", [])):
            continue
        if any(p in name for p in rule.get("unless_patterns", [])):
            continue
        return int(rule["tier"])

    return int(registry.get("default_tier", 1))
