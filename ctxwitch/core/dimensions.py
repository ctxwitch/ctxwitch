"""Behavioral dimensions model for AI context analysis.

Defines the 12 behavioral dimensions that fully characterize an AI agent's
behavior surface. Every context change is mapped to one or more dimensions
with a severity level and impact description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional


class Severity(IntEnum):
    NO_CHANGE = 0
    COSMETIC = 1
    MINOR = 2
    SIGNIFICANT = 3
    BREAKING = 4

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def color(self) -> str:
        return {
            0: "dim",
            1: "dim",
            2: "yellow",
            3: "bold yellow",
            4: "bold red",
        }[self.value]


class Dimension(str, Enum):
    TONE = "tone"
    PERSONA = "persona"
    TASK_SCOPE = "task_scope"
    CONSTRAINTS = "constraints"
    OUTPUT_FORMAT = "output_format"
    SAFETY = "safety"
    KNOWLEDGE_SCOPE = "knowledge_scope"
    AUTONOMY = "autonomy"
    ERROR_HANDLING = "error_handling"
    TOOLS_CAPABILITY = "tools_capability"
    MEMORY_POLICY = "memory_policy"
    INTERACTION_STYLE = "interaction_style"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


DIMENSION_DESCRIPTIONS: Dict[Dimension, str] = {
    Dimension.TONE: "Formal/informal, empathetic/strict, friendly/authoritative",
    Dimension.PERSONA: "The role or identity the agent assumes",
    Dimension.TASK_SCOPE: "What the agent is asked to do — broader or narrower",
    Dimension.CONSTRAINTS: "What the agent must/must not do — rules and boundaries",
    Dimension.OUTPUT_FORMAT: "How output is structured — JSON, markdown, length, style",
    Dimension.SAFETY: "Guardrails, content filters, blocked topics, refusal behavior",
    Dimension.KNOWLEDGE_SCOPE: "What information the agent can access — RAG, tools, memory",
    Dimension.AUTONOMY: "How much the agent decides on its own vs. escalating",
    Dimension.ERROR_HANDLING: "How the agent responds to unclear input or failures",
    Dimension.TOOLS_CAPABILITY: "Which tools the agent can call and how",
    Dimension.MEMORY_POLICY: "What the agent remembers across sessions",
    Dimension.INTERACTION_STYLE: "Turn-taking, proactivity, verbosity, question-asking",
}


@dataclass
class DimensionImpact:
    """Impact of a change on a single behavioral dimension."""

    dimension: Dimension
    severity: Severity
    reason: str
    old_signal: str = ""
    new_signal: str = ""
    confidence: float = 1.0

    @property
    def is_changed(self) -> bool:
        return self.severity > Severity.NO_CHANGE


@dataclass
class BehavioralReport:
    """Full behavioral impact report across all dimensions."""

    impacts: List[DimensionImpact] = field(default_factory=list)
    compound_severity: Severity = Severity.NO_CHANGE
    summary: str = ""
    details: List[str] = field(default_factory=list)

    @property
    def changed_dimensions(self) -> List[DimensionImpact]:
        return [i for i in self.impacts if i.is_changed]

    @property
    def breaking_changes(self) -> List[DimensionImpact]:
        return [i for i in self.impacts if i.severity == Severity.BREAKING]

    @property
    def significant_changes(self) -> List[DimensionImpact]:
        return [i for i in self.impacts if i.severity >= Severity.SIGNIFICANT]

    def compute_compound_severity(self) -> None:
        """Compute compound severity from individual dimension impacts.

        Compound severity accounts for interaction effects:
        multiple SIGNIFICANT changes compound to BREAKING.
        """
        if not self.impacts:
            self.compound_severity = Severity.NO_CHANGE
            return

        max_severity = max(i.severity for i in self.impacts)
        significant_count = sum(1 for i in self.impacts if i.severity >= Severity.SIGNIFICANT)

        if max_severity == Severity.BREAKING:
            self.compound_severity = Severity.BREAKING
        elif significant_count >= 3:
            self.compound_severity = Severity.BREAKING
        elif significant_count >= 2:
            self.compound_severity = Severity.SIGNIFICANT
        else:
            self.compound_severity = Severity(max_severity)

        parts = []
        if self.breaking_changes:
            dims = ", ".join(i.dimension.display_name for i in self.breaking_changes)
            parts.append(f"BREAKING in {dims}")
        sig = [i for i in self.impacts if i.severity == Severity.SIGNIFICANT]
        if sig:
            dims = ", ".join(i.dimension.display_name for i in sig)
            parts.append(f"significant in {dims}")
        minor = [i for i in self.impacts if i.severity == Severity.MINOR]
        if minor:
            parts.append(f"{len(minor)} minor")
        cosmetic = [i for i in self.impacts if i.severity == Severity.COSMETIC]
        if cosmetic:
            parts.append(f"{len(cosmetic)} cosmetic")

        if parts:
            self.summary = "; ".join(parts)
        else:
            self.summary = "No behavioral changes"
