"""Behavioral snapshot model + framework-adapter registry.

The extractor's whole job is to turn *agent code* (Google ADK, LangGraph,
a raw Anthropic/OpenAI call) into the same canonical dict shape that CBIA
already consumes for witch.yaml — so `analyze_behavioral_impact(old, new)`
works identically whether the behavioral surface was declared in YAML or
written in Python.

Canonical shape (subset of CONTEXT_SCHEMA that carries behavioral signal):

    {
      "version": "v0.0.0",
      "name": "<agent name>",
      "components": {
        "system_prompt": str,
        "model": str,
        "temperature": float,
        "max_tokens": int,
        "tool_definitions": [{"name", "description", "requires_confirmation"}],
        "guardrails": {"blocked_topics": [...], ...},
      },
      "metadata": {"source_framework", "source_file", "source_line",
                   "agent_symbol", "dynamic_fields": [...]},
    }
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """A tool the agent can call, normalized to CBIA's tool_definitions shape."""

    name: str
    description: str = ""
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class BehavioralSnapshot:
    """An agent's behavioral surface, extracted from code.

    `dynamic_fields` records anything the static extractor could not fully
    resolve (an f-string with runtime interpolation, a model name read from
    an env var, a tool list built in a loop). CBIA still runs on what *was*
    resolved; the field is surfaced so a reviewer knows the snapshot is
    partial rather than silently trusting a blank.
    """

    name: str
    system_prompt: str = ""
    model: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: List[ToolSpec] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    source_framework: str = ""
    source_file: str = ""
    source_line: int = 0
    agent_symbol: str = ""
    dynamic_fields: List[str] = field(default_factory=list)

    def to_context_dict(self) -> Dict[str, Any]:
        """Emit the canonical dict CBIA's analyze_behavioral_impact expects."""
        components: Dict[str, Any] = {
            "system_prompt": self.system_prompt,
            "model": self.model,
        }
        if self.temperature is not None:
            components["temperature"] = self.temperature
        if self.max_tokens is not None:
            components["max_tokens"] = self.max_tokens
        if self.tools:
            components["tool_definitions"] = [t.to_dict() for t in self.tools]
        if self.blocked_topics:
            components["guardrails"] = {"blocked_topics": list(self.blocked_topics)}

        return {
            "version": "v0.0.0",
            "name": self.name,
            "components": components,
            "metadata": {
                "source_framework": self.source_framework,
                "source_file": self.source_file,
                "source_line": self.source_line,
                "agent_symbol": self.agent_symbol,
                "dynamic_fields": list(self.dynamic_fields),
            },
        }


class FrameworkAdapter:
    """Base class for framework-specific extraction.

    An adapter recognizes the constructor call(s) that declare an agent in a
    given framework and maps their keyword arguments onto a BehavioralSnapshot.
    Adapters are static: they read the AST and never import or execute the
    target code, so `witch scan` is safe to run in CI on untrusted diffs.
    """

    #: short id used in metadata + the --framework flag (e.g. "adk")
    name: str = ""

    def matches(self, call: ast.Call, resolver: "SymbolResolver") -> bool:
        """Return True if this Call node constructs an agent this adapter owns."""
        raise NotImplementedError

    def extract(
        self,
        call: ast.Call,
        assigned_to: Optional[str],
        resolver: "SymbolResolver",
    ) -> Optional[BehavioralSnapshot]:
        """Build a snapshot from a matched constructor call."""
        raise NotImplementedError


# ── adapter registry ────────────────────────────────────────────────────────

_ADAPTERS: List[FrameworkAdapter] = []


def register_adapter(adapter_cls):
    """Class decorator: instantiate the adapter and add it to the registry.

    Returns the class unchanged so the decorated name still refers to the
    class, while the registry holds a ready-to-use instance.
    """
    _ADAPTERS.append(adapter_cls())
    return adapter_cls


def get_adapters(framework: Optional[str] = None) -> List[FrameworkAdapter]:
    """All registered adapters, or just the one named `framework`."""
    if framework is None:
        return list(_ADAPTERS)
    return [a for a in _ADAPTERS if a.name == framework]


def adapter_names() -> List[str]:
    return [a.name for a in _ADAPTERS]


# Forward reference for type checkers; real class lives in ast_utils.
if False:  # pragma: no cover
    from ctxwitch.extract.ast_utils import SymbolResolver
