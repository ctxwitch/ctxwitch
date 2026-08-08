"""Behavioral-surface extraction from agent code.

Turns framework code (Google ADK, LangGraph, raw Anthropic/OpenAI calls) into
the canonical snapshot CBIA already understands, so `witch diff` works on a
repo of Python agents — not just on witch.yaml. Static/AST-based: scanning
never imports or executes the target code.

    from ctxwitch.extract import extract_from_file, diff_code

    snap = extract_from_file("agent.py", framework="adk")
    report, old, new = diff_code(old_src, new_src, framework="adk")
"""

from __future__ import annotations

from ctxwitch.extract.base import (
    BehavioralSnapshot,
    ToolSpec,
    adapter_names,
)
from ctxwitch.extract.extractor import (
    diff_code,
    extract_from_file,
    extract_snapshot,
    extract_snapshots,
)

# Import adapters for their registration side effects.
from ctxwitch.extract.adapters import adk as _adk  # noqa: F401
from ctxwitch.extract.adapters import generic as _generic  # noqa: F401

__all__ = [
    "BehavioralSnapshot",
    "ToolSpec",
    "adapter_names",
    "extract_snapshot",
    "extract_snapshots",
    "extract_from_file",
    "diff_code",
]
