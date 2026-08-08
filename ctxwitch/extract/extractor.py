"""Extraction orchestration + public API.

`extract_snapshots` walks a module's AST, hands every constructor/completion
Call to the registered adapters, and returns one BehavioralSnapshot per agent
found. `diff_code` runs CBIA over two source versions of the same file — which
is exactly how you "point ctxwitch at your own code": extract the behavioral
surface at two git revisions and diff *those*, not the raw text.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxwitch.extract.ast_utils import SymbolResolver, call_name
from ctxwitch.extract.base import BehavioralSnapshot, get_adapters


def extract_snapshots(
    source: str,
    source_file: str = "<string>",
    framework: Optional[str] = None,
) -> List[BehavioralSnapshot]:
    """Extract every agent's behavioral surface from Python source.

    Args:
        source: Python source code.
        source_file: path shown in metadata / errors.
        framework: restrict to one adapter (e.g. "adk"); None tries all.

    Returns a list of snapshots, in source order. Empty if no agent found.
    """
    tree = ast.parse(source, filename=source_file)
    resolver = SymbolResolver(tree, source_file=source_file)
    adapters = get_adapters(framework)

    # Map each Call node to the variable it is assigned to (for naming).
    assigned = _assignment_targets(tree)

    snapshots: List[BehavioralSnapshot] = []
    seen: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        key = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        if key in seen:
            continue
        for adapter in adapters:
            try:
                if adapter.matches(node, resolver):
                    snap = adapter.extract(node, assigned.get(key), resolver)
                    if snap is not None:
                        snapshots.append(snap)
                        seen.add(key)
                        break  # first adapter to claim the call wins
            except Exception:
                # A malformed call must never crash a scan of a whole repo.
                continue

    snapshots.sort(key=lambda s: s.source_line)
    return snapshots


def extract_snapshot(
    source: str,
    source_file: str = "<string>",
    framework: Optional[str] = None,
    agent: Optional[str] = None,
) -> Optional[BehavioralSnapshot]:
    """Extract a single agent snapshot.

    If `agent` is given, return the snapshot whose name or symbol matches;
    otherwise return the first agent found (the common single-agent case).
    """
    snaps = extract_snapshots(source, source_file, framework)
    if not snaps:
        return None
    if agent is None:
        return snaps[0]
    for s in snaps:
        if agent in (s.name, s.agent_symbol):
            return s
    return None


def extract_from_file(
    path: str | Path,
    framework: Optional[str] = None,
    agent: Optional[str] = None,
) -> Optional[BehavioralSnapshot]:
    p = Path(path)
    return extract_snapshot(
        p.read_text(encoding="utf-8"),
        source_file=str(p),
        framework=framework,
        agent=agent,
    )


def diff_code(
    old_source: str,
    new_source: str,
    source_file: str = "<string>",
    framework: Optional[str] = None,
    agent: Optional[str] = None,
    use_judge: bool = False,
):
    """Run CBIA between two source versions of the same agent file.

    Returns a (BehavioralReport, old_snapshot, new_snapshot) tuple. Either
    snapshot may be None if the agent could not be found in that version
    (e.g. it was just added or removed).
    """
    from ctxwitch.core.behavioral import analyze_behavioral_impact

    old_snap = extract_snapshot(old_source, source_file, framework, agent)
    new_snap = extract_snapshot(new_source, source_file, framework, agent)

    old_data: Dict[str, Any] = old_snap.to_context_dict() if old_snap else {}
    new_data: Dict[str, Any] = new_snap.to_context_dict() if new_snap else {}

    report = analyze_behavioral_impact(old_data, new_data, use_judge=use_judge)
    return report, old_snap, new_snap


def _assignment_targets(tree: ast.Module) -> Dict[tuple, str]:
    """Map (lineno, col) of a Call value node -> the name it's assigned to."""
    out: Dict[tuple, str] = {}
    for node in ast.walk(tree):
        target_name: Optional[str] = None
        value: Optional[ast.AST] = None
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            value = node.value
            if node.targets and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            value = node.value
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
        if value is not None and target_name is not None:
            out[(value.lineno, value.col_offset)] = target_name
    return out
