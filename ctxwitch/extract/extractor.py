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
from ctxwitch.extract.base import BehavioralSnapshot, ToolSpec, get_adapters


_MARKDOWN_EXTS = (".md", ".markdown", ".mdx", ".txt")
_CONFIG_EXTS = (".yaml", ".yml", ".json")


def _file_kind(source_file: str) -> str:
    """Classify a file so non-Python agent config isn't fed to the AST parser.

    Real agents keep their behavioral surface in more than Python: system
    prompts as ``.md``/``.txt``, config as ``.yaml``/``.json``. Route those to
    dedicated readers instead of crashing ``ast.parse`` on markdown.
    """
    s = (source_file or "").lower()
    name = Path(s).name
    if s.endswith(_MARKDOWN_EXTS):
        return "markdown"
    if s.endswith(_CONFIG_EXTS):
        return "config"
    if s.endswith(".py") or "." not in name:
        # .py, or extensionless / "<string>" (back-compat for string callers)
        return "python"
    return "other"


def _prompt_file_snapshot(source: str, source_file: str) -> BehavioralSnapshot:
    """A standalone prompt file (.md/.txt): the whole file *is* the system prompt."""
    return BehavioralSnapshot(
        name=Path(source_file).stem or "prompt",
        system_prompt=(source or "").strip(),
        source_framework="prompt-file",
        source_file=source_file,
    )


def _config_file_snapshot(source: str, source_file: str) -> Optional[BehavioralSnapshot]:
    """A config file (.yaml/.json): best-effort map of common keys to the surface."""
    name = Path(source_file).stem or "config"
    if not (source or "").strip():
        return BehavioralSnapshot(name=name, source_framework="config-file", source_file=source_file)
    try:
        if source_file.lower().endswith(".json"):
            import json as _json
            data = _json.loads(source)
        else:
            import yaml as _yaml
            data = _yaml.safe_load(source)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    comp = data.get("components") if isinstance(data.get("components"), dict) else data

    def _pick(d, *keys):
        for k in keys:
            v = d.get(k) if isinstance(d, dict) else None
            if v not in (None, ""):
                return v
        return None

    sp = _pick(comp, "system_prompt", "system", "system_message", "instruction",
               "instructions", "prompt") or ""
    sp = sp if isinstance(sp, str) else str(sp)
    model = _pick(comp, "model", "model_name", "llm") or ""
    model = model if isinstance(model, str) else str(model)
    temp = _pick(comp, "temperature")
    try:
        temp = float(temp) if temp is not None else None
    except (TypeError, ValueError):
        temp = None
    maxt = _pick(comp, "max_tokens", "maxTokens")
    try:
        maxt = int(maxt) if maxt is not None else None
    except (TypeError, ValueError):
        maxt = None

    tools: List[ToolSpec] = []
    raw_tools = _pick(comp, "tools", "tool_definitions") or []
    if isinstance(raw_tools, list):
        for t in raw_tools:
            if isinstance(t, str):
                tools.append(ToolSpec(name=t))
            elif isinstance(t, dict) and t.get("name"):
                tools.append(ToolSpec(
                    name=str(t["name"]),
                    description=str(t.get("description", "")),
                    requires_confirmation=bool(t.get("requires_confirmation", False)),
                ))

    blocked: List[str] = []
    g = _pick(comp, "guardrails")
    if isinstance(g, dict) and isinstance(g.get("blocked_topics"), list):
        blocked = [str(x) for x in g["blocked_topics"]]
    elif isinstance(comp, dict) and isinstance(comp.get("blocked_topics"), list):
        blocked = [str(x) for x in comp["blocked_topics"]]

    return BehavioralSnapshot(
        name=name, system_prompt=sp, model=model, temperature=temp, max_tokens=maxt,
        tools=tools, blocked_topics=blocked, source_framework="config-file",
        source_file=source_file,
    )


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
    kind = _file_kind(source_file)
    if kind == "markdown":
        return [_prompt_file_snapshot(source, source_file)]
    if kind == "config":
        snap = _config_file_snapshot(source, source_file)
        return [snap] if snap is not None else []
    if kind == "other":
        return []

    # kind == "python"
    try:
        tree = ast.parse(source, filename=source_file)
    except SyntaxError:
        # Not valid Python (partial file, template, non-code) — never crash a scan.
        return []
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
