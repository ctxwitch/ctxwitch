"""Google ADK adapter.

Recognizes the ADK agent constructors, e.g.:

    from google.adk.agents import LlmAgent

    root_agent = LlmAgent(
        name="support",
        model="gemini-2.0-flash",
        instruction=SUPPORT_PROMPT,
        tools=[search_kb, lookup_account, escalate],
    )

Maps ADK's `instruction` -> system_prompt, `model` -> model,
`generate_content_config.temperature` (or a top-level `temperature`) ->
temperature, and the `tools=[...]` list -> tool_definitions, pulling each
tool's description from the referenced function's docstring.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from ctxwitch.extract.ast_utils import (
    SymbolResolver,
    call_name,
    first_line,
    get_kwarg,
)
from ctxwitch.extract.base import (
    BehavioralSnapshot,
    FrameworkAdapter,
    ToolSpec,
    register_adapter,
)

# ADK agent constructor names, matched on the trailing identifier so both
# `LlmAgent(...)` and `agents.LlmAgent(...)` / `adk.Agent(...)` are caught.
_ADK_CTORS = {"LlmAgent", "Agent", "SequentialAgent", "LoopAgent", "ParallelAgent"}

# Tool-name substrings that conventionally require human confirmation. ADK
# has no declarative flag for this, so we infer it (and record nothing false:
# only a positive hit sets the flag).
_CONFIRM_HINTS = ("escalate", "delete", "refund", "transfer", "purchase", "approve")


@register_adapter
class ADKAdapter(FrameworkAdapter):
    name = "adk"

    def matches(self, call: ast.Call, resolver: SymbolResolver) -> bool:
        cname = call_name(call)
        if not cname:
            return False
        return cname.split(".")[-1] in _ADK_CTORS

    def extract(
        self,
        call: ast.Call,
        assigned_to: Optional[str],
        resolver: SymbolResolver,
    ) -> Optional[BehavioralSnapshot]:
        dynamic: List[str] = []

        name_val, ok = resolver.resolve_str(get_kwarg(call, "name"))
        agent_name = name_val if ok else (assigned_to or "agent")

        snap = BehavioralSnapshot(
            name=agent_name,
            source_framework=self.name,
            source_file=resolver.source_file,
            source_line=first_line(call),
            agent_symbol=assigned_to or "",
        )

        # instruction -> system_prompt
        instr_node = get_kwarg(call, "instruction")
        if instr_node is None:
            instr_node = get_kwarg(call, "system_instruction")
        prompt, ok = resolver.resolve_str(instr_node)
        if ok:
            snap.system_prompt = prompt
        elif instr_node is not None:
            dynamic.append("system_prompt")

        # model
        model, ok = resolver.resolve_str(get_kwarg(call, "model"))
        if ok:
            snap.model = model
        elif get_kwarg(call, "model") is not None:
            dynamic.append("model")

        # temperature: top-level, or nested in generate_content_config=...
        self._extract_temperature(call, resolver, snap, dynamic)

        # tools=[...]
        self._extract_tools(call, resolver, snap, dynamic)

        snap.dynamic_fields = dynamic
        return snap

    def _extract_temperature(self, call, resolver, snap, dynamic) -> None:
        temp_node = get_kwarg(call, "temperature")
        if temp_node is None:
            cfg = get_kwarg(call, "generate_content_config")
            if isinstance(cfg, ast.Call):
                temp_node = get_kwarg(cfg, "temperature")
        if temp_node is None:
            return
        val = resolver.resolve(temp_node)
        if isinstance(val, (int, float)):
            snap.temperature = float(val)
        else:
            dynamic.append("temperature")

    def _extract_tools(self, call, resolver, snap, dynamic) -> None:
        tools_node = get_kwarg(call, "tools")
        if tools_node is None:
            return
        if not isinstance(tools_node, (ast.List, ast.Tuple)):
            dynamic.append("tool_definitions")
            return

        for elt in tools_node.elts:
            spec = self._tool_from_node(elt, resolver)
            if spec is None:
                dynamic.append("tool_definitions")
            else:
                snap.tools.append(spec)

    def _tool_from_node(self, node: ast.AST, resolver: SymbolResolver) -> Optional[ToolSpec]:
        # bare function reference: tools=[search_kb]
        if isinstance(node, ast.Name):
            fn_name = node.id
            desc = resolver.function_docstring(fn_name)
            return ToolSpec(
                name=fn_name,
                description=desc,
                requires_confirmation=_infer_confirm(fn_name),
            )

        # wrapped: FunctionTool(func=search_kb) or FunctionTool(search_kb)
        if isinstance(node, ast.Call):
            cname = (call_name(node) or "").split(".")[-1]
            if cname in {"FunctionTool", "Tool"}:
                inner = get_kwarg(node, "func")
                if inner is None and node.args:
                    inner = node.args[0]
                if isinstance(inner, ast.Name):
                    fn_name = inner.id
                    return ToolSpec(
                        name=fn_name,
                        description=resolver.function_docstring(fn_name),
                        requires_confirmation=_infer_confirm(fn_name),
                    )
            return None
        return None


def _infer_confirm(tool_name: str) -> bool:
    low = tool_name.lower()
    return any(h in low for h in _CONFIRM_HINTS)
