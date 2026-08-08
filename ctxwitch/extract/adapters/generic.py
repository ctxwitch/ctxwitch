"""Generic raw-SDK adapter — Anthropic / OpenAI direct calls.

Catches agents written without a framework, against the model SDK directly:

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=SUPPORT_PROMPT,
        temperature=0.3,
        tools=[{"name": "search_kb", "description": "..."}],
        messages=[...],
    )

Maps Anthropic `system` / OpenAI first system message -> system_prompt,
`model` -> model, `temperature` -> temperature, and inline tool dicts ->
tool_definitions. This is the lowest-common-denominator adapter and runs
whenever no framework-specific adapter claims a call.
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

# Trailing call names that denote a model completion call.
_COMPLETION_CALLS = {
    "create",   # client.messages.create / client.chat.completions.create
    "complete",
    "generate",
}
_COMPLETION_TAILS = ("messages.create", "completions.create")


@register_adapter
class GenericSDKAdapter(FrameworkAdapter):
    name = "generic"

    def matches(self, call: ast.Call, resolver: SymbolResolver) -> bool:
        cname = call_name(call)
        if not cname:
            return False
        if any(cname.endswith(tail) for tail in _COMPLETION_TAILS):
            return True
        # A `.create(...)` with model= present is very likely a completion call.
        if cname.split(".")[-1] in _COMPLETION_CALLS:
            return get_kwarg(call, "model") is not None
        return False

    def extract(
        self,
        call: ast.Call,
        assigned_to: Optional[str],
        resolver: SymbolResolver,
    ) -> Optional[BehavioralSnapshot]:
        dynamic: List[str] = []
        snap = BehavioralSnapshot(
            name=assigned_to or "agent",
            source_framework=self.name,
            source_file=resolver.source_file,
            source_line=first_line(call),
            agent_symbol=assigned_to or "",
        )

        # system prompt: Anthropic `system=`, else first system message
        sys_node = get_kwarg(call, "system")
        prompt, ok = resolver.resolve_str(sys_node)
        if ok:
            snap.system_prompt = prompt
        elif sys_node is not None:
            dynamic.append("system_prompt")
        else:
            msg_prompt, msg_ok, msg_dyn = self._system_from_messages(call, resolver)
            if msg_ok:
                snap.system_prompt = msg_prompt
            elif msg_dyn:
                dynamic.append("system_prompt")

        model, ok = resolver.resolve_str(get_kwarg(call, "model"))
        if ok:
            snap.model = model
        elif get_kwarg(call, "model") is not None:
            dynamic.append("model")

        temp_val = resolver.resolve(get_kwarg(call, "temperature"))
        if isinstance(temp_val, (int, float)):
            snap.temperature = float(temp_val)
        elif get_kwarg(call, "temperature") is not None:
            dynamic.append("temperature")

        max_val = resolver.resolve(get_kwarg(call, "max_tokens"))
        if isinstance(max_val, int):
            snap.max_tokens = max_val

        self._extract_tools(call, resolver, snap, dynamic)

        snap.dynamic_fields = dynamic
        return snap

    def _system_from_messages(self, call, resolver):
        """Pull a system prompt from messages=[{"role":"system","content":...}]."""
        messages = get_kwarg(call, "messages")
        if not isinstance(messages, (ast.List, ast.Tuple)):
            return None, False, messages is not None
        for elt in messages.elts:
            if not isinstance(elt, ast.Dict):
                continue
            d = resolver.resolve(elt)
            if isinstance(d, dict) and d.get("role") == "system":
                content = d.get("content")
                if isinstance(content, str):
                    return content, True, False
                return None, False, True
        return None, False, False

    def _extract_tools(self, call, resolver, snap, dynamic) -> None:
        tools_node = get_kwarg(call, "tools")
        if tools_node is None:
            return
        if not isinstance(tools_node, (ast.List, ast.Tuple)):
            dynamic.append("tool_definitions")
            return
        for elt in tools_node.elts:
            spec = self._tool_from_dict(elt, resolver)
            if spec is None:
                dynamic.append("tool_definitions")
            else:
                snap.tools.append(spec)

    def _tool_from_dict(self, node: ast.AST, resolver: SymbolResolver) -> Optional[ToolSpec]:
        d = resolver.resolve(node)
        if not isinstance(d, dict):
            return None
        # Anthropic: {"name","description"}; OpenAI: {"type":"function",
        # "function":{"name","description"}}
        fn = d.get("function") if isinstance(d.get("function"), dict) else d
        name = fn.get("name")
        if not isinstance(name, str):
            return None
        desc = fn.get("description")
        return ToolSpec(name=name, description=desc if isinstance(desc, str) else "")
