"""Static resolution helpers for the code extractor.

Real agent code rarely inlines everything into the constructor call:

    SUPPORT_PROMPT = "You are Aria...\\nEscalate refunds over $100."
    root_agent = LlmAgent(instruction=SUPPORT_PROMPT, model="gemini-2.0-flash")

To produce a strong snapshot the extractor has to follow that reference back
to its string value, concatenate `"a" + "b"` and implicitly-joined string
literals, evaluate simple literals (numbers, bools, lists), and pull a tool's
description from the docstring of the function it points at — all without
executing the module. `SymbolResolver` walks the module AST once and answers
those questions for the adapters.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

# Sentinel: "the extractor could not statically resolve this value."
UNRESOLVED = object()


class SymbolResolver:
    """A one-file static symbol table over a parsed module.

    Collects module-level (and class-level) constant assignments and function
    definitions, then resolves AST value nodes against them. Deliberately
    conservative: anything it cannot prove it returns as UNRESOLVED, and the
    caller records the field name in `dynamic_fields` rather than guessing.
    """

    def __init__(self, module: ast.Module, source_file: str = ""):
        self.module = module
        self.source_file = source_file
        self.constants: Dict[str, ast.AST] = {}
        self.functions: Dict[str, ast.FunctionDef] = {}
        self._index(module)

    def _index(self, module: ast.Module) -> None:
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, node)
            elif isinstance(node, ast.Assign):
                # NAME = <value>   (only simple, single-target name bindings)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.constants.setdefault(target.id, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                if isinstance(node.target, ast.Name):
                    self.constants.setdefault(node.target.id, node.value)

    # ── value resolution ────────────────────────────────────────────────

    def resolve(self, node: Optional[ast.AST], _depth: int = 0) -> Any:
        """Resolve an AST value node to a Python value, or UNRESOLVED.

        Handles: str/num/bool/None constants, implicitly-joined and
        `+`-concatenated strings, lists/tuples of resolvable items, dict
        literals with constant keys, and NAME references to module constants
        (followed recursively, with a depth guard against cycles).
        """
        if node is None or _depth > 12:
            return UNRESOLVED

        # Literal constant: "...", 0.8, True, None
        if isinstance(node, ast.Constant):
            return node.value

        # Implicitly joined adjacent string literals -> ast.JoinedStr only for
        # f-strings; plain adjacency is folded into a single Constant by the
        # parser, so nothing extra needed there.

        # f-string: keep the literal parts, mark the whole thing dynamic by
        # returning UNRESOLVED unless every piece is a plain constant.
        if isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    return UNRESOLVED  # runtime interpolation -> not static
            return "".join(parts)

        # String / numeric concatenation: a + b, a * n
        if isinstance(node, ast.BinOp):
            left = self.resolve(node.left, _depth + 1)
            right = self.resolve(node.right, _depth + 1)
            if left is UNRESOLVED or right is UNRESOLVED:
                return UNRESOLVED
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Mult):
                    return left * right
            except TypeError:
                return UNRESOLVED
            return UNRESOLVED

        # Collections
        if isinstance(node, (ast.List, ast.Tuple)):
            out = []
            for elt in node.elts:
                val = self.resolve(elt, _depth + 1)
                if val is UNRESOLVED:
                    # keep a placeholder so length/among-known items still read
                    out.append(UNRESOLVED)
                else:
                    out.append(val)
            return out

        if isinstance(node, ast.Dict):
            out_d: Dict[Any, Any] = {}
            for k, v in zip(node.keys, node.values):
                key = self.resolve(k, _depth + 1) if k is not None else UNRESOLVED
                val = self.resolve(v, _depth + 1)
                if key is UNRESOLVED:
                    return UNRESOLVED
                out_d[key] = val
            return out_d

        # NAME -> follow to its module-level binding
        if isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.resolve(self.constants[node.id], _depth + 1)
            return UNRESOLVED

        # Attribute like some_enum.VALUE or module.CONST: render dotted name
        # as a string fallback (useful for model constants). Kept conservative.
        if isinstance(node, ast.Attribute):
            dotted = _dotted_name(node)
            if dotted is not None:
                return UNRESOLVED  # a symbol we can't see the value of
            return UNRESOLVED

        return UNRESOLVED

    def resolve_str(self, node: Optional[ast.AST]) -> Tuple[Optional[str], bool]:
        """Resolve to a string. Returns (value, ok). ok=False -> dynamic."""
        val = self.resolve(node)
        if isinstance(val, str):
            return val, True
        return None, False

    # ── tool / callable resolution ──────────────────────────────────────

    def function_docstring(self, name: str) -> str:
        fn = self.functions.get(name)
        if fn is None:
            return ""
        return (ast.get_docstring(fn) or "").strip()

    def has_function(self, name: str) -> bool:
        return name in self.functions


def get_kwarg(call: ast.Call, name: str) -> Optional[ast.AST]:
    """Return the AST value node for keyword `name`, if present."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def call_name(call: ast.Call) -> Optional[str]:
    """The (possibly dotted) name of the thing being called.

    LlmAgent(...) -> "LlmAgent"; adk.Agent(...) -> "adk.Agent";
    client.messages.create(...) -> "client.messages.create".
    """
    return _dotted_name(call.func)


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def first_line(node: ast.AST) -> int:
    return getattr(node, "lineno", 0) or 0
