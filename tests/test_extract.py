"""Tests for the code extractor (ctxwitch.extract)."""

from __future__ import annotations

from ctxwitch.core.dimensions import Severity
from ctxwitch.extract import diff_code, extract_snapshot, extract_snapshots

# ── An ADK agent that mirrors the seed.py pattern: prompt in a module
#    constant, tools as bare function refs with docstrings. ────────────────

ADK_V1 = '''
from google.adk.agents import LlmAgent

SUPPORT_PROMPT = """You are Aria, the NovaBank support assistant.
Always verify the customer's identity before discussing account details.
You must escalate any refund above $100 to a human agent.
Never provide investment advice."""


def search_kb(query: str):
    """Search the support knowledge base."""
    ...


def escalate(case_id: str):
    """Escalate the case to a human agent."""
    ...


root_agent = LlmAgent(
    name="support",
    model="gemini-2.0-flash",
    instruction=SUPPORT_PROMPT,
    temperature=0.3,
    tools=[search_kb, escalate],
)
'''

ADK_V2 = '''
from google.adk.agents import LlmAgent

SUPPORT_PROMPT = """You are Aria, the NovaBank support assistant.
Always verify the customer's identity before discussing account details.
You may approve refunds up to $500 without escalation.
Never provide investment advice."""


def search_kb(query: str):
    """Search the support knowledge base."""
    ...


def escalate(case_id: str):
    """Escalate the case to a human agent."""
    ...


root_agent = LlmAgent(
    name="support",
    model="gemini-2.0-flash",
    instruction=SUPPORT_PROMPT,
    temperature=0.8,
    tools=[search_kb, escalate],
)
'''


def test_adk_extracts_prompt_from_module_constant():
    snap = extract_snapshot(ADK_V1, framework="adk")
    assert snap is not None
    assert "escalate any refund above $100" in snap.system_prompt
    assert snap.model == "gemini-2.0-flash"
    assert snap.temperature == 0.3
    assert snap.name == "support"
    assert not snap.dynamic_fields  # everything resolved statically


def test_adk_extracts_tools_with_docstrings():
    snap = extract_snapshot(ADK_V1, framework="adk")
    tools = {t.name: t for t in snap.tools}
    assert set(tools) == {"search_kb", "escalate"}
    assert tools["search_kb"].description == "Search the support knowledge base."
    # 'escalate' should be inferred as requiring confirmation
    assert tools["escalate"].requires_confirmation is True


def test_adk_to_context_dict_shape_matches_cbia():
    snap = extract_snapshot(ADK_V1, framework="adk")
    data = snap.to_context_dict()
    comp = data["components"]
    assert comp["system_prompt"]
    assert comp["model"] == "gemini-2.0-flash"
    assert comp["temperature"] == 0.3
    assert comp["tool_definitions"][0]["name"] in {"search_kb", "escalate"}
    assert "requires_confirmation" in comp["tool_definitions"][0]


def test_code_diff_flags_reversed_refund_directive_as_breaking():
    """The whole point: a refund-policy reversal written in ADK code, not
    YAML, still trips CBIA. Representation invariance."""
    report, old, new = diff_code(ADK_V1, ADK_V2, framework="adk")
    assert old is not None and new is not None
    # directive reversal (must escalate -> may approve) + temp jump 0.3->0.8
    assert report.compound_severity >= Severity.SIGNIFICANT


def test_generic_sdk_anthropic_call():
    src = '''
client = object()
SYS = "You are a helpful assistant. Never discuss competitors."
resp = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=SYS,
    temperature=0.2,
    tools=[{"name": "lookup", "description": "Look something up"}],
    messages=[{"role": "user", "content": "hi"}],
)
'''
    snap = extract_snapshot(src, framework="generic")
    assert snap is not None
    assert snap.model == "claude-sonnet-4-20250514"
    assert "Never discuss competitors" in snap.system_prompt
    assert snap.temperature == 0.2
    assert snap.tools[0].name == "lookup"


def test_dynamic_prompt_is_flagged_not_guessed():
    """An f-string prompt with runtime interpolation must be reported as
    dynamic, never silently blanked."""
    src = '''
from google.adk.agents import LlmAgent
user = "x"
root_agent = LlmAgent(
    name="a",
    model="gemini-2.0-flash",
    instruction=f"You are helping {user}. Be nice.",
)
'''
    snap = extract_snapshot(src, framework="adk")
    assert snap is not None
    assert "system_prompt" in snap.dynamic_fields


def test_multiple_agents_in_one_file():
    src = '''
from google.adk.agents import LlmAgent
a = LlmAgent(name="one", model="gemini-2.0-flash", instruction="First.")
b = LlmAgent(name="two", model="gemini-2.0-flash", instruction="Second.")
'''
    snaps = extract_snapshots(src, framework="adk")
    assert [s.name for s in snaps] == ["one", "two"]


def test_no_agent_returns_empty():
    assert extract_snapshots("x = 1 + 1", framework="adk") == []


def test_python_prompt_module_is_read_as_system_prompt():
    """A prompts.py with no framework constructor is still a behavioral surface:
    a module-level SYSTEM_PROMPT constant is read as the system prompt."""
    src = (
        'SYSTEM_PROMPT = "You are a careful assistant. '
        'Escalate any refund over $50 to a human reviewer."\n'
    )
    snap = extract_snapshot(src, source_file="agent/prompt.py")
    assert snap is not None
    assert snap.source_framework == "python-prompt-module"
    assert "Escalate any refund" in snap.system_prompt


def test_python_prompt_module_diff_flags_prompt_change():
    """Changing the prompt constant in a prompts.py is diffed like any other
    system-prompt change, not silently dropped."""
    old = 'SYSTEM_PROMPT = "Escalate any refund over $50 to a human reviewer."\n'
    new = 'SYSTEM_PROMPT = "You may approve any refund automatically."\n'
    report, o, n = diff_code(old, new, source_file="agent/prompt.py")
    assert o is not None and n is not None
    assert report.compound_severity >= Severity.SIGNIFICANT


def test_ordinary_python_file_is_not_a_prompt_module():
    """A plain code file has no prompt-named constants -> still a clean miss,
    so the fallback never fabricates an agent out of ordinary code."""
    src = "def add(a, b):\n    total = a + b\n    return total\n"
    assert extract_snapshots(src) == []


def test_short_prompt_named_constant_is_ignored():
    """A trivially short string under a prompt-ish name is not a system prompt."""
    src = 'PROMPT = "> "\n'
    assert extract_snapshots(src) == []
