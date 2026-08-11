"""A tiny Google ADK agent, used by docs/demo.tape to show `witch scan`.

It's a normal agent file — no witch.yaml — so the demo can prove ctxwitch reads
the behavioral surface straight out of existing framework code.
"""

from google.adk.agents import LlmAgent


def search_kb(query: str):
    """Search the support knowledge base."""
    ...


def escalate(case_id: str):
    """Escalate the case to a human agent."""
    ...


support_agent = LlmAgent(
    name="support",
    model="gemini-2.0-flash",
    instruction="You are Aria, the NovaBank support assistant. Escalate any refund above $100 to a human agent.",
    temperature=0.3,
    tools=[search_kb, escalate],
)
