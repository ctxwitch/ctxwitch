"""Context YAML schema definition and validation.

The witch.yaml is the atomic unit of ctxwitch. It captures the full
behavioral surface of an AI application: prompt, model, parameters,
RAG config, tool definitions, memory policy, and A2A handover bundles.
"""

from __future__ import annotations

CONTEXT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ctxwitch context",
    "description": "Version-controlled AI context definition",
    "type": "object",
    "required": ["version", "name", "components"],
    "properties": {
        "version": {
            "type": "string",
            "pattern": r"^v\d+\.\d+\.\d+$",
            "description": "Semantic version of this context snapshot",
        },
        "name": {
            "type": "string",
            "description": "Human-readable name of the AI application or agent",
        },
        "description": {
            "type": "string",
        },
        "owner": {
            "type": "string",
            "description": "Team or individual responsible for this context",
        },
        "components": {
            "type": "object",
            "required": ["system_prompt", "model"],
            "properties": {
                "system_prompt": {"type": "string"},
                "model": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "max_tokens": {"type": "integer", "minimum": 1},
                "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                "stop_sequences": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rag_config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "source": {"type": "string"},
                        "chunk_size": {"type": "integer"},
                        "chunk_overlap": {"type": "integer"},
                        "top_k": {"type": "integer"},
                        "embedding_model": {"type": "string"},
                        "similarity_threshold": {"type": "number"},
                    },
                },
                "tool_definitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"},
                            "requires_confirmation": {"type": "boolean"},
                        },
                    },
                },
                "memory": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "backend": {"type": "string"},
                        "retention_days": {"type": "integer"},
                        "write_policy": {
                            "type": "string",
                            "enum": ["always", "on_trigger", "manual"],
                        },
                    },
                },
                "guardrails": {
                    "type": "object",
                    "properties": {
                        "input_filters": {"type": "array", "items": {"type": "string"}},
                        "output_filters": {"type": "array", "items": {"type": "string"}},
                        "blocked_topics": {"type": "array", "items": {"type": "string"}},
                        "max_turns": {"type": "integer"},
                    },
                },
            },
        },
        "environments": {
            "type": "object",
            "description": "Environment-specific overrides (dev, staging, prod)",
            "additionalProperties": {
                "type": "object",
            },
        },
        "eval": {
            "type": "object",
            "description": "Eval gate configuration",
            "properties": {
                "golden_dataset": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["structural", "live"],
                    "description": "structural = config heuristics (default); live = run golden set against the model",
                },
                "max_examples": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cap on golden examples used per live eval run (cost control)",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "threshold"],
                        "properties": {
                            "name": {"type": "string"},
                            "threshold": {"type": "number"},
                            "direction": {
                                "type": "string",
                                "enum": ["higher_is_better", "lower_is_better"],
                            },
                        },
                    },
                },
                "block_on_failure": {"type": "boolean"},
            },
        },
        "a2a": {
            "type": "object",
            "description": "Agent-to-Agent handover configuration (future)",
            "properties": {
                "handover_bundles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["from_agent", "to_agent"],
                        "properties": {
                            "from_agent": {"type": "string"},
                            "to_agent": {"type": "string"},
                            "passed_context": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "version": {"type": "string"},
                        },
                    },
                },
            },
        },
        "metadata": {
            "type": "object",
            "additionalProperties": True,
        },
    },
}

SCAFFOLD_CONTEXT = """\
version: "v0.1.0"
name: "{name}"
description: "AI context managed by ctxwitch"
owner: "{owner}"

components:
  system_prompt: |
    You are a helpful assistant.
    Respond clearly and concisely.

  model: "claude-sonnet-4-20250514"
  temperature: 0.3
  max_tokens: 4096

  rag_config:
    enabled: false
    source: ""
    chunk_size: 512
    chunk_overlap: 50
    top_k: 5
    embedding_model: "text-embedding-3-small"

  tool_definitions: []

  memory:
    enabled: false
    backend: "local"
    retention_days: 30
    write_policy: "on_trigger"

  guardrails:
    input_filters: []
    output_filters: []
    blocked_topics: []
    max_turns: 50

environments:
  dev:
    components:
      temperature: 0.7
  staging:
    components:
      temperature: 0.3
  prod:
    components:
      temperature: 0.3

eval:
  golden_dataset: "evals/golden.jsonl"
  metrics:
    - name: "helpfulness"
      threshold: 70
      direction: "higher_is_better"
    - name: "safety"
      threshold: 60
      direction: "higher_is_better"
  block_on_failure: true

# A2A handover bundles (future — uncomment when multi-agent)
# a2a:
#   handover_bundles:
#     - from_agent: "triage-agent"
#       to_agent: "support-agent"
#       passed_context:
#         - customer_issue
#         - sentiment
#         - conversation_history
#       version: "v1"
"""
