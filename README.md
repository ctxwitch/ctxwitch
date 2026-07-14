# ctxwitch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20741295.svg)](https://doi.org/10.5281/zenodo.20741295)

**Version control for AI agent behavior.** Git tells you what changed in your prompt — ctxwitch tells you what the change will *do*: semantic diffs across 12 behavioral dimensions, eval gates, and Context PRs for prompts, RAG configs, tool definitions, and guardrails.

![witch tour demo — a prompt edit scored as a behavioral change across 12 dimensions](docs/demo.gif)

ctxwitch is the reference implementation of [Context Change Impact Analysis (CCIA)](https://doi.org/10.5281/zenodo.20741295) — a discipline for predicting how changes to an AI agent's context configuration affect its observable behavior. The core engine, CBIA (Compound Behavioral Impact Analysis), is a 6-tier pipeline that scores any context change across 12 behavioral dimensions at 5 severity levels, deterministically, in under 100ms, without LLM inference.

## The Problem

AI app behavior is controlled by context -- prompts, RAG configs, tool definitions -- that changes frequently and needs input from engineers, PMs, domain experts, and compliance teams. Today this falls into one of two broken patterns:

1. **Locked in code** (the ADK/LangChain pattern): only engineers can touch it. PMs file Jira tickets and wait 3-5 days for a prompt change.
2. **Scattered in tools** with no team workflow, no eval-gating, and no deployment governance.

Neither pattern supports safe, collaborative, multi-stakeholder contribution to a production AI system.

## The Solution

ctxwitch treats AI context like code -- but better. Every change goes through a **Context PR** with semantic diffs, automated eval gates, review workflows, and one-command rollback.

```
WITHOUT ctxwitch              WITH ctxwitch
-------------------------------  --------------------------------
PM changes prompt in Jira     ->  PM opens Context PR
Goes through eng sprint       ->  Eval gate runs automatically
3-5 day delay                 ->  Problem caught before prod
No semantic review            ->  Reviewer sees exact behavior diff
No rollback                   ->  Tagged versions, instant rollback
Compliance audit fails        ->  Complete audit trail in 30 sec
```

## Install

```bash
pip install ctxwitch

# or, for development
pip install -e ".[dev]"
```

## Quick Start

New here? The fastest way in is the guided tour — it creates a disposable
sandbox agent and walks you through the whole loop (behavioral diff → commit →
branch → breaking change → Context PR → eval gate) in about 3 minutes:

```bash
witch tour
```

Or do it manually:

```bash
# Initialize a project
witch init my-support-agent

# Edit witch.yaml with your AI config, then commit
witch commit -m "configure support agent prompt"

# Create a branch for changes
witch checkout -b refund-policy-update

# Edit witch.yaml...
witch commit -m "tighten refund approval per CEO feedback"

# Create a Context PR
witch pr create -t "Tighten refund approval policy"

# Run eval gate
witch eval

# View the semantic diff with behavioral impact analysis
witch diff --ref main

# Enable LLM-as-judge for deeper subjective analysis
witch diff --ref main --judge

# View history
witch log
```

> **Alias:** You can also use `ctxw` instead of `witch` for all commands.

## Use the governed context in your app

Your agent loads its context from witch.yaml instead of hardcoding it — so
behavior changes ship through Context PRs, not redeploys:

```python
from ctxwitch.runtime import load_components

components = load_components(env="prod")  # or set CTXWITCH_ENV

response = client.messages.create(
    model=components["model"],
    system=components["system_prompt"],
    temperature=components["temperature"],
    max_tokens=components["max_tokens"],
    messages=[...],
)
```

Environment overrides from the `environments:` block are deep-merged, so dev
and prod diverge only where they say they do. Non-Python stacks: `witch spell
export --format json` in your build step.

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `witch tour` | Guided hands-on walkthrough in a disposable sandbox (start here) |
| `witch init <name>` | Initialize a new ctxwitch project |
| `witch status` | Show current context state |
| `witch commit -m "msg"` | Commit context changes with version bump + rollback tag |
| `witch checkout [-b] <branch>` | Switch to or create a context branch |
| `witch diff [--ref REF] [--judge]` | Behavioral diff vs last commit (or any ref), like `git diff` |
| `witch log [-n COUNT]` | Show context change history |
| `witch eval [--judge] [--allow-breaking]` | Run the gate: metric thresholds + CBIA; Breaking changes block (exit 2) unless overridden |
| `witch rollback <version>` | Rollback to a specific version |
| `witch branches` | List all context branches |

### Context PRs

| Command | Description |
|---------|-------------|
| `witch pr create -t "title"` | Create a context PR from current branch |
| `witch pr list` | List all context PRs |
| `witch pr show <number>` | Show PR details with diff and comments |
| `witch pr merge <number>` | Merge a PR (blocked on Breaking changes unless `--allow-breaking`) |

### Inspect

| Command | Description |
|---------|-------------|
| `witch inspect prompt` | Show the full system prompt |
| `witch inspect tools` | List all tool definitions |
| `witch inspect rag` | Show RAG configuration |
| `witch inspect env [ENV]` | Show environment-specific overrides |

### Spell (Transform)

| Command | Description |
|---------|-------------|
| `witch spell set <key> <value>` | Set a context component value |
| `witch spell add-tool <name>` | Add a tool definition |
| `witch spell validate` | Validate witch.yaml against schema |
| `witch spell export [--format]` | Export context as YAML or JSON |

## witch.yaml Schema

The `witch.yaml` file is the atomic unit of ctxwitch. It captures the full behavioral surface of your AI application. A complete reference is at [`examples/witch.yaml`](examples/witch.yaml).

```yaml
version: "v0.1.0"
name: "my-support-agent"
description: "AI context managed by ctxwitch"
owner: "team-name"

components:
  system_prompt: |
    You are a helpful customer support assistant.
    Always verify identity before discussing account details.

  model: "claude-sonnet-4-20250514"
  temperature: 0.3
  max_tokens: 4096

  rag_config:
    enabled: false
    chunk_size: 512
    top_k: 5
    embedding_model: "text-embedding-3-small"

  tool_definitions:
    - name: "search_kb"
      description: "Search the knowledge base"
    - name: "escalate"
      description: "Escalate to human agent"
      requires_confirmation: true

  memory:
    enabled: false
    backend: "local"
    retention_days: 30
    write_policy: "on_trigger"

  guardrails:
    blocked_topics: ["violence", "illegal_activity"]
    max_turns: 50

environments:
  dev:
    components:
      temperature: 0.7
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
      threshold: 90
      direction: "higher_is_better"
  block_on_failure: true
```

## Architecture

```
ctxwitch/
  core/          # Context schema, model, diff engine, CBIA pipeline
  cli/           # Click-based CLI (witch, tour, inspect, spell commands)
  engine/        # Git-backed store, PR workflow engine
  eval/          # Pluggable eval gate framework + live model runner
  runtime.py     # Load governed context into your agent (env overrides)
  a2a/           # Agent-to-agent handover versioning (future)
ccia-bench/      # Public benchmark: labeled context-change pairs + scorer
examples/        # Sample witch.yaml and golden.jsonl
tests/           # Test suite (170 tests)
```

## What's Built

- [x] Context YAML schema and validation
- [x] Git-backed versioning engine with rollback tags
- [x] CLI: init, commit, checkout, diff, log, status, rollback + guided `witch tour`
- [x] Context PR workflow (create, list, review, merge with Breaking-change gate)
- [x] Eval gate framework: structural heuristics + live model eval (`eval.mode: live`)
- [x] 6-tier CBIA behavioral semantic diff pipeline
- [x] 12-dimension behavioral taxonomy with compound severity
- [x] Directive contradiction, numeric-threshold, and environment-override detection
- [x] Typo/punctuation-robust negation detection (orthographic verdict stability)
- [x] Confidence-gated LLM-as-judge (Tier 6)
- [x] CI-ready exit codes (`witch diff --strict`, `witch eval`)
- [x] Runtime API (`ctxwitch.runtime.load_components`)
- [x] Public benchmark (`ccia-bench`: 55 labeled pairs)

## What's Next

- Remote PR integration (GitHub, GitLab)
- CI/CD templates
- Multi-agent context versioning
- Plain-English CBIA guide (docs/)
- More to come — [follow the project](https://github.com/ctxwitch/ctxwitch) for updates

## Research

This tool implements the framework described in:

> Kulkarni, A. A. (2026). *Context Change Impact Analysis: A Framework for Governing AI Agent Behavior Through Structured Context Versioning.* Zenodo. [https://doi.org/10.5281/zenodo.20741295](https://doi.org/10.5281/zenodo.20741295)

## License

Business Source License 1.1 -- see [LICENSE](LICENSE) for details.
