<h1 align="center">
  <img src="https://raw.githubusercontent.com/ctxwitch/ctxwitch/main/docs/hat_icon.png" width="96" alt="ctxwitch" /><br/>
  ctxwitch
</h1>

<p align="center"><b>Scan AI agent changes for behavioral risk</b></p>

<p align="center">
<a href="https://doi.org/10.5281/zenodo.20741295"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20741295-blue" alt="DOI" /></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License" /></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python" /></a>
<a href="tests/"><img src="https://img.shields.io/badge/tests-184%20passing-brightgreen" alt="Tests" /></a>
</p>

ctxwitch analyzes changes to your agent's prompts, models, tools, RAG, memory, and guardrails — before they ship — and classifies the behavioral risk of each change. Git tells you *what changed*; ctxwitch tells you what the change will *do*.

Remove a guardrail or reverse a rule and it flags the change **Breaking** and tells your CI a deeper eval or human review is warranted — while cosmetic edits pass clean.

Runs locally or as a GitHub Action, in ~100ms, deterministically — no agent execution, no traces, no LLM required. *(An optional LLM judge handles the subjective cases.)*

![witch tour demo — a prompt edit scored as a behavioral change across 12 dimensions](https://raw.githubusercontent.com/ctxwitch/ctxwitch/main/docs/demo.gif)

## Try it in 3 minutes

```bash
pip install ctxwitch
witch tour
```

The tour drops you into a disposable sandbox agent and walks you through the
whole loop — behavioral diff, commit, branch, a Breaking change, a Context PR,
and the eval gate that blocks it. Everything runs locally; no API key needed.

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

## The manual workflow

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

## No rewrite required: scan your existing agent code

Already have an agent built with Google ADK, LangGraph, or the raw
Anthropic/OpenAI SDK? You don't have to move anything into witch.yaml to get a
behavioral diff. `witch scan` reads the **behavioral surface** — system prompt,
model, temperature, tools, guardrails — straight out of your Python and runs
CBIA on it.

```bash
# Show what ctxwitch extracts from your agent (no code changes)
witch scan agent.py

# Score the behavioral impact of your uncommitted changes vs a git revision
witch scan agent.py --diff HEAD~1     # exit code 2 if the change is Breaking
```

The scan is **static** — it reads your code, it never imports or runs it, so
it's safe in CI on untrusted PRs. It follows prompt values assigned to module
constants, pulls each tool's description from its function docstring, and — this
is the important part — is **honest about what it can't resolve**. If a prompt is
built at runtime (an f-string, or loaded from witch.yaml), scan marks that field
`unresolved` rather than guessing, so a green result never hides a change it
simply couldn't see.

That gives you two clean ways in, for two stages of adoption:

| Where your prompt/config values live | Diff with | Get |
|---|---|---|
| In your **agent code** (inline or constants) | `witch scan --diff` | Zero-rewrite CBIA on your existing repo |
| In **witch.yaml** (code references it) | `witch diff` | Full governance: environments, PRs, rollback |

Start by scanning your code today; graduate to witch.yaml when you want the full
review workflow below.

## Behavioral scans on every PR (GitHub Action)

Add ctxwitch to CI and every pull request gets a behavioral-impact comment — the
severity of each change, the dimension it affects, and whether it should block.
Copy [docs/examples/behavioral-scan.yml](docs/examples/behavioral-scan.yml) into
`.github/workflows/` in your repo:

```yaml
name: Agent behavioral scan
on: pull_request
permissions:
  contents: read
  pull-requests: write
jobs:
  ctxwitch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0        # required: gives the action the PR base to diff
      - uses: ctxwitch/ctxwitch@v0
        with:
          fail-on: breaking     # breaking | significant | minor | never
```

The PR comment looks like:

> **5 behavioral changes detected** across 2 scanned files.
>
> | Severity | Change | Dimension | Recommended |
> |---|---|---|---|
> | 🔴 Breaking | Constraints removed: Never give investment advice. | Constraints | requires human / security review |
> | 🟠 Significant | Escalation rule reversed: must escalate → may approve | Constraints | run evaluation suite / agent replay |
> | 🟢 Cosmetic | tone wording changed | Tone | no additional testing |
>
> **Policy result:** ❌ merge blocked (Breaking change)

**It runs entirely in your CI.** The scan uses your own git history and your own
`GITHUB_TOKEN` to post the comment — ctxwitch sends **no telemetry**, needs no
account, and nothing (no prompts, no code) ever leaves your infrastructure.

Prefer to wire it into your own pipeline? `witch ci` is the underlying command:

```bash
witch ci --base origin/main --fail-on breaking      # exits 2 if blocked
witch ci --base origin/main --format json           # machine-readable
```

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
| `witch scan <file> [--diff REF] [--framework adk\|generic]` | Extract the behavioral surface from existing agent code and run CBIA — no witch.yaml required |
| `witch ci --base REF [--fail-on breaking\|significant\|minor\|never]` | Scan a PR's changed files (code + witch.yaml), emit a report, exit 2 when blocked — powers the GitHub Action |
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
- [x] Code extraction (`witch scan`): CBIA on existing ADK / raw-SDK agents, no witch.yaml required
- [x] GitHub Action (`witch ci`): behavioral-impact comment on every PR, runs in your CI, zero telemetry
- [x] Public benchmark (`ccia-bench`: 58 labeled pairs)

## What's Next

- More framework adapters for `witch scan` (LangGraph, CrewAI)
- Remote PR integration (GitHub, GitLab)
- CI/CD templates
- Multi-agent context versioning
- Plain-English CBIA guide (docs/)
- More to come — [follow the project](https://github.com/ctxwitch/ctxwitch) for updates

## Community

- **Questions & ideas** → [GitHub Discussions](https://github.com/ctxwitch/ctxwitch/discussions)
- **Bugs & misclassifications** → [Issues](https://github.com/ctxwitch/ctxwitch/issues) — CBIA misses are gold; they become benchmark pairs
- **Contributing** → [CONTRIBUTING.md](CONTRIBUTING.md) — new `ccia-bench` pairs are the most valuable first PR

If ctxwitch is useful to you, a ⭐ helps other agent builders find it.

## Research

This tool implements the framework described in:

> Kulkarni, A. A. (2026). *Context Change Impact Analysis: A Framework for Governing AI Agent Behavior Through Structured Context Versioning.* Zenodo. [https://doi.org/10.5281/zenodo.20741295](https://doi.org/10.5281/zenodo.20741295)

Also available on SSRN: [https://doi.org/10.2139/ssrn.7011398](https://doi.org/10.2139/ssrn.7011398)

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.
