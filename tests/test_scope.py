"""Tests for ctxwitch.extract.scope — which changed files are an agent surface.

Each out-of-population category from the benchmark audit gets positive cases
(must leave the `agent` surface) and negative look-alikes (must stay `agent`),
so a rule can't quietly swallow real product prompts.
"""

from __future__ import annotations

import pytest

from ctxwitch.extract.scope import (
    AGENT,
    CODING_ASSISTANT,
    EXCLUDED,
    classify_path,
    detect_moves,
    is_analyzed,
)


def _reason(path):
    s = classify_path(path)
    return s.reason if s.surface == EXCLUDED else s.surface


# ── 1. coding-assistant instructions: a separate, opt-in surface ────────────


@pytest.mark.parametrize("path", [
    "CLAUDE.md",
    "packages/api/CLAUDE.md",
    "CLAUDE.local.md",
    "AGENTS.md",
    "GEMINI.md",
    ".github/copilot-instructions.md",
    ".github/instructions/python.instructions.md",
    ".github/prompts/release.prompt.md",
    ".claude/skills/codebase-design/agents/openai.yaml",
    ".claude/agents/reviewer.md",
    ".cursor/rules/style.mdc",
    ".cursorrules",
])
def test_coding_assistant_files_are_their_own_surface(path):
    s = classify_path(path)
    assert s.surface == CODING_ASSISTANT
    assert s.label == "coding-assistant instructions"
    assert not is_analyzed(path)                                  # off by default
    assert is_analyzed(path, surfaces={AGENT, CODING_ASSISTANT})  # opt-in


@pytest.mark.parametrize("path", [
    "prompts/claude.md",            # a product prompt tuned for a model
    "agents/agents.md",
    "src/agent/instructions.md",
    "skills/summarize/SKILL.md",    # a shipped skill, not .claude/
    ".github/ISSUE_TEMPLATE.md",
])
def test_coding_assistant_lookalikes_are_not_coding_assistant(path):
    assert classify_path(path).surface != CODING_ASSISTANT


# ── 2. evaluation / benchmark prompts ────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "evaluation/prompt.py",
    "evaluation/browsecomp_plus/search_agent/prompts.py",
    "evals/judge.md",
    "benchmarks/run.py",
    "ccia-bench/prompts/system.md",
    "src/prompts/tora/gsm8k.md",
    "src/prompts/cot/math.md",
    "src/prompts/pal/math.md",
    "prompts/mmlu.md",
    "agent/llm_judge.py",
    "prompts/judge_prompt.md",
    "prompts/eval_prompts.yaml",
])
def test_eval_and_benchmark_prompts_are_excluded(path):
    assert _reason(path) == "evaluation"


@pytest.mark.parametrize("path", [
    "prompts/math.md",                       # a product math-tutor prompt
    "tutor/math.py",
    "codex-rs/prompts/templates/review/rubric.md",
    "nanoresearch/prompts/review/method.yaml",
    "guardrails/judge.py",                   # runtime judge guardrail
    "src/evaluator_agent.py",
])
def test_eval_lookalikes_stay_agent(path):
    assert classify_path(path).surface == AGENT


# ── 3. offline data pipelines ────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "apps/collect-trace/utils/converters/system_prompts.py",
    "data_gen/prompts.py",
    "synthetic_data/seed_prompt.md",
    "sft/format.py",
])
def test_data_pipelines_are_excluded(path):
    assert _reason(path) == "data-pipeline"


@pytest.mark.parametrize("path", [
    "apps/miroflow-agent/src/utils/prompt_utils.py",
    "agent/memory/summarizer.py",
    "app/traces.py",
    "src/data_access/prompts.py",
])
def test_data_pipeline_lookalikes_stay_agent(path):
    assert classify_path(path).surface == AGENT


# ── 4. vendored third-party code ─────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "external_tools/run_experiment_tool/ai_scientist/llm.py",
    "vendor/sakana/prompts.py",
    "third_party/agent/system.md",
    "libs/third-party/x/agent.py",
    "node_modules/pkg/prompt.md",
    "vendor/CLAUDE.md",     # vendored wins over coding-assistant
])
def test_vendored_code_is_excluded(path):
    assert _reason(path) == "vendored"


@pytest.mark.parametrize("path", [
    "src/vendors_agent.py",
    "integrations/external_api.py",
    "freephdlabor/llm.py",
])
def test_vendored_lookalikes_stay_agent(path):
    assert classify_path(path).surface == AGENT


# ── 5. generated data + packaging manifests ──────────────────────────────────


@pytest.mark.parametrize("path, reason", [
    ("Crewai-agents/code_analyzer/agent1qvxxckn9y0_data.json", "generated"),
    ("registry/agents-data.yaml", "generated"),
    ("package-lock.json", "generated"),
    ("pnpm-lock.yaml", "generated"),
    ("proto/agent_pb2.py", "generated"),
    ("generated/prompts.json", "generated"),
    ("plugins/headroom-agent-hooks/.github/plugin/plugin.json", "packaging"),
    ("plugins/headroom-agent-hooks/.claude-plugin/plugin.json", "packaging"),
    ("package.json", "packaging"),
    ("manifest.json", "packaging"),
    ("tsconfig.build.json", "packaging"),
    (".github/workflows/ci.yml", "packaging"),
    ("docker-compose.yml", "packaging"),
    ("setup.py", "packaging"),
    ("requirements.txt", "packaging"),
    ("requirements-dev.txt", "packaging"),
    ("dev-requirements.txt", "packaging"),
    ("requirements/base.txt", "packaging"),
    ("constraints.txt", "packaging"),
    ("CMakeLists.txt", "packaging"),
    ("public/robots.txt", "packaging"),
])
def test_generated_and_packaging_are_excluded(path, reason):
    assert _reason(path) == reason


@pytest.mark.parametrize("path", [
    "config/agent.json",
    "agent.yaml",
    "prompts/metadata.yaml",
    "config/adapters/mcp-agent.yaml",
    "use_cases/rag/build/rag.py",       # `build/` is not assumed generated
    "src/agent_config.json",
])
def test_generated_lookalikes_stay_agent(path):
    assert classify_path(path).surface == AGENT


# ── docs ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", [
    "README.md", "docs/guide.md", "docs/blog/posts/hot-reload.md",
    "CHANGELOG.md", "SECURITY.md", "pkg/readme.txt",
    "ROADMAP.md", "ARCHITECTURE.md", "TODO.md", ".changeset/brave-owls.md",
    "llms.txt",
])
def test_docs_are_excluded(path):
    assert _reason(path) == "docs"


@pytest.mark.parametrize("path", [
    "prompts/security.md", "memory/history.md",
    "prompts/roadmap.md", "prompts/requirements.md", "prompts/requirements_analyst.txt",
])
def test_doc_lookalikes_stay_agent(path):
    assert classify_path(path).surface == AGENT


# ── real product-agent paths from the audit's kept set ───────────────────────


@pytest.mark.parametrize("path", [
    "crates/goose/src/prompts/system.md",
    "prompts/persona_builder.md",
    "backend/prompts.py",
    "src/prompts/planner.py",
    "config/demo/no_instructions.yaml",
    "demo/chat_v2/backend_app/services/chat.py",
    "apps/gradio-demo/prompt_patch.py",
    "app/config/realtime_instructions.txt",
    "shortGPT/prompt_templates/editing_generate_images.yaml",
    "references/prompt/background_template.md",
    "agent.py",
    "witch.yaml",
])
def test_product_agent_paths_are_analyzed(path):
    assert classify_path(path).surface == AGENT
    assert is_analyzed(path)


def test_path_normalization():
    assert classify_path("./evaluation/prompt.py").path == "evaluation/prompt.py"
    assert classify_path("vendor\\x\\agent.py").reason == "vendored"


def test_include_glob_forces_a_path_in():
    assert not is_analyzed("evaluation/agent.py")
    assert is_analyzed("evaluation/agent.py", include=["evaluation/*"])
    assert not is_analyzed("vendor/agent.py", include=["evaluation/*"])


# ── 6. moves ─────────────────────────────────────────────────────────────────


def test_detect_moves_pairs_identical_delete_and_add():
    moves = detect_moves({
        "crates/goose/src/prompts/compaction.md": ("Summarize.", ""),
        "crates/goose-cm/src/prompts/compaction.md": ("", "Summarize."),
    })
    assert moves == [("crates/goose/src/prompts/compaction.md",
                      "crates/goose-cm/src/prompts/compaction.md")]


def test_detect_moves_ignores_move_with_edit():
    assert detect_moves({
        "a/prompt.md": ("Be terse.", ""),
        "b/prompt.md": ("", "Be verbose."),
    }) == []


def test_detect_moves_ignores_modifications_and_empty_files():
    assert detect_moves({
        "a.md": ("x", "x!"),       # modified in place
        "old.md": ("", ""),        # empty on both sides
        "gone.md": ("", ""),
    }) == []
    # an empty file deleted and another added is not a "move"
    assert detect_moves({"a.py": ("", ""), "b.py": ("", "")}) == []


def test_detect_moves_prefers_same_basename_and_pairs_one_to_one():
    moves = detect_moves({
        "old/a.md": ("same", ""),
        "old/b.md": ("same", ""),
        "new/b.md": ("", "same"),
    })
    assert moves == [("old/b.md", "new/b.md")]
