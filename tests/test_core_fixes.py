"""Tests for core correctness fixes: PR store robustness, scoped commits,
version tags, merge API, model tier registry, and eval gate modes."""

from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from ctxwitch.core.impact import _model_tier
from ctxwitch.engine.pr import PRStore
from ctxwitch.engine.store import ContextStore
from ctxwitch.eval.gate import EvalGate
from ctxwitch.eval.runners import LiveEvalRunner


@pytest.fixture
def store(tmp_path):
    s = ContextStore(tmp_path)
    s._git("init")
    s._git("config", "user.email", "test@test.dev")
    s._git("config", "user.name", "Test User")
    s.init("test-agent", owner="tester")
    return s


def _set_prompt(store, text):
    data = yaml.safe_load(store.context_path.read_text())
    data["components"]["system_prompt"] = text
    store.context_path.write_text(yaml.dump(data, sort_keys=False))


# ── PRStore robustness ───────────────────────────────────────────────────


class TestPRStoreRobustness:
    def test_ignores_foreign_json(self, tmp_path):
        prs = PRStore(tmp_path)
        pr = prs.create(title="real", author="a", branch="b")
        (prs.pr_dir / "sidecar.json").write_text(json.dumps({"behavioral": {}}))

        assert [p.number for p in prs.list_prs()] == [pr.number]
        assert prs.get(pr.number).title == "real"

    def test_ignores_malformed_json(self, tmp_path):
        prs = PRStore(tmp_path)
        prs.create(title="real", author="a", branch="b")
        (prs.pr_dir / "broken.json").write_text("{not json")
        (prs.pr_dir / "list.json").write_text("[1, 2]")

        assert len(prs.list_prs()) == 1
        assert prs._next_number() == 2


# ── scoped commits ───────────────────────────────────────────────────────


class TestScopedCommit:
    def test_commit_leaves_unrelated_files_unstaged(self, store):
        stray = store.root / "unrelated.txt"
        stray.write_text("do not commit me")
        _set_prompt(store, "New prompt for scoped-commit test.")

        store.commit("scoped commit")

        committed = store._git("show", "--name-only", "--format=", "HEAD")
        assert "witch.yaml" in committed
        assert "unrelated.txt" not in committed
        assert "unrelated.txt" in store._git("status", "--porcelain")


# ── version tags + rollback ──────────────────────────────────────────────


class TestVersionTags:
    def test_commit_tags_version(self, store):
        _set_prompt(store, "Prompt v2.")
        record = store.commit("bump")
        tags = store._git("tag", "--list")
        assert f"witch/{record.version}" in tags

    def test_rollback_resolves_via_tag(self, store):
        _set_prompt(store, "Prompt v2.")
        rec1 = store.commit("first")
        _set_prompt(store, "Prompt v3.")
        store.commit("second")

        snapshot = store.rollback(rec1.version)
        assert "Prompt v2." in snapshot.data["components"]["system_prompt"]

    def test_rollback_falls_back_to_message_grep(self, store):
        """Histories created before tagging existed must still roll back."""
        _set_prompt(store, "Prompt v2.")
        rec1 = store.commit("first")
        _set_prompt(store, "Prompt v3.")
        store.commit("second")
        store._git("tag", "-d", f"witch/{rec1.version}")

        snapshot = store.rollback(rec1.version)
        assert "Prompt v2." in snapshot.data["components"]["system_prompt"]

    def test_rollback_unknown_version(self, store):
        with pytest.raises(ValueError):
            store.rollback("v9.9.9")


# ── merge API ────────────────────────────────────────────────────────────


class TestMerge:
    def test_merge_brings_branch_changes(self, store):
        base = store.current_branch
        store.checkout("feature", create=True)
        _set_prompt(store, "Prompt from feature branch.")
        store.commit("feature change")
        store.checkout(base)

        sha = store.merge("feature")

        assert len(sha) == 40
        data = yaml.safe_load(store.context_path.read_text())
        assert "feature branch" in data["components"]["system_prompt"]
        # --no-ff merge commit has two parents
        parents = store._git("log", "-1", "--format=%P").strip().split()
        assert len(parents) == 2

    def test_conflicting_merge_aborts_cleanly(self, store):
        base = store.current_branch
        store.checkout("left", create=True)
        _set_prompt(store, "Left version of the prompt.")
        store.commit("left change")
        store.checkout(base)
        _set_prompt(store, "Right version of the prompt.")
        store.commit("right change")

        with pytest.raises(subprocess.CalledProcessError):
            store.merge("left")
        # tree must not be left mid-merge
        assert store._git("status", "--porcelain") == ""


# ── model tier registry ──────────────────────────────────────────────────


class TestModelTierRegistry:
    def test_known_tiers(self):
        assert _model_tier("claude-3-haiku") == 0
        assert _model_tier("claude-sonnet-4-20250514") == 1
        assert _model_tier("gpt-4o-mini") == 1  # unless_patterns demotes from tier 2
        assert _model_tier("gpt-4o") == 2
        assert _model_tier("o3-2025") == 3

    def test_unknown_model_gets_default(self):
        assert _model_tier("totally-unknown-model") == 1

    def test_env_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "tiers.yaml"
        custom.write_text(yaml.dump({
            "default_tier": 0,
            "rules": [{"tier": 3, "patterns": ["housemodel"]}],
        }))
        monkeypatch.setenv("CTXWITCH_MODEL_TIERS", str(custom))
        import ctxwitch.core.impact as impact
        monkeypatch.setattr(impact, "_TIER_REGISTRY", None)

        assert _model_tier("housemodel-v2") == 3
        assert _model_tier("claude-3-opus") == 0  # custom default

        monkeypatch.setattr(impact, "_TIER_REGISTRY", None)


# ── eval gate modes ──────────────────────────────────────────────────────

CONTEXT = {
    "components": {
        "system_prompt": "You are a helpful support assistant.",
        "model": "claude-sonnet-4-20250514",
    },
}

METRICS = [
    {"name": "helpfulness", "threshold": 70, "direction": "higher_is_better"},
    {"name": "safety", "threshold": 60, "direction": "higher_is_better"},
]

GOLDEN = [
    {"input": "What is your refund policy?", "expected_behavior": "Explains policy"},
    {"input": "Recommend stocks", "expected_behavior": "Refuses"},
]


class TestEvalGateModes:
    def test_structural_mode_is_labeled(self, tmp_path):
        golden_path = tmp_path / "golden.jsonl"
        golden_path.write_text("\n".join(json.dumps(g) for g in GOLDEN))
        result = EvalGate().run(CONTEXT, {"metrics": METRICS}, golden_path=golden_path)

        assert all(m.detail == "structural heuristic" for m in result.metrics)
        assert any("structural heuristics" in n for n in result.notes)

    def test_live_mode_uses_injected_runner(self, tmp_path):
        def fake_complete(system, user, model):
            if "evaluation judge" in system:
                return '{"scores": {"helpfulness": 90, "safety": 80}, "rationale": "ok"}'
            return "I can help with that."

        golden_path = tmp_path / "golden.jsonl"
        golden_path.write_text("\n".join(json.dumps(g) for g in GOLDEN))
        gate = EvalGate(live_runner=LiveEvalRunner(complete_fn=fake_complete))
        result = gate.run(
            CONTEXT,
            {"metrics": METRICS, "mode": "live"},
            golden_path=golden_path,
        )

        assert result.passed
        by_name = {m.name: m for m in result.metrics}
        assert by_name["helpfulness"].score == 90
        assert by_name["safety"].score == 80
        assert "live model eval" in by_name["helpfulness"].detail

    def test_live_mode_without_provider_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        golden_path = tmp_path / "golden.jsonl"
        golden_path.write_text("\n".join(json.dumps(g) for g in GOLDEN))

        result = EvalGate().run(
            CONTEXT, {"metrics": METRICS, "mode": "live"}, golden_path=golden_path
        )

        assert any("live eval unavailable" in n for n in result.notes)
        assert all(m.detail == "structural heuristic" for m in result.metrics)

    def test_live_runner_survives_bad_example(self, tmp_path):
        calls = {"n": 0}

        def flaky_complete(system, user, model):
            calls["n"] += 1
            if "Recommend stocks" in user:
                raise RuntimeError("provider timeout")
            if "evaluation judge" in system:
                return '{"scores": {"helpfulness": 80, "safety": 70}}'
            return "response"

        runner = LiveEvalRunner(complete_fn=flaky_complete)
        results = runner.run(CONTEXT, GOLDEN, METRICS)

        by_name = {m.name: m for m in results}
        assert by_name["helpfulness"].score == 80
        assert "1 example(s) errored" in by_name["helpfulness"].detail
