"""Eval gate — pluggable evaluation framework for context changes.

Eval gates run automatically on context PRs. They compare the behavior
of the new context against golden datasets and quality thresholds.
If the gate fails, the merge is blocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class GateVerdict(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class MetricResult:
    name: str
    score: float
    threshold: float
    direction: str  # higher_is_better or lower_is_better
    passed: bool
    detail: str = ""

    @property
    def icon(self) -> str:
        if self.passed:
            return "pass"
        return "FAIL"


@dataclass
class EvalResult:
    verdict: GateVerdict
    metrics: List[MetricResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    golden_count: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict in (GateVerdict.PASSED, GateVerdict.WARNING)

    @property
    def summary(self) -> str:
        passed = sum(1 for m in self.metrics if m.passed)
        total = len(self.metrics)
        return f"{passed}/{total} metrics passed — {self.verdict.value}"


class EvalGate:
    """Pluggable eval gate that can run multiple evaluators."""

    def __init__(self):
        self._evaluators: List[Callable] = []

    def register(self, evaluator: Callable) -> None:
        """Register an evaluator function.

        Evaluators receive (context_data, golden_dataset) and return MetricResult.
        """
        self._evaluators.append(evaluator)

    def run(
        self,
        context_data: Dict[str, Any],
        eval_config: Optional[Dict[str, Any]] = None,
        golden_path: Optional[Path] = None,
    ) -> EvalResult:
        """Run all registered evaluators against the context."""
        if not eval_config:
            return EvalResult(
                verdict=GateVerdict.SKIPPED,
                notes=["No eval config found — skipping gate"],
            )

        golden = []
        if golden_path and golden_path.exists():
            with open(golden_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        golden.append(json.loads(line))

        metrics_config = eval_config.get("metrics", [])
        results: List[MetricResult] = []

        for metric_def in metrics_config:
            name = metric_def["name"]
            threshold = metric_def["threshold"]
            direction = metric_def.get("direction", "higher_is_better")

            score = self._evaluate_metric(name, context_data, golden)

            if direction == "higher_is_better":
                passed = score >= threshold
            else:
                passed = score <= threshold

            results.append(
                MetricResult(
                    name=name,
                    score=score,
                    threshold=threshold,
                    direction=direction,
                    passed=passed,
                )
            )

        for evaluator in self._evaluators:
            try:
                result = evaluator(context_data, golden)
                if isinstance(result, MetricResult):
                    results.append(result)
            except Exception as e:
                results.append(
                    MetricResult(
                        name=evaluator.__name__,
                        score=0,
                        threshold=0,
                        direction="higher_is_better",
                        passed=False,
                        detail=f"Evaluator error: {e}",
                    )
                )

        block_on_failure = eval_config.get("block_on_failure", True)
        all_passed = all(m.passed for m in results)
        has_warnings = any(not m.passed for m in results)

        if all_passed:
            verdict = GateVerdict.PASSED
        elif not block_on_failure and has_warnings:
            verdict = GateVerdict.WARNING
        else:
            verdict = GateVerdict.FAILED

        return EvalResult(
            verdict=verdict,
            metrics=results,
            golden_count=len(golden),
        )

    def _evaluate_metric(
        self, name: str, context_data: Dict[str, Any], golden: List[Dict]
    ) -> float:
        """Built-in metric evaluation.

        In production this would call the actual model and score responses.
        For now, provides structural validation scores.
        """
        components = context_data.get("components", {})

        if name == "helpfulness":
            prompt = components.get("system_prompt", "")
            score = 50.0
            if len(prompt) > 20:
                score += 15
            if "helpful" in prompt.lower() or "assist" in prompt.lower():
                score += 15
            if components.get("rag_config", {}).get("enabled"):
                score += 10
            if components.get("tool_definitions"):
                score += 10
            return min(score, 100)

        if name == "safety":
            score = 60.0
            guardrails = components.get("guardrails", {})
            if guardrails.get("blocked_topics"):
                score += 15
            if guardrails.get("output_filters"):
                score += 15
            if guardrails.get("max_turns", 0) > 0:
                score += 10
            return min(score, 100)

        if name == "consistency":
            temp = components.get("temperature", 0.5)
            score = 100 - (temp * 40)
            return max(score, 0)

        if name == "policy_compliance":
            prompt = components.get("system_prompt", "")
            score = 50.0
            if "policy" in prompt.lower() or "strictly" in prompt.lower():
                score += 25
            if "escalate" in prompt.lower():
                score += 15
            if golden:
                score += 10
            return min(score, 100)

        return 75.0


def create_default_gate() -> EvalGate:
    """Create an eval gate with default evaluators."""
    return EvalGate()
