"""Live eval runner — score context changes against the actual model.

The structural metrics in gate.py are fast, deterministic heuristics; they
validate configuration shape, not behavior. This runner does the real thing:

  1. For each golden example, generate a response using the context's own
     system prompt and model.
  2. Have an LLM judge score that response against the example's
     expected_behavior, once per configured metric (0-100).
  3. Average per-metric scores across the golden set.

Enable with `eval.mode: live` in witch.yaml. Requires ANTHROPIC_API_KEY or
OPENAI_API_KEY (provider auto-detected, same as Tier 6). Cost is bounded by
`eval.max_examples` (default 10).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from ctxwitch.eval.gate import MetricResult

# complete_fn(system_prompt, user_message, model) -> response text.
# Injectable so tests (and self-hosted gateways) never need a vendor SDK.
CompleteFn = Callable[[str, str, str], str]

DEFAULT_MAX_EXAMPLES = 10

JUDGE_SYSTEM = """\
You are an evaluation judge for AI assistant responses.

You will be given: the user input, the assistant's actual response, the
expected behavior, and a list of metric names.

Score the response 0-100 on each metric, where 100 means the response fully
exhibits the expected behavior with respect to that metric.

Respond ONLY with valid JSON: {"scores": {"<metric>": <0-100>, ...}, "rationale": "<one sentence>"}
"""


class NoProviderError(RuntimeError):
    """Raised when live eval is requested but no LLM provider is available."""


class LiveEvalRunner:
    def __init__(
        self,
        complete_fn: Optional[CompleteFn] = None,
        judge_model: Optional[str] = None,
    ):
        if complete_fn is None:
            complete_fn = _detect_complete_fn()
        self.complete_fn = complete_fn
        self.judge_model = judge_model

    def run(
        self,
        context_data: Dict[str, Any],
        golden: List[Dict[str, Any]],
        metrics_config: List[Dict[str, Any]],
        max_examples: int = DEFAULT_MAX_EXAMPLES,
    ) -> List[MetricResult]:
        components = context_data.get("components", {})
        system_prompt = components.get("system_prompt", "")
        model = components.get("model", "")
        metric_names = [m["name"] for m in metrics_config]

        if not golden:
            return [
                _result(m, 0.0, passed=False, detail="live eval: golden dataset is empty")
                for m in metrics_config
            ]

        totals: Dict[str, float] = {n: 0.0 for n in metric_names}
        scored = 0
        errors: List[str] = []

        for example in golden[:max_examples]:
            user_input = example.get("input", "")
            expected = example.get("expected_behavior", "")
            try:
                response = self.complete_fn(system_prompt, user_input, model)
                scores = self._judge(user_input, response, expected, metric_names)
            except Exception as e:  # keep one bad example from sinking the gate
                errors.append(f"{user_input[:40]!r}: {e}")
                continue
            for name in metric_names:
                totals[name] += float(scores.get(name, 0))
            scored += 1

        results = []
        for metric_def in metrics_config:
            name = metric_def["name"]
            direction = metric_def.get("direction", "higher_is_better")
            threshold = metric_def["threshold"]
            if scored == 0:
                results.append(_result(
                    metric_def, 0.0, passed=False,
                    detail="live eval: all examples failed — " + "; ".join(errors[:2]),
                ))
                continue
            score = round(totals[name] / scored, 1)
            passed = score >= threshold if direction == "higher_is_better" else score <= threshold
            detail = f"live model eval over {scored} golden example(s)"
            if errors:
                detail += f" ({len(errors)} example(s) errored)"
            results.append(_result(metric_def, score, passed=passed, detail=detail))
        return results

    def _judge(
        self,
        user_input: str,
        response: str,
        expected: str,
        metric_names: List[str],
    ) -> Dict[str, float]:
        judge_prompt = (
            f"USER INPUT:\n{user_input}\n\n"
            f"ASSISTANT RESPONSE:\n{response}\n\n"
            f"EXPECTED BEHAVIOR:\n{expected}\n\n"
            f"METRICS TO SCORE: {', '.join(metric_names)}"
        )
        raw = self.complete_fn(JUDGE_SYSTEM, judge_prompt, self.judge_model or "")
        return _parse_scores(raw)


def _result(metric_def: Dict[str, Any], score: float, passed: bool, detail: str) -> MetricResult:
    return MetricResult(
        name=metric_def["name"],
        score=score,
        threshold=metric_def["threshold"],
        direction=metric_def.get("direction", "higher_is_better"),
        passed=passed,
        detail=detail,
    )


def _parse_scores(text: str) -> Dict[str, float]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    data = json.loads(text)
    return {k: float(v) for k, v in data.get("scores", {}).items()}


def _detect_complete_fn() -> CompleteFn:
    """Build a completion function from whichever provider key is set."""
    from ctxwitch.core.judge import get_provider

    provider = get_provider()
    if provider == "anthropic":
        return _anthropic_complete
    if provider == "openai":
        return _openai_complete
    raise NoProviderError(
        "Live eval needs ANTHROPIC_API_KEY or OPENAI_API_KEY (or an injected complete_fn)."
    )


def _anthropic_complete(system: str, user: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _openai_complete(system: str, user: str, model: str) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content
