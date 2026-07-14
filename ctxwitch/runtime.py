"""Runtime integration API — load governed context into your agent code.

This is the consumption side of ctxwitch: the repo governs witch.yaml,
and the running agent loads it instead of hardcoding prompts and params.

    from ctxwitch.runtime import load_components

    components = load_components(env="prod")
    response = client.messages.create(
        model=components["model"],
        system=components["system_prompt"],
        temperature=components.get("temperature", 0.3),
        max_tokens=components.get("max_tokens", 4096),
        messages=[...],
    )

Environment overrides from the `environments:` block are deep-merged on
top of the base components, so dev/staging/prod diverge only where they
say they do. Set CTXWITCH_ENV to pick the environment without code changes.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ctxwitch.core.context import load_context


def load_agent_context(
    path: Optional[Path] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a validated witch.yaml with environment overrides applied.

    Args:
        path: witch.yaml location; defaults to ./witch.yaml.
        env:  environment name from the `environments:` block; defaults to
              the CTXWITCH_ENV variable, or no overrides when unset.

    Returns the full context dict (version, name, components, ...) with
    the chosen environment's overrides deep-merged in.
    """
    path = Path(path) if path else Path.cwd() / "witch.yaml"
    snapshot = load_context(path)
    data = copy.deepcopy(snapshot.data)

    env = env or os.environ.get("CTXWITCH_ENV")
    if env:
        overrides = data.get("environments", {}).get(env)
        if overrides is None:
            raise KeyError(
                f"Environment '{env}' not defined in {path} "
                f"(available: {sorted(data.get('environments', {}))})"
            )
        _deep_merge(data, overrides)

    return data


def load_components(
    path: Optional[Path] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """Load just the merged `components` block — prompt, model, params."""
    return load_agent_context(path=path, env=env).get("components", {})


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively merge override values into base, in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
