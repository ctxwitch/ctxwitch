"""CI integration — run CBIA across a PR's changed files and render a report.

Powers the `witch ci` command and the ctxwitch GitHub Action. Everything runs
locally in the user's CI against their own git history; nothing leaves their
infrastructure (no telemetry, no account, no network calls).
"""

from ctxwitch.ci.runner import CIChange, CIReport, run_ci

__all__ = ["CIChange", "CIReport", "run_ci"]
