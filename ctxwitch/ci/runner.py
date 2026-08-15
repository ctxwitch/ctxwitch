"""Run CBIA across every changed file in a PR and aggregate one report.

Given a base git ref and a head (default: working tree / HEAD), this:
  1. lists the files changed between them,
  2. for each witch.yaml-style context file, diffs it via CBIA,
  3. for each agent-code file (.py), extracts the behavioral surface at both
     revisions and diffs *that* via CBIA,
  4. aggregates all per-dimension impacts, attributes them to files, and
     computes an overall compound severity + a pass/block verdict.

No network, no telemetry — pure local git + the deterministic CBIA pipeline.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.dimensions import Severity

# severity → the resource the change should be routed to (advisory text only;
# the free Action does not trigger these — it recommends them).
_ADVISORY = {
    Severity.BREAKING: "requires human / security review",
    Severity.SIGNIFICANT: "run evaluation suite / agent replay",
    Severity.MINOR: "review recommended",
    Severity.COSMETIC: "no additional testing",
}

_EMOJI = {
    Severity.BREAKING: "🔴",
    Severity.SIGNIFICANT: "🟠",
    Severity.MINOR: "🟡",
    Severity.COSMETIC: "🟢",
    Severity.NO_CHANGE: "⚪",
}

_FAIL_ON = {
    "breaking": Severity.BREAKING,
    "significant": Severity.SIGNIFICANT,
    "minor": Severity.MINOR,
    "never": None,
}


@dataclass
class CIChange:
    """One behavioral change attributed to a file."""

    file: str
    dimension: str
    severity: Severity
    reason: str

    @property
    def advisory(self) -> str:
        return _ADVISORY.get(self.severity, "review recommended")

    @property
    def emoji(self) -> str:
        return _EMOJI.get(self.severity, "⚪")


@dataclass
class CIReport:
    changes: List[CIChange] = field(default_factory=list)
    files_scanned: int = 0
    compound_severity: Severity = Severity.NO_CHANGE
    fail_on: Optional[Severity] = Severity.BREAKING

    @property
    def blocked(self) -> bool:
        if self.fail_on is None:
            return False
        return self.compound_severity >= self.fail_on

    def to_markdown(self, version: str = "") -> str:
        title = "## 🧙 ctxwitch — Agent Behavioral Scan\n"
        if not self.changes:
            body = (
                "No behavioral changes detected in this PR's agent config or code.\n"
            )
            return title + "\n" + body + _footer(version)

        # order most-severe first
        rows = sorted(self.changes, key=lambda c: -int(c.severity))
        n = len(rows)
        header = (
            f"\n**{n} behavioral change{'s' if n != 1 else ''} detected** "
            f"across {self.files_scanned} scanned file"
            f"{'s' if self.files_scanned != 1 else ''}.\n\n"
        )
        table = (
            "| Severity | Change | Dimension | Recommended |\n"
            "|---|---|---|---|\n"
        )
        for c in rows:
            reason = c.reason.replace("|", "\\|").strip()
            if len(reason) > 80:
                reason = reason[:77] + "…"
            table += (
                f"| {c.emoji} {c.severity.label} | {reason} | "
                f"{c.dimension} | {c.advisory} |\n"
            )

        verdict = (
            f"\n**Policy result:** ❌ merge blocked "
            f"({self.compound_severity.label} change)\n"
            if self.blocked
            else "\n**Policy result:** ✅ passed\n"
        )
        return title + header + table + verdict + _footer(version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "compound_severity": self.compound_severity.label,
            "blocked": self.blocked,
            "changes": [
                {
                    "file": c.file,
                    "dimension": c.dimension,
                    "severity": c.severity.label,
                    "reason": c.reason,
                    "advisory": c.advisory,
                }
                for c in self.changes
            ],
        }


def _footer(version: str) -> str:
    v = f" v{version}" if version else ""
    return (
        "\n<sub>🔒 Ran locally in your CI — no prompts, code, or data left your "
        f"infrastructure. ctxwitch{v}</sub>\n"
    )


# ── the runner ──────────────────────────────────────────────────────────────


def run_ci(
    base: str,
    head: Optional[str] = None,
    repo_root: Optional[Path] = None,
    framework: Optional[str] = None,
    fail_on: str = "breaking",
) -> CIReport:
    """Aggregate CBIA across the files changed between `base` and `head`."""
    root = Path(repo_root or Path.cwd())
    report = CIReport(fail_on=_FAIL_ON.get(fail_on, Severity.BREAKING))

    for path in _changed_files(base, head, root):
        old_src = _git_show(base, path, root)
        new_src = _git_show(head, path, root) if head else _read_worktree(root / path)
        impacts = _analyze_file(path, old_src, new_src, framework)
        for dim, sev, reason in impacts:
            if sev > Severity.NO_CHANGE:
                report.changes.append(CIChange(str(path), dim, sev, reason))
        if impacts:
            report.files_scanned += 1

    if report.changes:
        report.compound_severity = max(c.severity for c in report.changes)
    return report


def _analyze_file(path, old_src, new_src, framework):
    """Return [(dimension, severity, reason)] for one changed file, or []."""
    p = str(path).lower()
    try:
        if p.endswith((".yaml", ".yml")):
            return _analyze_yaml(old_src, new_src)
        if p.endswith(".py"):
            return _analyze_code(old_src, new_src, framework)
    except Exception:
        # A single malformed file must never fail the whole CI run.
        return []
    return []


def _analyze_yaml(old_src, new_src):
    old = yaml.safe_load(old_src) if old_src else {}
    new = yaml.safe_load(new_src) if new_src else {}
    # Only treat it as an agent context if it carries behavioral components.
    if not _is_context(old) and not _is_context(new):
        return []
    report = analyze_behavioral_impact(old or {}, new or {})
    return _impacts(report)


def _analyze_code(old_src, new_src, framework):
    from ctxwitch.extract.extractor import diff_code

    if not old_src and not new_src:
        return []
    report, old_snap, new_snap = diff_code(
        old_src or "", new_src or "", framework=framework
    )
    if old_snap is None and new_snap is None:
        return []  # no agent in this file
    return _impacts(report)


def _impacts(report):
    return [
        (i.dimension.display_name, i.severity, i.reason or "")
        for i in report.impacts
        if i.severity > Severity.NO_CHANGE
    ]


def _is_context(data) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("components"), dict)
        and ("system_prompt" in data["components"] or "model" in data["components"])
    )


# ── git helpers ─────────────────────────────────────────────────────────────


def _changed_files(base: str, head: Optional[str], root: Path) -> List[str]:
    args = ["git", "diff", "--name-only", base]
    if head:
        args.append(head)
    try:
        out = subprocess.run(
            args, cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except Exception:
        return []
    files = []
    for line in out.splitlines():
        line = line.strip()
        if line.endswith((".py", ".yaml", ".yml")):
            files.append(line)
    return files


def _git_show(ref: str, path: str, root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return ""  # file didn't exist at that ref (added/removed)


def _read_worktree(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
