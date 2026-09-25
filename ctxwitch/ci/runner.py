"""Run CBIA across every changed file in a PR and aggregate one report.

Given a base git ref and a head (default: working tree / HEAD), this:
  1. lists the files changed between them and keeps only those on an analyzed
     surface (see ctxwitch.extract.scope) — eval prompts, vendored code,
     generated data etc. are listed as skipped, not scored,
  2. reports byte-identical delete + add pairs as moves, not as changes,
  3. for each witch.yaml-style context file, diffs it via CBIA,
  4. for each agent-code file (.py), extracts the behavioral surface at both
     revisions and diffs *that* via CBIA,
  5. aggregates all per-dimension impacts, attributes them to files, and
     computes an overall compound severity + a pass/block verdict.

No network, no telemetry — pure local git + the deterministic CBIA pipeline.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ctxwitch.core.behavioral import analyze_behavioral_impact
from ctxwitch.core.dimensions import Severity
from ctxwitch.extract.scope import (
    AGENT,
    CODING_ASSISTANT,
    EXCLUDED,
    classify_path,
    detect_moves,
    is_analyzed,
)

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
    surface: str = AGENT

    @property
    def advisory(self) -> str:
        return _ADVISORY.get(self.severity, "review recommended")

    @property
    def emoji(self) -> str:
        return _EMOJI.get(self.severity, "⚪")


@dataclass
class CIMove:
    """A file moved with byte-identical content — no behavioral change."""

    old_file: str
    new_file: str


@dataclass
class CISkip:
    """A changed file left out because it isn't an analyzed agent surface."""

    file: str
    reason: str
    surface: str = EXCLUDED


@dataclass
class CIReport:
    changes: List[CIChange] = field(default_factory=list)
    files_scanned: int = 0
    moves: List[CIMove] = field(default_factory=list)
    skipped: List[CISkip] = field(default_factory=list)
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
            return title + "\n" + body + self._scope_notes() + _footer(version)

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
            if c.surface != AGENT:
                reason = f"[{c.surface}] {reason}"
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
        return title + header + table + verdict + self._scope_notes() + _footer(version)

    def _scope_notes(self) -> str:
        out = ""
        if self.moves:
            out += "\n**Moved without content change** (not scored):\n"
            for m in self.moves:
                out += f"- `{m.old_file}` → `{m.new_file}`\n"
        if self.skipped:
            n = len(self.skipped)
            out += (
                f"\n<details><summary>{n} changed file{'s' if n != 1 else ''} "
                "not analyzed (not the product agent's behavioral surface)"
                "</summary>\n\n"
            )
            for s in self.skipped:
                out += f"- `{s.file}` — {s.reason}\n"
            if any(s.surface == CODING_ASSISTANT for s in self.skipped):
                out += (
                    "\nCoding-assistant instructions (CLAUDE.md, AGENTS.md, "
                    ".claude/ …) are analyzed with `--include-coding-assistant`.\n"
                )
            out += "\n</details>\n"
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "compound_severity": self.compound_severity.label,
            "blocked": self.blocked,
            "changes": [
                {
                    "file": c.file,
                    "surface": c.surface,
                    "dimension": c.dimension,
                    "severity": c.severity.label,
                    "reason": c.reason,
                    "advisory": c.advisory,
                }
                for c in self.changes
            ],
            "moves": [{"from": m.old_file, "to": m.new_file} for m in self.moves],
            "skipped": [{"file": s.file, "reason": s.reason} for s in self.skipped],
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
    include_coding_assistant: bool = False,
    include_paths: Sequence[str] = (),
) -> CIReport:
    """Aggregate CBIA across the files changed between `base` and `head`.

    Only the product agent's surface is analyzed by default. Coding-assistant
    instructions (CLAUDE.md, AGENTS.md, .claude/ …) are opt-in via
    `include_coding_assistant`; `include_paths` globs force any path in.
    """
    root = Path(repo_root or Path.cwd())
    report = CIReport(fail_on=_FAIL_ON.get(fail_on, Severity.BREAKING))
    surfaces = {AGENT, CODING_ASSISTANT} if include_coding_assistant else {AGENT}

    sources: Dict[str, Tuple[str, str]] = {}
    for path in _changed_files(base, head, root):
        if not is_analyzed(path, surfaces, include_paths):
            scope = classify_path(path)
            report.skipped.append(CISkip(path, scope.label, scope.surface))
            continue
        old_src = _git_show(base, path, root)
        new_src = _git_show(head, path, root) if head else _read_worktree(root / path)
        sources[path] = (old_src, new_src)

    for old_path, new_path in detect_moves(sources):
        src = sources[new_path][1]
        if _has_surface(new_path, src, framework):
            report.moves.append(CIMove(old_path, new_path))
        del sources[old_path], sources[new_path]

    for path, (old_src, new_src) in sources.items():
        impacts = _analyze_file(path, old_src, new_src, framework)
        surface = classify_path(path).surface
        for dim, sev, reason in impacts:
            if sev > Severity.NO_CHANGE:
                report.changes.append(CIChange(str(path), dim, sev, reason, surface))
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


def _has_surface(path, src, framework) -> bool:
    """Would this file be analyzed as an agent surface at all?"""
    p = str(path).lower()
    try:
        if p.endswith((".yaml", ".yml")):
            return _is_context(yaml.safe_load(src))
        if p.endswith(".py"):
            from ctxwitch.extract.extractor import extract_snapshots

            return bool(extract_snapshots(src, source_file=str(path), framework=framework))
    except Exception:
        return False
    return False


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
    # --no-renames: a rename then lists both the deleted and the added path, so
    # detect_moves can tell a pure move from a move-plus-edit. With rename
    # detection on (git's default), --name-only lists only the new path and
    # the move reads as a brand-new file.
    args = ["git", "diff", "--name-only", "--no-renames", base]
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
