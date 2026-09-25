"""Which changed files are a shipped agent's behavioral surface.

The extractor reads anything prompt- or config-shaped, which is right for a
file you point it at and wrong for a whole PR: repos also carry eval/benchmark
prompts, vendored copies of other projects, generated data, packaging
manifests and the instruction files for the coding assistant that works *on*
the repo. Scoring those as product-agent changes is a false positive.

`classify_path` sorts a repo-relative path into one of three surfaces:

    agent             the product's own agent — analyzed by default
    coding-assistant  CLAUDE.md, AGENTS.md, .claude/**, copilot instructions …
                      real agent config, but for the dev assistant; opt-in
    excluded          not an agent surface (see EXCLUSION_REASONS)

Rules are path shapes, never per-repo names, and are deliberately
conservative: a path is only moved out of `agent` on a clear signal.

`detect_moves` pairs byte-identical delete + add in one change so a file move
is reported as a move, not as one prompt removed and another added.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

AGENT = "agent"
CODING_ASSISTANT = "coding-assistant"
EXCLUDED = "excluded"

DEFAULT_SURFACES = frozenset({AGENT})

EXCLUSION_REASONS: Dict[str, str] = {
    "vendored": "vendored third-party code",
    "generated": "generated data / lockfile",
    "packaging": "packaging / build / CI manifest",
    "evaluation": "evaluation / benchmark prompt",
    "data-pipeline": "offline data pipeline",
    "docs": "documentation",
}


@dataclass(frozen=True)
class PathScope:
    """Where a path sits: its surface, and why when it isn't the product agent."""

    path: str
    surface: str               # AGENT | CODING_ASSISTANT | EXCLUDED
    reason: Optional[str] = None  # EXCLUSION_REASONS key when EXCLUDED

    @property
    def label(self) -> str:
        if self.surface == CODING_ASSISTANT:
            return "coding-assistant instructions"
        if self.surface == EXCLUDED:
            return EXCLUSION_REASONS.get(self.reason or "", self.reason or "excluded")
        return "agent"


# ── rules ────────────────────────────────────────────────────────────────────
# Directory names are matched against each path segment, case-insensitively.

_VENDORED_DIRS = re.compile(
    r"^(vendor|vendors|vendored|_vendor|third[-_]?party|3rd[-_]?party|"
    r"external|externals|external[-_]tools|extern|node_modules|"
    r"bower_components|site-packages|\.?venv)$",
    re.I,
)

_GENERATED_DIRS = re.compile(r"^(generated|__generated__|dist|coverage)$", re.I)
_GENERATED_FILE = re.compile(
    r"(\.lock$|^package-lock\.json$|^pnpm-lock\.ya?ml$|^npm-shrinkwrap\.json$|"
    r"\.generated\.[^.]+$|_pb2(_grpc)?\.py$)",
    re.I,
)
# data dumps named *_data.json / *-data.yaml (e.g. generated agent-address registries)
_DATA_FILE = re.compile(r"(^|[_.-])data\.(json|ya?ml)$", re.I)

# requirements-dev.txt, requirements.lock.txt … but not requirements_analyst.txt
_REQ_VARIANT = (r"(dev|test|tests|testing|docs?|prod|production|lint|ci|build|"
                r"base|local|optional|extras?|all|min|lock|py\d*)")

_PACKAGING_FILE = re.compile(
    r"^(package|plugin|manifest|marketplace|composer|bower|lerna|nx|turbo|"
    r"vercel|deno|renovate|jsconfig|tsconfig(\.[\w-]+)?|\.eslintrc|\.prettierrc|"
    r"\.babelrc)\.json$|"
    r"^(action|dependabot|codecov|mkdocs|environment|conda|chart|pubspec|"
    r"\.readthedocs|\.pre-commit-config|\.gitlab-ci|\.travis|"
    r"(docker-)?compose(\.[\w-]+)?)\.ya?ml$|"
    r"^(setup|_version)\.py$|"
    # plain-text manifests: read as a prompt file they'd score as a prompt edit
    r"^((dev|test|docs?)[-_])?(requirements|constraints)([-_.]" + _REQ_VARIANT + r")*\.txt$|"
    r"^(cmakelists|robots|runtime)\.txt$",
    re.I,
)
_PACKAGING_DIRS = re.compile(
    r"^(\.claude-plugin|\.circleci|\.devcontainer|\.vscode|\.idea)$", re.I
)

# Instruction files for the coding assistant working ON the repo. Filenames are
# case-sensitive on purpose: `CLAUDE.md` is the convention, while a lowercase
# `prompts/claude.md` is far more likely a product prompt tuned for a model.
_CODING_ASSISTANT_FILES = frozenset({
    "CLAUDE.md", "CLAUDE.local.md", "AGENTS.md", "AGENTS.override.md",
    "GEMINI.md", "copilot-instructions.md",
    ".cursorrules", ".windsurfrules", ".clinerules", ".roorules",
    ".aider.conf.yml",
})
_CODING_ASSISTANT_DIRS = re.compile(
    r"^\.(claude|cursor|windsurf|clinerules|roo|continue|gemini|codex|kiro|"
    r"amazonq|junie|aider[\w.-]*)$",
    re.I,
)
# .github/{instructions,prompts,chatmodes}/ are Copilot customization dirs.
_GITHUB_COPILOT_DIRS = frozenset({"instructions", "prompts", "chatmodes", "agents"})

_EVAL_DIRS = re.compile(
    r"^(evals?|evaluations?|evaluators?|benchmarks?|judges?|graders?|"
    r"eval[-_]prompts|[\w.]+[-_]bench)$",
    re.I,
)
# A bare `judge.py` may be a runtime guardrail, so only eval-flavoured names.
_JUDGE_STEM = re.compile(
    r"^(llm[-_]?judges?|([\w-]*[-_])?judges?[-_]prompts?|"
    r"(eval|evaluation|grading|grader)[-_]prompts?)$",
    re.I,
)
# Benchmark datasets whose name alone identifies an eval few-shot prompt.
_BENCHMARKS = frozenset({
    "gsm8k", "gsm_hard", "gsm-hard", "gsmhard", "svamp", "asdiv", "mawps",
    "tabmwp", "mgsm", "math500", "math_500", "minerva_math", "theoremqa",
    "mmlu", "mmlu_pro", "mmlu-pro", "gpqa", "aime", "aime24", "aime25",
    "amc23", "olympiadbench", "humaneval", "mbpp", "livecodebench", "swebench",
    "swe_bench", "swe-bench", "hotpotqa", "strategyqa", "triviaqa", "truthfulqa",
    "commonsenseqa", "hellaswag", "winogrande", "bbh", "finqa", "tatqa",
    "browsecomp", "browsecomp_plus",
})
# Names that are benchmarks only in a few-shot context (`prompts/cot/math.md`).
_AMBIGUOUS_BENCHMARKS = frozenset({"math", "arc", "drop", "nq", "amc", "sat_math"})
_FEWSHOT_DIRS = re.compile(
    r"^(cot|pal|pot|tora|few[-_]?shots?|icl|demos|exemplars|shots)$", re.I
)

_DATA_PIPELINE_DIRS = re.compile(
    r"^(collect[-_]?traces?|traces?[-_]collection|trace[-_]collector|datagen|"
    r"data[-_](gen|generation|pipelines?|prep|preparation|processing|collection|"
    r"synthesis|curation)|synthetic[-_]data|training[-_]data|sft[-_]data|sft|"
    r"fine[-_]?tun(e|ing))$",
    re.I,
)

_DOC_DIRS = re.compile(r"^(docs?|blog|issue_template|\.changesets?)$", re.I)
# Unambiguous doc names match any case; words that could also name a prompt
# (`prompts/security.md`, `history.md`) only in the uppercase convention.
_DOC_FILE = re.compile(
    r"^((?i:readme|changelog|contributing|code_of_conduct|license|licence|"
    r"pull_request_template|llms|llms-full)|SECURITY|NOTICE|AUTHORS|SUPPORT|HISTORY|"
    r"CHANGES|CITATION|ROADMAP|ARCHITECTURE|GOVERNANCE|MAINTAINERS|CONTRIBUTORS|"
    r"UPGRADING|RELEASING|TODO)(\.[\w.-]+)?$"
)


def _normalize(path: str) -> str:
    p = (path or "").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def classify_path(path: str) -> PathScope:
    """Classify a repo-relative path into agent / coding-assistant / excluded."""
    norm = _normalize(path)
    parts = PurePosixPath(norm).parts
    if not parts:
        return PathScope(norm, AGENT)
    name = parts[-1]
    dirs = parts[:-1]
    stem = PurePosixPath(name).stem.lower()

    def excluded(reason: str) -> PathScope:
        return PathScope(norm, EXCLUDED, reason)

    # Order matters: a CLAUDE.md inside vendor/ is vendored first.
    if any(_VENDORED_DIRS.match(d) for d in dirs):
        return excluded("vendored")

    if any(_GENERATED_DIRS.match(d) for d in dirs) or _GENERATED_FILE.search(name) \
            or _DATA_FILE.search(name):
        return excluded("generated")

    if _PACKAGING_FILE.match(name) or any(_PACKAGING_DIRS.match(d) for d in dirs) \
            or (len(dirs) >= 2 and dirs[-2] == ".github" and dirs[-1] == "workflows") \
            or (name.lower().endswith(".txt") and any(d.lower() == "requirements" for d in dirs)):
        return excluded("packaging")

    if name in _CODING_ASSISTANT_FILES \
            or any(_CODING_ASSISTANT_DIRS.match(d) for d in dirs) \
            or (len(dirs) >= 2 and dirs[0] == ".github" and dirs[1] in _GITHUB_COPILOT_DIRS) \
            or (".github" in dirs
                and name.endswith((".instructions.md", ".prompt.md", ".chatmode.md"))):
        return PathScope(norm, CODING_ASSISTANT)

    lower_dirs = [d.lower() for d in dirs]
    if any(_EVAL_DIRS.match(d) for d in dirs) or _JUDGE_STEM.match(stem) \
            or stem in _BENCHMARKS or any(d in _BENCHMARKS for d in lower_dirs) \
            or (stem in _AMBIGUOUS_BENCHMARKS and any(_FEWSHOT_DIRS.match(d) for d in dirs)):
        return excluded("evaluation")

    if any(_DATA_PIPELINE_DIRS.match(d) for d in dirs):
        return excluded("data-pipeline")

    if any(_DOC_DIRS.match(d) for d in dirs) or _DOC_FILE.match(name):
        return excluded("docs")

    return PathScope(norm, AGENT)


def is_analyzed(
    path: str,
    surfaces: Iterable[str] = DEFAULT_SURFACES,
    include: Iterable[str] = (),
) -> bool:
    """True when `path` should be analyzed for the given surfaces.

    `include` is a list of glob patterns that force a path in regardless of
    the default rules — the escape hatch for a repo whose real agent happens
    to live somewhere the heuristics exclude.
    """
    norm = _normalize(path)
    if any(fnmatch.fnmatchcase(norm, g) for g in include):
        return True
    return classify_path(norm).surface in set(surfaces)


# ── moves ────────────────────────────────────────────────────────────────────


def detect_moves(sources: Mapping[str, Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Pair pure moves: identical content deleted at one path, added at another.

    `sources` maps path -> (old_content, new_content), with "" meaning the file
    didn't exist on that side. Returns [(old_path, new_path)]. Empty files are
    never paired (every empty file is "identical"). When several candidates
    share content, a same-basename pair is preferred, then path order.
    """
    deleted: Dict[str, List[str]] = {}
    added: List[str] = []
    for path in sorted(sources):
        old, new = sources[path]
        if old and not new:
            deleted.setdefault(old, []).append(path)
        elif new and not old:
            added.append(path)

    moves: List[Tuple[str, str]] = []
    for new_path in added:
        candidates = deleted.get(sources[new_path][1])
        if not candidates:
            continue
        base = PurePosixPath(new_path).name
        pick = next((c for c in candidates if PurePosixPath(c).name == base), candidates[0])
        candidates.remove(pick)
        moves.append((pick, new_path))
    return moves
