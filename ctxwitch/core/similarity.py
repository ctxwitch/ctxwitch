"""Segment similarity engine and directive contradiction detector.

Tier 3 of the CBIA pipeline. Provides two analysis modes:
1. Segment similarity — compare decomposed prompt segments between versions
   using embedding cosine similarity (with SequenceMatcher fallback)
2. Directive contradiction — detect when old and new prompts contain
   contradictory instructions (e.g., "approve refunds" vs "reject refunds")

The embedding path uses sentence-transformers if available. Falls back to
a token-overlap + SequenceMatcher approach that requires zero dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from ctxwitch.core.decompose import DecomposedPrompt, PromptSegment, SegmentType
from ctxwitch.core.dimensions import Dimension, DimensionImpact, Severity

_EMBEDDER = None
_EMBEDDING_AVAILABLE = None


def _check_embeddings() -> bool:
    global _EMBEDDING_AVAILABLE
    if _EMBEDDING_AVAILABLE is not None:
        return _EMBEDDING_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_AVAILABLE = True
    except ImportError:
        _EMBEDDING_AVAILABLE = False
    return _EMBEDDING_AVAILABLE


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


@dataclass
class SegmentMatch:
    """A matched pair of segments between old and new prompts."""

    old_segment: Optional[PromptSegment]
    new_segment: Optional[PromptSegment]
    similarity: float
    match_type: str  # "matched", "added", "removed", "replaced"
    similarity_method: str = ""  # "cosine", "token-sequence", or "" for exact/no-comparison


@dataclass
class ContradictionResult:
    """A detected contradiction between old and new directives."""

    old_directive: str
    new_directive: str
    contradiction_type: str  # "negated", "reversed", "conflicting", "weakened", "strengthened"
    confidence: float
    description: str


SEGMENT_TO_DIMENSION: Dict[SegmentType, Dimension] = {
    SegmentType.PERSONA: Dimension.PERSONA,
    SegmentType.TASK: Dimension.TASK_SCOPE,
    SegmentType.CONSTRAINT_MUST: Dimension.CONSTRAINTS,
    SegmentType.CONSTRAINT_MUST_NOT: Dimension.CONSTRAINTS,
    SegmentType.OUTPUT_FORMAT: Dimension.OUTPUT_FORMAT,
    SegmentType.TONE: Dimension.TONE,
    SegmentType.SAFETY: Dimension.SAFETY,
    SegmentType.EXAMPLE: Dimension.KNOWLEDGE_SCOPE,
    SegmentType.CONTEXT: Dimension.KNOWLEDGE_SCOPE,
    SegmentType.DECISION_RULE: Dimension.AUTONOMY,
    SegmentType.ESCALATION: Dimension.AUTONOMY,
    SegmentType.FALLBACK: Dimension.ERROR_HANDLING,
    SegmentType.UNKNOWN: Dimension.TASK_SCOPE,
}


def match_segments(
    old: DecomposedPrompt, new: DecomposedPrompt
) -> List[SegmentMatch]:
    """Match segments between two decomposed prompts by type, then similarity.

    Segments of the same type are matched greedily by highest similarity.
    Unmatched old segments are "removed". Unmatched new segments are "added".
    """
    matches: List[SegmentMatch] = []
    used_new: Set[int] = set()

    all_types = set(
        s.segment_type for s in old.segments
    ) | set(
        s.segment_type for s in new.segments
    )

    for seg_type in all_types:
        old_segs = [s for s in old.segments if s.segment_type == seg_type]
        new_segs = [(i, s) for i, s in enumerate(new.segments) if s.segment_type == seg_type]

        available_new = [(i, s) for i, s in new_segs if i not in used_new]

        for old_seg in old_segs:
            best_sim = -1.0
            best_idx = -1
            best_new_seg = None
            best_method = ""

            for i, new_seg in available_new:
                sim, method = _compute_similarity(old_seg.text, new_seg.text)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
                    best_new_seg = new_seg
                    best_method = method

            if best_new_seg is not None and best_sim > 0.3:
                used_new.add(best_idx)
                available_new = [(i, s) for i, s in available_new if i != best_idx]

                if best_sim > 0.5:
                    match_type = "matched"
                else:
                    match_type = "replaced"

                matches.append(SegmentMatch(
                    old_segment=old_seg,
                    new_segment=best_new_seg,
                    similarity=best_sim,
                    match_type=match_type,
                    similarity_method=best_method,
                ))
            else:
                matches.append(SegmentMatch(
                    old_segment=old_seg,
                    new_segment=None,
                    similarity=0.0,
                    match_type="removed",
                ))

    for i, seg in enumerate(new.segments):
        if i not in used_new:
            matches.append(SegmentMatch(
                old_segment=None,
                new_segment=seg,
                similarity=0.0,
                match_type="added",
            ))

    return matches


def analyze_segment_changes(matches: List[SegmentMatch]) -> List[DimensionImpact]:
    """Convert segment matches into behavioral dimension impacts."""
    impacts: List[DimensionImpact] = []

    for m in matches:
        seg_type = (
            m.old_segment.segment_type if m.old_segment
            else m.new_segment.segment_type if m.new_segment
            else SegmentType.UNKNOWN
        )
        dimension = SEGMENT_TO_DIMENSION.get(seg_type, Dimension.TASK_SCOPE)

        dim_label = dimension.display_name

        if m.match_type == "removed":
            is_safety = seg_type in (SegmentType.SAFETY, SegmentType.CONSTRAINT_MUST_NOT)
            severity = Severity.BREAKING if is_safety else Severity.SIGNIFICANT
            impacts.append(DimensionImpact(
                dimension=dimension,
                severity=severity,
                reason=f"{dim_label} removed: {_truncate(m.old_segment.text)}",
                old_signal=m.old_segment.text,
                new_signal="(removed)",
                confidence=m.old_segment.confidence,
            ))

        elif m.match_type == "added":
            severity = Severity.SIGNIFICANT if seg_type != SegmentType.UNKNOWN else Severity.MINOR
            impacts.append(DimensionImpact(
                dimension=dimension,
                severity=severity,
                reason=f"{dim_label} added: {_truncate(m.new_segment.text)}",
                old_signal="(none)",
                new_signal=m.new_segment.text,
                confidence=m.new_segment.confidence,
            ))

        elif m.match_type == "replaced":
            sim_label = _method_display(m.similarity_method)
            impacts.append(DimensionImpact(
                dimension=dimension,
                severity=Severity.SIGNIFICANT,
                reason=f"{dim_label} replaced ({m.similarity:.0%} {sim_label})",
                old_signal=m.old_segment.text,
                new_signal=m.new_segment.text,
                confidence=min(m.old_segment.confidence, m.new_segment.confidence),
            ))

        elif m.match_type == "matched" and m.similarity < 0.85:
            if m.similarity < 0.6:
                severity = Severity.SIGNIFICANT
            elif m.similarity < 0.75:
                severity = Severity.MINOR
            else:
                severity = Severity.COSMETIC

            sim_label = _method_display(m.similarity_method)
            impacts.append(DimensionImpact(
                dimension=dimension,
                severity=severity,
                reason=f"{dim_label} modified ({m.similarity:.0%} {sim_label})",
                old_signal=m.old_segment.text,
                new_signal=m.new_segment.text,
                confidence=min(m.old_segment.confidence, m.new_segment.confidence),
            ))

    return impacts


# ─── Numeric Threshold Detection ───────────────────────────────────────────

# Currency/percent-aware number token, e.g. $100, 1,000, 3.5, 20%, €50
_NUMBER_RE = re.compile(r"[$€£₹]?\d[\d,]*(?:\.\d+)?%?")

_DIRECTIVE_TYPES = {
    SegmentType.CONSTRAINT_MUST,
    SegmentType.CONSTRAINT_MUST_NOT,
    SegmentType.SAFETY,
    SegmentType.DECISION_RULE,
    SegmentType.ESCALATION,
}


def detect_numeric_threshold_changes(
    matches: List[SegmentMatch],
) -> List[DimensionImpact]:
    """Detect numeric threshold changes inside otherwise-unchanged segments.

    "Escalate refunds above $100" → "Escalate refunds above $500" is nearly
    identical textually (similarity ~0.95, so Tier 3 scores it Cosmetic at
    best), but the number is the behaviorally load-bearing part of the rule.
    This detector compares matched segments with their numbers masked out:
    if the surrounding text is (near-)identical while the numbers differ,
    the change is a threshold shift, not a rewording.
    """
    impacts: List[DimensionImpact] = []

    for m in matches:
        if m.match_type != "matched" or not m.old_segment or not m.new_segment:
            continue

        old_nums = _NUMBER_RE.findall(m.old_segment.text)
        new_nums = _NUMBER_RE.findall(m.new_segment.text)
        if old_nums == new_nums or (not old_nums and not new_nums):
            continue

        old_masked = _mask_numbers(m.old_segment.text)
        new_masked = _mask_numbers(m.new_segment.text)
        if old_masked == new_masked:
            confidence = 0.95
        elif _fallback_similarity(old_masked, new_masked) > 0.8:
            # numbers changed amid light rewording — still likely a threshold shift
            confidence = 0.75
        else:
            continue  # a real rewrite; Tier 3 already scored it

        seg_type = m.old_segment.segment_type
        dimension = SEGMENT_TO_DIMENSION.get(seg_type, Dimension.CONSTRAINTS)
        severity = (
            Severity.SIGNIFICANT if seg_type in _DIRECTIVE_TYPES else Severity.MINOR
        )

        changed = _describe_number_change(old_nums, new_nums)
        impacts.append(DimensionImpact(
            dimension=dimension,
            severity=severity,
            reason=(
                f"Numeric threshold changed in {dimension.display_name.lower()} "
                f"directive: {changed}. The rule's wording is unchanged — only "
                f"its trigger value moved."
            ),
            old_signal=m.old_segment.text,
            new_signal=m.new_segment.text,
            confidence=confidence,
        ))

    return impacts


def _mask_numbers(text: str) -> str:
    """Replace number tokens with a placeholder and normalize whitespace."""
    return " ".join(_NUMBER_RE.sub("#", text).split()).lower()


def _describe_number_change(old_nums: List[str], new_nums: List[str]) -> str:
    """Human-readable summary of which numbers moved."""
    changed = [
        f"{o} → {n}" for o, n in zip(old_nums, new_nums) if o != n
    ]
    if len(old_nums) > len(new_nums):
        changed.extend(f"{o} removed" for o in old_nums[len(new_nums):])
    elif len(new_nums) > len(old_nums):
        changed.extend(f"{n} added" for n in new_nums[len(old_nums):])
    return ", ".join(changed)


# ─── Contradiction Detection ───────────────────────────────────────────────

_NEGATION_PAIRS = [
    (r"\bapprove\b", r"\breject\b"),
    (r"\bapprove\b", r"\bdeny\b"),
    (r"\baccept\b", r"\brefuse\b"),
    (r"\ballow\b", r"\bblock\b"),
    (r"\ballow\b", r"\bprohibit\b"),
    (r"\binclude\b", r"\bexclude\b"),
    (r"\benable\b", r"\bdisable\b"),
    (r"\bfriendly\b", r"\bstrict\b"),
    (r"\bempathetic\b", r"\bstrict\b"),
    (r"\bcasual\b", r"\bformal\b"),
    (r"\bverbose\b", r"\bconcise\b"),
    (r"\bdetailed\b", r"\bbrief\b"),
    (r"\bhelpful\b", r"\brestrictive\b"),
    (r"\bgenerous\b", r"\bstrict\b"),
    (r"\bflexible\b", r"\brigid\b"),
    (r"\bencourage\b", r"\bdiscourage\b"),
]

_WEAKENING_PATTERNS = [
    (r"\bmust\b", r"\bshould\b", "weakened"),
    (r"\balways\b", r"\busually\b", "weakened"),
    (r"\bnever\b", r"\brarely\b", "weakened"),
    (r"\brequired\b", r"\boptional\b", "weakened"),
    (r"\bshould\b", r"\bmust\b", "strengthened"),
    (r"\busually\b", r"\balways\b", "strengthened"),
    (r"\brarely\b", r"\bnever\b", "strengthened"),
    (r"\boptional\b", r"\brequired\b", "strengthened"),
]

# Canonical (apostrophe-less, lowercase) negation vocabulary. Tokens are
# normalized before lookup, so "don't", "dont", "Don't." and the curly
# "don’t" all resolve to the same entry.
_NEGATION_WORDS = {
    "not", "never", "no", "dont", "doesnt", "wont", "cannot",
    "cant", "shouldnt", "mustnt", "isnt", "arent", "didnt",
    "couldnt", "wouldnt", "without", "neither", "nor",
    "refuse", "reject", "deny", "prohibit", "forbid", "block",
}

# Words eligible for typo-tolerant (edit-distance-1) matching. Restricted to
# longer, distinctive entries; short words like "no"/"deny"/"wont" are exact-
# only because their 1-edit neighborhoods collide with everyday words
# ("want", "eject", "won").
_FUZZY_NEGATION = {
    w for w in _NEGATION_WORDS if len(w) >= 5 and w != "block"
}

# Legitimate words that sit one edit away from a negation term and must never
# count as one: ever/never is handled by the length gate; these are not.
_FUZZY_STOPLIST = {
    "either",   # 1 edit from "neither"
    "reuse",    # 1 edit from "refuse"
    "eject",    # 1 edit from "reject"
    "fever", "lever", "sever",  # 1 edit from "never"
    "cannon",   # 1 edit from "cannot"
    "black",    # 1 edit from "block" (also excluded via _FUZZY_NEGATION)
}

_SUFFIXES = ("s", "es", "ed", "ing")

_PUNCT_STRIP = ".,;:!?()[]{}\"'`‘’“”"


def _normalize_token(token: str) -> str:
    """Lowercase, strip surrounding punctuation, drop apostrophes."""
    t = token.lower().strip(_PUNCT_STRIP)
    return t.replace("'", "").replace("’", "")


def _edit_distance_leq1(a: str, b: str) -> bool:
    """True when a and b differ by at most one insert/delete/substitute."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if la == lb:
            i += 1
        j += 1
    return True


def _is_negation_token(token: str) -> bool:
    """Vocabulary lookup with normalization, morphology, and typo tolerance."""
    t = _normalize_token(token)
    if not t:
        return False
    if t in _NEGATION_WORDS:
        return True
    # simple morphology: refuses / rejected / forbidding → base form
    for suffix in _SUFFIXES:
        if t.endswith(suffix) and t[: -len(suffix)] in _NEGATION_WORDS:
            return True
    # guarded typo tolerance: one edit from a long, distinctive negation word
    if len(t) >= 5 and t not in _FUZZY_STOPLIST:
        return any(_edit_distance_leq1(t, w) for w in _FUZZY_NEGATION)
    return False


def _count_negations(text: str) -> int:
    return sum(1 for tok in text.split() if _is_negation_token(tok))


def detect_contradictions(
    old: DecomposedPrompt, new: DecomposedPrompt
) -> List[ContradictionResult]:
    """Detect contradictions between old and new prompt directives.

    Checks for:
    1. Semantic negation (approve -> reject)
    2. Added/removed negation words (must do X -> must not do X)
    3. Weakened/strengthened modality (must -> should, always -> usually)
    """
    contradictions: List[ContradictionResult] = []

    old_directives = _extract_directives(old)
    new_directives = _extract_directives(new)

    for old_d in old_directives:
        for new_d in new_directives:
            if _text_overlap(old_d, new_d) < 0.2:
                continue

            for pat_a, pat_b in _NEGATION_PAIRS:
                a_in_old = bool(re.search(pat_a, old_d, re.I))
                b_in_old = bool(re.search(pat_b, old_d, re.I))
                a_in_new = bool(re.search(pat_a, new_d, re.I))
                b_in_new = bool(re.search(pat_b, new_d, re.I))

                if (a_in_old and b_in_new) or (b_in_old and a_in_new):
                    contradictions.append(ContradictionResult(
                        old_directive=old_d,
                        new_directive=new_d,
                        contradiction_type="reversed",
                        confidence=0.85,
                        description=f"Directive reversed: '{_extract_verb(pat_a)}' changed to '{_extract_verb(pat_b)}'",
                    ))
                    break

            old_neg_count = _count_negations(old_d)
            new_neg_count = _count_negations(new_d)

            if abs(old_neg_count - new_neg_count) > 0 and _text_overlap(old_d, new_d) > 0.4:
                if old_neg_count > new_neg_count:
                    contradictions.append(ContradictionResult(
                        old_directive=old_d,
                        new_directive=new_d,
                        contradiction_type="negated",
                        confidence=0.70,
                        description="Negation removed — prohibition may have become permission.",
                    ))
                else:
                    contradictions.append(ContradictionResult(
                        old_directive=old_d,
                        new_directive=new_d,
                        contradiction_type="negated",
                        confidence=0.70,
                        description="Negation added — permission may have become prohibition.",
                    ))

            for pat_old, pat_new, change_type in _WEAKENING_PATTERNS:
                if re.search(pat_old, old_d, re.I) and re.search(pat_new, new_d, re.I):
                    if _text_overlap(old_d, new_d) > 0.3:
                        contradictions.append(ContradictionResult(
                            old_directive=old_d,
                            new_directive=new_d,
                            contradiction_type=change_type,
                            confidence=0.75,
                            description=f"Directive {change_type}: modality shifted.",
                        ))

    seen = set()
    unique = []
    for c in contradictions:
        key = (c.old_directive[:50], c.new_directive[:50], c.contradiction_type)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def contradictions_to_impacts(
    contradictions: List[ContradictionResult],
) -> List[DimensionImpact]:
    """Convert contradiction results into dimension impacts."""
    impacts = []
    for c in contradictions:
        if c.contradiction_type == "reversed":
            severity = Severity.BREAKING
        elif c.contradiction_type == "negated":
            severity = Severity.BREAKING
        elif c.contradiction_type == "strengthened":
            severity = Severity.SIGNIFICANT
        elif c.contradiction_type == "weakened":
            severity = Severity.SIGNIFICANT
        else:
            severity = Severity.MINOR

        impacts.append(DimensionImpact(
            dimension=Dimension.CONSTRAINTS,
            severity=severity,
            reason=c.description,
            old_signal=c.old_directive,
            new_signal=c.new_directive,
            confidence=c.confidence,
        ))

    return impacts


# ─── Internal helpers ──────────────────────────────────────────────────────

def _compute_similarity(text_a: str, text_b: str) -> Tuple[float, str]:
    """Compute similarity between two text segments.

    Returns (score, method_used).
    Uses sentence-transformers cosine similarity if available,
    falls back to token overlap + SequenceMatcher.
    """
    if not text_a or not text_b:
        return 0.0, ""

    if text_a.strip() == text_b.strip():
        return 1.0, "exact"

    if _check_embeddings():
        try:
            return _embedding_similarity(text_a, text_b), "cosine"
        except Exception:
            pass

    return _fallback_similarity(text_a, text_b), "token-sequence"


def _embedding_similarity(text_a: str, text_b: str) -> float:
    import numpy as np
    embedder = _get_embedder()
    embeddings = embedder.encode([text_a, text_b])
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(cos_sim)


def _fallback_similarity(text_a: str, text_b: str) -> float:
    """Token overlap + SequenceMatcher when embeddings unavailable."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    seq_ratio = SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    return 0.4 * jaccard + 0.6 * seq_ratio


def _extract_directives(prompt: DecomposedPrompt) -> List[str]:
    """Extract directive sentences from a decomposed prompt."""
    directive_types = {
        SegmentType.CONSTRAINT_MUST,
        SegmentType.CONSTRAINT_MUST_NOT,
        SegmentType.SAFETY,
        SegmentType.DECISION_RULE,
        SegmentType.ESCALATION,
        SegmentType.TASK,
        SegmentType.TONE,
    }
    return [
        s.text for s in prompt.segments
        if s.segment_type in directive_types
    ]


def _text_overlap(a: str, b: str) -> float:
    """Quick token overlap ratio (punctuation/apostrophe-normalized)."""
    tokens_a = {t for t in (_normalize_token(w) for w in a.split()) if t}
    tokens_b = {t for t in (_normalize_token(w) for w in b.split()) if t}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _extract_verb(pattern: str) -> str:
    """Extract the word from a regex pattern like r'\\bapprove\\b'."""
    return pattern.replace(r"\b", "")


_METHOD_LABELS = {
    "cosine": "cosine similarity",
    "token-sequence": "token-sequence similarity",
    "exact": "exact match",
}


def _method_display(method: str) -> str:
    """Return a human-readable label for the actual method used."""
    return _METHOD_LABELS.get(method, "similarity")


def _truncate(text: str, length: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    return text[:length] + "..."
