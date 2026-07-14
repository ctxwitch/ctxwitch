"""Prompt decomposer — parse system prompts into typed behavioral segments.

Extracts structured segments from free-form prompt text using heuristic
pattern matching. Each segment is typed (persona, constraint, task, etc.)
enabling segment-by-segment behavioral comparison between prompt versions.

This is fully deterministic — no LLM calls, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SegmentType(str, Enum):
    PERSONA = "persona"
    TASK = "task"
    CONSTRAINT_MUST = "constraint_must"
    CONSTRAINT_MUST_NOT = "constraint_must_not"
    OUTPUT_FORMAT = "output_format"
    TONE = "tone"
    SAFETY = "safety"
    EXAMPLE = "example"
    CONTEXT = "context"
    DECISION_RULE = "decision_rule"
    ESCALATION = "escalation"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        return self.value.replace("_", " ").title()


@dataclass
class PromptSegment:
    """A typed segment extracted from a prompt."""

    segment_type: SegmentType
    text: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    markers: List[str] = field(default_factory=list)

    @property
    def normalized_text(self) -> str:
        return " ".join(self.text.lower().split())


@dataclass
class DecomposedPrompt:
    """A fully decomposed prompt with typed segments."""

    raw_text: str
    segments: List[PromptSegment] = field(default_factory=list)

    def segments_of_type(self, seg_type: SegmentType) -> List[PromptSegment]:
        return [s for s in self.segments if s.segment_type == seg_type]

    @property
    def constraints(self) -> List[PromptSegment]:
        return [
            s for s in self.segments
            if s.segment_type in (SegmentType.CONSTRAINT_MUST, SegmentType.CONSTRAINT_MUST_NOT)
        ]

    @property
    def has_persona(self) -> bool:
        return any(s.segment_type == SegmentType.PERSONA for s in self.segments)

    @property
    def has_safety(self) -> bool:
        return any(s.segment_type == SegmentType.SAFETY for s in self.segments)

    @property
    def segment_types_present(self) -> List[SegmentType]:
        return list(set(s.segment_type for s in self.segments))

    def to_dict(self) -> Dict:
        return {
            "segment_count": len(self.segments),
            "types": [s.value for s in self.segment_types_present],
            "segments": [
                {
                    "type": s.segment_type.value,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in self.segments
            ],
        }


# Pattern definitions for segment classification.
# Each pattern is (compiled_regex, SegmentType, confidence, marker_label).
# Order matters: first match wins for overlapping patterns.

_PERSONA_PATTERNS = [
    (re.compile(r"^you\s+are\s+(?:a|an|the)\s+", re.I), 0.95, "you are a..."),
    (re.compile(r"^act\s+as\s+(?:a|an|the)?\s*", re.I), 0.95, "act as..."),
    (re.compile(r"^you\s+are\s+", re.I), 0.85, "you are..."),
    (re.compile(r"^as\s+(?:a|an)\s+", re.I), 0.80, "as a..."),
    (re.compile(r"^imagine\s+you\s+are\s+", re.I), 0.85, "imagine you are..."),
    (re.compile(r"^pretend\s+(?:to\s+be|you\s+are)\s+", re.I), 0.85, "pretend to be..."),
    (re.compile(r"^your\s+(?:role|name|identity)\s+is\s+", re.I), 0.90, "your role is..."),
]

_CONSTRAINT_MUST_NOT_PATTERNS = [
    # Covers spaced, contracted, and typo'd (apostrophe-less) negated modals:
    # "must not", "mustn't", "mustnt", "don't", "dont", "cannot", "can't",
    # "won't", "shouldn't", ... — straight or curly apostrophe. Verdicts must
    # not depend on orthography (see tests/test_negation_robustness.py).
    (re.compile(
        r"^(?:you\s+)?(?:cannot|can[’']?t|won[’']?t|"
        r"(?:must|should|shall|will|can|do|does|did|could|would)"
        r"(?:n[’']?t|\s+(?:not|never)))\s+",
        re.I,
    ), 0.95, "must not..."),
    (re.compile(r"^(?:do\s+not|don[’']?t|never)\s+", re.I), 0.95, "do not..."),
    (re.compile(r"^avoid\s+", re.I), 0.85, "avoid..."),
    (re.compile(r"^(?:it\s+is\s+)?(?:forbidden|prohibited|not\s+allowed)\s+", re.I), 0.90, "forbidden..."),
    (re.compile(r"^refrain\s+from\s+", re.I), 0.85, "refrain from..."),
    (re.compile(r"^under\s+no\s+circumstances?\s+", re.I), 0.95, "under no circumstances..."),
]

_CONSTRAINT_MUST_PATTERNS = [
    (re.compile(r"^(?:you\s+)?(?:must|shall|should)\s+(?:always\s+)?", re.I), 0.90, "must..."),
    (re.compile(r"^always\s+", re.I), 0.85, "always..."),
    (re.compile(r"^ensure\s+(?:that\s+)?", re.I), 0.80, "ensure..."),
    (re.compile(r"^make\s+sure\s+(?:to\s+|that\s+)?", re.I), 0.80, "make sure..."),
    (re.compile(r"^it\s+is\s+(?:required|mandatory|essential)\s+", re.I), 0.90, "required..."),
    (re.compile(r"^(?:only|exclusively)\s+", re.I), 0.85, "only..."),
]

_OUTPUT_FORMAT_PATTERNS = [
    (re.compile(r"^(?:respond|reply|answer|output|format)\s+(?:in|as|using|with)\s+", re.I), 0.90, "respond in..."),
    (re.compile(r"^(?:respond|reply|answer)\s+(?:clearly|concisely|briefly|precisely|accurately|directly|succinctly)\b", re.I), 0.85, "respond clearly/concisely..."),
    (re.compile(r"^use\s+(?:JSON|markdown|XML|YAML|bullet|numbered|table)\s*", re.I), 0.90, "use format..."),
    (re.compile(r"^(?:the\s+)?(?:output|response|answer)\s+(?:should|must)\s+(?:be\s+)?(?:in\s+)?", re.I), 0.85, "output should be..."),
    (re.compile(r"^(?:keep|limit)\s+(?:your\s+)?(?:response|answer|output)\s+(?:to|under|within)\s+", re.I), 0.85, "keep response to..."),
    (re.compile(r"^(?:be\s+)?(?:concise|brief|detailed|verbose|succinct)\b", re.I), 0.75, "be concise/detailed..."),
    (re.compile(r"^maximum\s+(?:of\s+)?\d+\s+(?:words?|sentences?|paragraphs?|tokens?|characters?)", re.I), 0.90, "maximum N words..."),
]

_TONE_PATTERNS = [
    (re.compile(r"^(?:be\s+)?(?:friendly|professional|casual|formal|empathetic|warm|polite|courteous)\b", re.I), 0.85, "be [tone]..."),
    (re.compile(r"^(?:use\s+a\s+)?(?:friendly|professional|casual|formal|empathetic|warm)\s+(?:tone|voice|manner|style)\b", re.I), 0.90, "use [tone] tone..."),
    (re.compile(r"^(?:speak|talk|communicate)\s+(?:in\s+a\s+)?(?:friendly|professional|casual|formal)\s+", re.I), 0.85, "speak in [tone]..."),
    (re.compile(r"^(?:maintain|adopt|keep)\s+(?:a\s+)?(?:friendly|professional|casual|formal|neutral)\s+", re.I), 0.85, "maintain [tone]..."),
    (re.compile(r"^(?:your\s+)?tone\s+(?:should|must|is)\s+", re.I), 0.90, "tone should..."),
]

_SAFETY_PATTERNS = [
    (re.compile(r"^(?:do\s+not|don'?t|never)\s+(?:reveal|share|disclose|expose|show)\s+(?:your\s+)?(?:system|internal|hidden)\s+", re.I), 0.95, "don't reveal system..."),
    (re.compile(r"^(?:if|when)\s+(?:asked|prompted|requested)\s+(?:about|to\s+reveal)\s+(?:your\s+)?(?:instructions?|prompt|system)\s*", re.I), 0.90, "if asked about instructions..."),
    (re.compile(r"^(?:refuse|decline|reject)\s+(?:to\s+)?(?:answer|respond|engage)\s+", re.I), 0.85, "refuse to answer..."),
    (re.compile(r"^(?:for|regarding)\s+(?:safety|security|privacy|compliance)\s*", re.I), 0.80, "for safety..."),
    (re.compile(r"^(?:content|topic)\s+(?:that\s+is\s+)?(?:harmful|dangerous|illegal|inappropriate|offensive)\s*", re.I), 0.90, "harmful content..."),
    (re.compile(r"^(?:blocked|prohibited|restricted|forbidden)\s+topics?\s*", re.I), 0.90, "blocked topics..."),
]

_DECISION_RULE_PATTERNS = [
    (re.compile(r"^if\s+.+(?:then|,)\s+", re.I), 0.85, "if...then..."),
    (re.compile(r"^when\s+.+(?:then|,)\s+", re.I), 0.85, "when...then..."),
    (re.compile(r"^(?:in\s+(?:the\s+)?case|in\s+cases?)\s+(?:of|where|when)\s+", re.I), 0.85, "in case of..."),
    (re.compile(r"^(?:for|regarding)\s+.+(?:,\s+)", re.I), 0.60, "for X, do Y..."),
]

_ESCALATION_PATTERNS = [
    (re.compile(r"^escalate\s+", re.I), 0.95, "escalate..."),
    (re.compile(r"^(?:transfer|hand\s*off|redirect|forward)\s+(?:to\s+)?(?:a\s+)?(?:human|agent|supervisor|manager|team)\s*", re.I), 0.90, "transfer to human..."),
    (re.compile(r"^(?:if|when)\s+.+(?:escalate|transfer|hand\s*off|redirect)\s+", re.I), 0.85, "if...escalate..."),
    (re.compile(r"^(?:for\s+)?(?:complex|difficult|sensitive|edge)\s+(?:cases?|issues?|situations?)\s*", re.I), 0.75, "for complex cases..."),
]

_EXAMPLE_PATTERNS = [
    (re.compile(r"^(?:for\s+)?example\s*:", re.I), 0.95, "example:"),
    (re.compile(r"^(?:e\.g\.|for\s+instance|such\s+as)\s*", re.I), 0.85, "e.g...."),
    (re.compile(r"^(?:input|user|query|question)\s*:\s*", re.I), 0.75, "input:..."),
    (re.compile(r"^(?:here\s+is\s+an?\s+)?(?:sample|example)\s+", re.I), 0.85, "sample..."),
]

_CONTEXT_PATTERNS = [
    (re.compile(r"^(?:background|context|note|important)\s*:\s*", re.I), 0.85, "background:..."),
    (re.compile(r"^(?:the\s+)?(?:company|organization|product|service)\s+", re.I), 0.70, "the company..."),
    (re.compile(r"^(?:you\s+)?(?:have\s+access|can\s+access)\s+(?:to\s+)?", re.I), 0.80, "you have access to..."),
    (re.compile(r"^(?:the\s+)?(?:user|customer|client)\s+(?:is|has|wants|needs)\s+", re.I), 0.70, "the user is/has..."),
]

_TASK_PATTERNS = [
    (re.compile(r"^(?:your\s+)?(?:task|job|goal|objective|purpose|mission)\s+(?:is\s+)?(?:to\s+)?", re.I), 0.90, "your task is to..."),
    (re.compile(r"^(?:help|assist|support|guide)\s+(?:the\s+)?(?:user|customer|client)\s+", re.I), 0.80, "help the user..."),
    (re.compile(r"^(?:you\s+)?(?:will|should|need\s+to)\s+(?:handle|manage|process|respond\s+to)\s+", re.I), 0.80, "you will handle..."),
    (re.compile(r"^(?:answer|respond\s+to|address|handle)\s+(?:questions?|queries?|requests?|inquiries?)\s+", re.I), 0.80, "answer questions..."),
]

_ALL_PATTERN_GROUPS: List[Tuple[List, SegmentType]] = [
    ([p for p, _, _ in _PERSONA_PATTERNS], SegmentType.PERSONA),
    ([p for p, _, _ in _CONSTRAINT_MUST_NOT_PATTERNS], SegmentType.CONSTRAINT_MUST_NOT),
    ([p for p, _, _ in _CONSTRAINT_MUST_PATTERNS], SegmentType.CONSTRAINT_MUST),
    ([p for p, _, _ in _SAFETY_PATTERNS], SegmentType.SAFETY),
    ([p for p, _, _ in _ESCALATION_PATTERNS], SegmentType.ESCALATION),
    ([p for p, _, _ in _DECISION_RULE_PATTERNS], SegmentType.DECISION_RULE),
    ([p for p, _, _ in _OUTPUT_FORMAT_PATTERNS], SegmentType.OUTPUT_FORMAT),
    ([p for p, _, _ in _TONE_PATTERNS], SegmentType.TONE),
    ([p for p, _, _ in _EXAMPLE_PATTERNS], SegmentType.EXAMPLE),
    ([p for p, _, _ in _CONTEXT_PATTERNS], SegmentType.CONTEXT),
    ([p for p, _, _ in _TASK_PATTERNS], SegmentType.TASK),
]

_PATTERN_LOOKUP = {
    SegmentType.PERSONA: _PERSONA_PATTERNS,
    SegmentType.CONSTRAINT_MUST_NOT: _CONSTRAINT_MUST_NOT_PATTERNS,
    SegmentType.CONSTRAINT_MUST: _CONSTRAINT_MUST_PATTERNS,
    SegmentType.OUTPUT_FORMAT: _OUTPUT_FORMAT_PATTERNS,
    SegmentType.TONE: _TONE_PATTERNS,
    SegmentType.SAFETY: _SAFETY_PATTERNS,
    SegmentType.DECISION_RULE: _DECISION_RULE_PATTERNS,
    SegmentType.ESCALATION: _ESCALATION_PATTERNS,
    SegmentType.EXAMPLE: _EXAMPLE_PATTERNS,
    SegmentType.CONTEXT: _CONTEXT_PATTERNS,
    SegmentType.TASK: _TASK_PATTERNS,
}


def decompose_prompt(text: str) -> DecomposedPrompt:
    """Decompose a system prompt into typed behavioral segments.

    Splits on sentence boundaries, then classifies each sentence/clause
    using heuristic pattern matching against known directive patterns.
    """
    if not text or not text.strip():
        return DecomposedPrompt(raw_text=text)

    sentences = _split_into_sentences(text)
    segments: List[PromptSegment] = []
    pos = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            pos += 1
            continue

        seg_type, confidence, markers = _classify_sentence(sentence)
        start = text.find(sentence, pos)
        end = start + len(sentence) if start >= 0 else pos + len(sentence)

        segments.append(PromptSegment(
            segment_type=seg_type,
            text=sentence,
            confidence=confidence,
            start_pos=max(start, 0),
            end_pos=end,
            markers=markers,
        ))

        pos = end

    return DecomposedPrompt(raw_text=text, segments=segments)


def _split_into_sentences(text: str) -> List[str]:
    """Split prompt text into sentences/clauses for classification."""
    lines = text.split("\n")
    sentences = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith(("-", "*", "•", "–")):
            item = line.lstrip("-*•– ").strip()
            if item:
                sentences.append(item)
            continue

        if re.match(r"^\d+[\.\)]\s+", line):
            item = re.sub(r"^\d+[\.\)]\s+", "", line).strip()
            if item:
                sentences.append(item)
            continue

        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        sentences.extend(p.strip() for p in parts if p.strip())

    return sentences


def _classify_sentence(sentence: str) -> Tuple[SegmentType, float, List[str]]:
    """Classify a sentence into a segment type using pattern matching.

    Returns (segment_type, confidence, list_of_matched_markers).
    Tries patterns in priority order: must-not before must (to avoid
    "must not" matching as "must"), safety before general constraints, etc.
    """
    best_type = SegmentType.UNKNOWN
    best_confidence = 0.0
    best_markers: List[str] = []

    for seg_type, patterns in _PATTERN_LOOKUP.items():
        for pattern, confidence, marker in patterns:
            if pattern.search(sentence):
                if confidence > best_confidence:
                    best_type = seg_type
                    best_confidence = confidence
                    best_markers = [marker]
                elif confidence == best_confidence and seg_type == best_type:
                    best_markers.append(marker)
                break

    if best_type == SegmentType.UNKNOWN:
        best_confidence = _infer_from_keywords(sentence)
        if best_confidence > 0:
            best_type, best_confidence = _keyword_classify(sentence)

    return best_type, best_confidence, best_markers


def _infer_from_keywords(sentence: str) -> float:
    """Check if sentence has any classifiable keywords at all."""
    lower = sentence.lower()
    keywords = [
        "helpful", "assistant", "agent", "respond", "answer",
        "must", "should", "never", "always", "ensure",
        "format", "json", "concise", "escalate", "refuse",
    ]
    return 0.5 if any(kw in lower for kw in keywords) else 0.0


def _keyword_classify(sentence: str) -> Tuple[SegmentType, float]:
    """Fallback keyword-based classification for unmatched sentences."""
    lower = sentence.lower()

    tone_words = {"friendly", "professional", "formal", "informal", "empathetic",
                  "warm", "polite", "courteous", "casual", "strict", "authoritative"}
    if any(w in lower for w in tone_words):
        return SegmentType.TONE, 0.60

    safety_words = {"harmful", "dangerous", "illegal", "inappropriate", "offensive",
                    "sensitive", "confidential", "private", "secret"}
    if any(w in lower for w in safety_words):
        return SegmentType.SAFETY, 0.60

    constraint_words = {"required", "mandatory", "essential", "critical", "important"}
    if any(w in lower for w in constraint_words):
        return SegmentType.CONSTRAINT_MUST, 0.50

    task_words = {"help", "assist", "support", "handle", "process", "manage"}
    if any(w in lower for w in task_words):
        return SegmentType.TASK, 0.50

    return SegmentType.UNKNOWN, 0.30
