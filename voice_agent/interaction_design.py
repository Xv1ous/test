"""Confidence, uncertainty, and capability disclosure for responsible responses."""

from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class AnnotatedReply:
    """Final text shown/synthesized after interaction-design wrapping."""

    text: str
    confidence: ConfidenceLevel
    similarity: float | None


def capability_preamble(capability_statement: str) -> str:
    return capability_statement.strip()


def wrap_answer(
    base_answer: str,
    similarity: float | None,
    *,
    high_threshold: float,
    medium_threshold: float,
) -> AnnotatedReply:
    """
    Prefix/suffix cues based on retrieval confidence (cosine similarity).
    Low similarity → uncertainty + redirect to official sources.
    """
    if similarity is None:
        text = base_answer.strip()
        return AnnotatedReply(text=text, confidence=ConfidenceLevel.NONE, similarity=None)

    if similarity >= high_threshold:
        prefix = "I'm fairly confident in this summary: "
        level = ConfidenceLevel.HIGH
        suffix = ""
    elif similarity >= medium_threshold:
        prefix = "I think this matches your question, but please confirm on the official site: "
        level = ConfidenceLevel.MEDIUM
        suffix = " If anything conflicts, trust the university's published requirements."
    else:
        prefix = "I'm not sure I understood your question against my knowledge base. "
        level = ConfidenceLevel.LOW
        suffix = (
            " I can only help with general admissions topics I've been given. "
            "For accurate, up-to-date answers, contact the admissions office or check the official portal."
        )
        base_answer = base_answer.strip() or ""

    body = base_answer.strip()
    if level == ConfidenceLevel.LOW and not body:
        text = (prefix + suffix).strip()
    else:
        text = (prefix + body + suffix).strip()

    return AnnotatedReply(text=text, confidence=level, similarity=similarity)


def disclosure_for_out_of_scope(capability_statement: str) -> str:
    return (
        "That may be outside what I'm set up to answer. "
        + capability_statement.strip()
    )
