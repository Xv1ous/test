"""NLP: intent-style routing + semantic retrieval over curated Q&A pairs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util

from voice_agent.config import QA_PATH


@dataclass
class NLPResult:
    reply_text: str
    similarity: float | None
    source_question: str | None


_model: SentenceTransformer | None = None
_cached_questions: list[str] = []
_cached_answers: list[str] = []
_cached_embeddings = None
_capability_statement: str = ""


def _embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_qa_corpus(path=None) -> None:
    global _cached_questions, _cached_answers, _cached_embeddings, _capability_statement
    path = path or QA_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _capability_statement = data.get("capability_statement", "").strip()
    pairs = data.get("pairs", [])
    _cached_questions = [p["q"].strip() for p in pairs]
    _cached_answers = [p["a"].strip() for p in pairs]

    if not _cached_questions:
        _cached_embeddings = None
        return

    emb = _embed_model()
    _cached_embeddings = emb.encode(
        _cached_questions,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def get_capability_statement() -> str:
    return _capability_statement


def _greeting_response(user_text: str) -> NLPResult | None:
    t = user_text.lower().strip()
    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", t):
        return NLPResult(
            reply_text="Hello. Ask me a question about university admissions, or say goodbye when you're done.",
            similarity=None,
            source_question=None,
        )
    if re.search(r"\b(thanks|thank you)\b", t):
        return NLPResult(
            reply_text="You're welcome. Let me know if you have another admissions question.",
            similarity=None,
            source_question=None,
        )
    if re.search(r"\b(bye|goodbye|see you)\b", t):
        return NLPResult(
            reply_text="Goodbye. Remember to verify important details with the official admissions office.",
            similarity=None,
            source_question=None,
        )
    return None


def respond(
    user_text: str,
    *,
    high_threshold: float,
    medium_threshold: float,
) -> NLPResult:
    if not _cached_questions or _cached_embeddings is None:
        load_qa_corpus()

    user_text = (user_text or "").strip()
    if not user_text:
        return NLPResult(
            reply_text="I didn't catch that. Please speak again or type your question.",
            similarity=None,
            source_question=None,
        )

    greet = _greeting_response(user_text)
    if greet:
        return greet

    emb = _embed_model()
    q_emb = emb.encode(user_text, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, _cached_embeddings)[0]
    best_idx = int(np.argmax(sims.cpu().numpy()))
    best_score = float(sims[best_idx].item())

    answer = _cached_answers[best_idx]
    source_q = _cached_questions[best_idx]

    if best_score < medium_threshold:
        return NLPResult(
            reply_text="",
            similarity=best_score,
            source_question=source_q,
        )

    return NLPResult(
        reply_text=answer,
        similarity=best_score,
        source_question=source_q,
    )
