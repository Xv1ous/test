"""End-to-end orchestration: STT → NLP (+ interaction design) → TTS."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from voice_agent import interaction_design
from voice_agent.config import MATCH_HIGH, MATCH_MEDIUM, WHISPER_MODEL
from voice_agent.nlp_module import get_capability_statement, load_qa_corpus, respond
from voice_agent.stt_module import (
    transcribe_audio,
    transcribe_audio_path,
    transcribe_text_input,
)
from voice_agent.tts_module import speak_to_file

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """One user→agent turn. ``ok`` is False when the pipeline raises."""

    ok: bool
    transcript: str = ""
    agent_text: str = ""
    audio_path: str | None = None
    confidence: str = ""
    similarity: float | None = None
    error: str | None = None


def _ensure_corpus() -> None:
    load_qa_corpus()


def run_voice_turn(
    audio,
    *,
    log_path: Path | None = None,
) -> TurnResult:
    """Gradio / in-process numpy audio → transcript + reply + optional audio file."""
    try:
        _ensure_corpus()
        transcript = transcribe_audio(audio, model_name=WHISPER_MODEL)
        return _finish_turn(transcript, log_path=log_path, source="user_transcript")
    except Exception as e:
        logger.exception("run_voice_turn failed")
        return TurnResult(ok=False, error=str(e))


def run_voice_turn_file(
    file_path: str,
    *,
    log_path: Path | None = None,
) -> TurnResult:
    """Uploaded audio file on disk (API) → same as voice turn."""
    try:
        _ensure_corpus()
        transcript = transcribe_audio_path(file_path, model_name=WHISPER_MODEL)
        return _finish_turn(transcript, log_path=log_path, source="user_transcript_file")
    except Exception as e:
        logger.exception("run_voice_turn_file failed")
        return TurnResult(ok=False, error=str(e))


def run_text_turn(
    text: str,
    *,
    log_path: Path | None = None,
) -> TurnResult:
    """Typed input; skips STT."""
    try:
        _ensure_corpus()
        transcript = transcribe_text_input(text)
        return _finish_turn(transcript, log_path=log_path, source="user_typed")
    except Exception as e:
        logger.exception("run_text_turn failed")
        return TurnResult(ok=False, error=str(e))


def _finish_turn(
    transcript: str,
    *,
    log_path: Path | None,
    source: str,
) -> TurnResult:
    agent_text, conf_label, sim = _text_to_reply(transcript)
    out = speak_to_file(agent_text) if agent_text else ""
    audio_out = out if out and Path(out).exists() else None

    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{source}={transcript!r}\nagent={agent_text!r}\n\n")
        except OSError as e:
            logger.warning("Could not write log: %s", e)

    return TurnResult(
        ok=True,
        transcript=transcript,
        agent_text=agent_text,
        audio_path=audio_out,
        confidence=conf_label,
        similarity=sim,
    )


def _text_to_reply(user_text: str) -> tuple[str, str, float | None]:
    cap = get_capability_statement()
    nlp = respond(
        user_text,
        high_threshold=MATCH_HIGH,
        medium_threshold=MATCH_MEDIUM,
    )

    if nlp.similarity is not None and nlp.reply_text == "":
        text = interaction_design.disclosure_for_out_of_scope(cap)
        return text, interaction_design.ConfidenceLevel.LOW.value, nlp.similarity

    if nlp.similarity is None:
        return nlp.reply_text, interaction_design.ConfidenceLevel.NONE.value, None

    ann = interaction_design.wrap_answer(
        nlp.reply_text,
        nlp.similarity,
        high_threshold=MATCH_HIGH,
        medium_threshold=MATCH_MEDIUM,
    )
    return ann.text, ann.confidence.value, nlp.similarity
