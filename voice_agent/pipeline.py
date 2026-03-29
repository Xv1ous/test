"""End-to-end orchestration: STT → NLP (+ interaction design) → TTS."""

from __future__ import annotations

from pathlib import Path

from voice_agent import interaction_design
from voice_agent.config import MATCH_HIGH, MATCH_MEDIUM, WHISPER_MODEL
from voice_agent.nlp_module import get_capability_statement, load_qa_corpus, respond
from voice_agent.stt_module import transcribe_audio, transcribe_text_input
from voice_agent.tts_module import speak_to_file


def _ensure_corpus() -> None:
    load_qa_corpus()


def run_voice_turn(
    audio,
    *,
    log_path: Path | None = None,
) -> tuple[str, str, str | None, str]:
    """
    Gradio-facing API: (audio) -> (transcript, agent_text, audio_path|None, confidence_label)
    """
    _ensure_corpus()
    transcript = transcribe_audio(audio, model_name=WHISPER_MODEL)
    agent_text, conf_label, sim = _text_to_reply(transcript)

    wav = speak_to_file(agent_text) if agent_text else ""
    audio_out = wav if wav and Path(wav).exists() else None

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"user_transcript={transcript!r}\nagent={agent_text!r}\n\n")

    return transcript, agent_text, audio_out, conf_label


def run_text_turn(
    text: str,
    *,
    log_path: Path | None = None,
) -> tuple[str, str, str | None, str]:
    """Same as voice turn but skip STT (typed input)."""
    _ensure_corpus()
    transcript = transcribe_text_input(text)
    agent_text, conf_label, sim = _text_to_reply(transcript)

    wav = speak_to_file(agent_text) if agent_text else ""
    audio_out = wav if wav and Path(wav).exists() else None

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"user_typed={transcript!r}\nagent={agent_text!r}\n\n")

    return transcript, agent_text, audio_out, conf_label


def _text_to_reply(user_text: str) -> tuple[str, str, str, float | None]:
    cap = get_capability_statement()
    nlp: NLPResult = respond(
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
