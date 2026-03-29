"""Text-to-speech using pyttsx3 (offline on Windows)."""

from __future__ import annotations

import logging
import os
import tempfile

import pyttsx3

from voice_agent.config import TTS_OUTPUT_DIR, TTS_OUTPUT_WAV

logger = logging.getLogger(__name__)

_engine: pyttsx3.Engine | None = None


def _engine_singleton() -> pyttsx3.Engine:
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        try:
            _engine.setProperty("rate", 175)
        except Exception:
            pass
    return _engine


def speak_to_file(text: str, out_path=None) -> str:
    """
    Synthesize `text` to a WAV file and return the path.
    Gradio Audio can play the returned file.
    """
    text = (text or "").strip()
    out_path = out_path or TTS_OUTPUT_WAV
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not text:
        return ""

    engine = _engine_singleton()
    try:
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
    except Exception as e:
        logger.warning("pyttsx3 save_to_file failed (%s); trying temp wav", e)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        return tmp

    return str(out_path)


def speak_local(text: str) -> None:
    """Play audio immediately on the machine speakers (CLI / debugging)."""
    engine = _engine_singleton()
    engine.say((text or "").strip())
    engine.runAndWait()
