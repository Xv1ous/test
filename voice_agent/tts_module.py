"""Text-to-speech: gTTS (default, server/cloud) or pyttsx3 (local desktop)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from voice_agent.config import (
    GTTS_LANG,
    TTS_ENGINE,
    TTS_OUTPUT_AUDIO,
    TTS_OUTPUT_DIR,
    TTS_OUTPUT_WAV,
)

logger = logging.getLogger(__name__)

_engine = None


def _pyttsx3_engine():
    global _engine
    if _engine is None:
        import pyttsx3

        _engine = pyttsx3.init()
        try:
            _engine.setProperty("rate", 175)
        except Exception:
            pass
    return _engine


def speak_to_file(text: str, out_path=None) -> str:
    """
    Synthesize `text` to an audio file and return the path.

    Default engine is **gTTS** → MP3 (works headless; needs outbound HTTPS).
    Set ``TTS_ENGINE=pyttsx3`` for offline WAV on a machine with SAPI/voices.
    """
    text = (text or "").strip()
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not text:
        return ""

    if TTS_ENGINE == "pyttsx3":
        return _speak_pyttsx3_to_wav(text, out_path)

    return _speak_gtts_to_mp3(text, out_path)


def _speak_gtts_to_mp3(text: str, out_path) -> str:
    from gtts import gTTS

    path = Path(out_path) if out_path else TTS_OUTPUT_AUDIO
    path = Path(path)
    if path.suffix.lower() != ".mp3":
        path = path.with_suffix(".mp3")

    tts = gTTS(text=text, lang=GTTS_LANG)
    tts.save(str(path))
    return str(path)


def _speak_pyttsx3_to_wav(text: str, out_path) -> str:
    path = Path(out_path) if out_path else TTS_OUTPUT_WAV
    path = Path(path)
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")

    engine = _pyttsx3_engine()
    try:
        engine.save_to_file(text, str(path))
        engine.runAndWait()
    except Exception as e:
        logger.warning("pyttsx3 save_to_file failed (%s); trying temp wav", e)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        return tmp

    return str(path)


def speak_local(text: str) -> None:
    """Play on local speakers (desktop only). Uses pyttsx3; may fail headless."""
    text = (text or "").strip()
    if not text:
        return
    try:
        engine = _pyttsx3_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.warning("speak_local unavailable (%s). Install/use a desktop audio stack.", e)
        raise
