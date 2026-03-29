"""Speech-to-text using OpenAI Whisper (local)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

WHISPER_SR = 16000


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    y = np.asarray(data, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.5:
        y = y / 32768.0
    return y


def _resample(y: np.ndarray, orig_sr: int, target_sr: int = WHISPER_SR) -> np.ndarray:
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float32)
    num = int(round(len(y) * float(target_sr) / orig_sr))
    if num <= 0:
        return y.astype(np.float32)
    y_out = signal.resample(y, num)
    return y_out.astype(np.float32)

_whisper_model: Any = None


def _load_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        import whisper

        _whisper_model = whisper.load_model(model_name)
        logger.info("Loaded Whisper model: %s", model_name)
    return _whisper_model


def transcribe_audio(
    audio: tuple[int, np.ndarray] | None,
    *,
    model_name: str = "base",
    language: str | None = "en",
) -> str:
    """
    Transcribe Gradio-style audio (sample_rate, int16/float waveform).

    Returns empty string if input is missing or silent.
    """
    if audio is None:
        return ""

    sample_rate, data = audio
    if data is None or len(data) == 0:
        return ""

    y = _to_mono_float32(np.asarray(data))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-6:
        return ""

    y = _resample(y, int(sample_rate), WHISPER_SR)

    model = _load_whisper(model_name)
    # fp16=False for CPU / broader compatibility
    kwargs: dict[str, Any] = {"fp16": False}
    if language:
        kwargs["language"] = language

    result = model.transcribe(y, **kwargs)
    text = (result.get("text") or "").strip()
    return text


def transcribe_text_input(text: str) -> str:
    """Bypass STT when user types instead of speaking (debug / accessibility)."""
    return (text or "").strip()


def transcribe_audio_path(
    path: str,
    *,
    model_name: str = "base",
    language: str | None = "en",
) -> str:
    """
    Transcribe an audio file (wav, mp3, webm, etc.) using Whisper.

    Requires ffmpeg on the system for many formats (Whisper dependency).
    """
    path = (path or "").strip()
    if not path:
        return ""

    model = _load_whisper(model_name)
    kwargs: dict[str, Any] = {"fp16": False}
    if language:
        kwargs["language"] = language

    result = model.transcribe(path, **kwargs)
    return (result.get("text") or "").strip()
