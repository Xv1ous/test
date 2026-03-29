"""Configuration for the voice agent prototype."""

import os
from pathlib import Path

# Project root (parent of voice_agent package)
ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "data" / "qa_pairs.json"

# STT: Whisper model size — tiny/base/small/medium (larger = better quality, slower)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base").strip() or "base"

# NLP: minimum cosine similarity to trust a retrieved answer (0–1)
MATCH_HIGH = 0.55
MATCH_MEDIUM = 0.40

# TTS: "gtts" (default, server-safe, writes MP3) or "pyttsx3" (local Windows/SAPI only)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "gtts").strip().lower()

# gTTS language code (ISO 639-1)
GTTS_LANG = os.environ.get("GTTS_LANG", "en").strip() or "en"

# Where to save synthesized speech
TTS_OUTPUT_DIR = ROOT / "voice_agent" / "_tts_cache"
TTS_OUTPUT_AUDIO = TTS_OUTPUT_DIR / "last_reply.mp3"
TTS_OUTPUT_WAV = TTS_OUTPUT_DIR / "last_reply.wav"
