"""Configuration for the voice agent prototype."""

from pathlib import Path

# Project root (parent of voice_agent package)
ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "data" / "qa_pairs.json"

# STT: Whisper model size — tiny/base/small/medium (larger = better quality, slower)
WHISPER_MODEL = "base"

# NLP: minimum cosine similarity to trust a retrieved answer (0–1)
MATCH_HIGH = 0.55
MATCH_MEDIUM = 0.40

# Where to save synthesized speech for Gradio playback
TTS_OUTPUT_DIR = ROOT / "voice_agent" / "_tts_cache"
TTS_OUTPUT_WAV = TTS_OUTPUT_DIR / "last_reply.wav"
