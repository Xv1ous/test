"""
FastAPI backend for Flutter / mobile clients.

Run (from project root, after installing requirements):

    uvicorn api:app --host 0.0.0.0 --port 8000

Environment:
    TTS_ENGINE=gtts   (default; headless-safe)
    TTS_ENGINE=pyttsx3  (local desktop only)
"""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voice_agent.nlp_module import get_capability_statement, load_qa_corpus
from voice_agent.pipeline import run_text_turn, run_voice_turn_file

logger = logging.getLogger(__name__)

LOG_FILE = Path(__file__).resolve().parent / "voice_agent" / "interaction_log.txt"

app = FastAPI(
    title="Voice Admissions Agent API",
    description="STT (Whisper) → semantic Q&A → TTS (gTTS).",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message (typed)")
    include_audio: bool = Field(False, description="If true, include base64-encoded reply audio")


class TextChatResponse(BaseModel):
    ok: bool
    transcript: str
    reply: str
    confidence: str
    similarity: float | None = None
    audio_base64: str | None = None
    audio_mime: str | None = None
    error: str | None = None
    capability_statement: str | None = None


@app.on_event("startup")
def _startup() -> None:
    load_qa_corpus()
    logger.info("API startup: corpus loaded.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/meta")
def meta() -> dict:
    return {"capability_statement": get_capability_statement()}


@app.post("/v1/chat/text", response_model=TextChatResponse)
def chat_text(body: TextChatRequest) -> TextChatResponse:
    r = run_text_turn(body.message.strip(), log_path=LOG_FILE)
    if not r.ok:
        return TextChatResponse(
            ok=False,
            transcript="",
            reply="",
            confidence="",
            error=r.error or "Unknown error",
            capability_statement=get_capability_statement(),
        )

    audio_b64 = None
    mime = None
    if body.include_audio and r.audio_path and Path(r.audio_path).exists():
        data = Path(r.audio_path).read_bytes()
        audio_b64 = base64.standard_b64encode(data).decode("ascii")
        mime = "audio/mpeg" if r.audio_path.lower().endswith(".mp3") else "audio/wav"

    return TextChatResponse(
        ok=True,
        transcript=r.transcript,
        reply=r.agent_text,
        confidence=r.confidence,
        similarity=r.similarity,
        audio_base64=audio_b64,
        audio_mime=mime,
    )


@app.post("/v1/chat/voice", response_model=TextChatResponse)
async def chat_voice(
    audio: UploadFile = File(..., description="User recording (wav, webm, mp3, …)"),
    include_audio: bool = Query(True, description="Include base64 reply audio"),
) -> TextChatResponse:
    suffix = Path(audio.filename or "upload").suffix or ".webm"
    if suffix.lower() not in {".wav", ".webm", ".mp3", ".m4a", ".ogg", ".flac", ".mp4"}:
        suffix = ".webm"

    try:
        raw = await audio.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty audio upload")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            r = run_voice_turn_file(tmp_path, log_path=LOG_FILE)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat_voice failed")
        return TextChatResponse(
            ok=False,
            transcript="",
            reply="",
            confidence="",
            error=str(e),
            capability_statement=get_capability_statement(),
        )

    if not r.ok:
        return TextChatResponse(
            ok=False,
            transcript="",
            reply="",
            confidence="",
            error=r.error or "Unknown error",
            capability_statement=get_capability_statement(),
        )

    audio_b64 = None
    mime = None
    if include_audio and r.audio_path and Path(r.audio_path).exists():
        data = Path(r.audio_path).read_bytes()
        audio_b64 = base64.standard_b64encode(data).decode("ascii")
        mime = "audio/mpeg" if r.audio_path.lower().endswith(".mp3") else "audio/wav"

    return TextChatResponse(
        ok=True,
        transcript=r.transcript,
        reply=r.agent_text,
        confidence=r.confidence,
        similarity=r.similarity,
        audio_base64=audio_b64,
        audio_mime=mime,
    )
