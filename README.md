# Voice admissions agent (research prototype)

Voice-oriented conversational assistant for **low-stakes university admissions Q&A**. It chains **Speech-to-Text (Whisper)**, **semantic retrieval** over a small curated JSON knowledge base (sentence embeddings), and **Text-to-Speech** (default: **gTTS** → MP3 for headless servers; optional **pyttsx3** for offline desktop).

Interaction-design cues (confidence / uncertainty / capability disclosure) are applied in text before TTS.

## Repository naming

The code was initially pushed to a repo named `test`. For coursework or a portfolio, consider renaming the GitHub repository to something like **`voice-admissions-agent`** (GitHub: *Settings → General → Repository name*), then update your local remote:

```bash
git remote set-url origin https://github.com/<you>/voice-admissions-agent.git
```

## Requirements

- **Python 3.10+**
- **ffmpeg** on `PATH` (used by Whisper for many audio formats; install from [ffmpeg.org](https://ffmpeg.org/) or your OS package manager)
- Outbound **HTTPS** for **gTTS** (Google’s TTS endpoint) and first-time **Hugging Face** model downloads
- Enough disk/RAM for **PyTorch** + **Whisper** (use `WHISPER_MODEL=tiny` on weak CPUs)

## Setup

```powershell
cd path\to\THESIS
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

You may see a **pip resolver warning** about `click` versions (`gTTS` vs `typer`); the stack still runs in practice. If install fails, upgrade `gTTS` or pin `click` per the error message.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `TTS_ENGINE` | `gtts` | `gtts` = MP3 via Google (server-safe). `pyttsx3` = local WAV (Windows SAPI; not for headless cloud). |
| `GTTS_LANG` | `en` | gTTS language code. |
| `WHISPER_MODEL` | `base` | Whisper size: `tiny` is faster for weak CPUs / small hosts. |

## Run: Gradio UI (local demo)

```powershell
.\.venv\Scripts\python main.py
```

Open the URL shown in the terminal; use the microphone or type a question.

## Run: CLI (typed input only)

```powershell
.\.venv\Scripts\python cli_demo.py "When is the application deadline?"
.\.venv\Scripts\python cli_demo.py --speak "Hello"   # local speakers via pyttsx3
```

## Run: HTTP API (Flutter / mobile)

Start **FastAPI** with **Uvicorn**:

```powershell
.\.venv\Scripts\uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check |
| GET | `/v1/meta` | Capability disclosure text |
| POST | `/v1/chat/text` | JSON body: `{"message": "...", "include_audio": false}` |
| POST | `/v1/chat/voice` | Multipart file field `audio` (wav/webm/mp3/…); optional query `include_audio=true` |

Responses use JSON (`TextChatResponse`). When `include_audio` is true, the reply includes `audio_base64` and `audio_mime` (`audio/mpeg` for gTTS).

**CORS** is set to allow all origins for prototyping; tighten this before production.

## Deploying (e.g. Render)

1. Set build command: `pip install -r requirements.txt`
2. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
3. Use **`TTS_ENGINE=gtts`** (default). Do **not** use `pyttsx3` on a headless dyno.
4. Whisper + PyTorch are heavy; a free tier may be slow or hit memory limits—consider `WHISPER_MODEL=tiny` in code or env-driven config for demos.

## Project layout

- `data/qa_pairs.json` — Q&A pairs and capability statement (extend toward 50–100 verified pairs).
- `voice_agent/config.py` — paths, Whisper size, similarity thresholds, TTS engine.
- `voice_agent/stt_module.py` — Whisper (numpy for Gradio; file path for API uploads).
- `voice_agent/nlp_module.py` — Embeddings + cosine match + light intent phrases.
- `voice_agent/interaction_design.py` — Trust / uncertainty wording.
- `voice_agent/tts_module.py` — gTTS (default) or pyttsx3.
- `voice_agent/pipeline.py` — Orchestration; returns `TurnResult` with `ok` / `error`.
- `api.py` — FastAPI app for mobile clients.
- `main.py` — Gradio UI.

## Flutter integration (outline)

1. Call **`POST /v1/chat/text`** with the user’s typed text, or **`POST /v1/chat/voice`** with recorded bytes.
2. Decode **`audio_base64`** when present and play with an audio widget.
3. Point the app at your deployed base URL (HTTPS).

## License / academic use

Use and cite appropriately for your thesis or course; verify admissions content with your institution’s official sources.
