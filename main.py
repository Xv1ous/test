"""
Voice admissions assistant — run from project root:

    python -m venv .venv
    .\\.venv\\Scripts\\pip install -r requirements.txt
    .\\.venv\\Scripts\\python main.py

First launch downloads SentenceTransformer and Whisper weights (can take several minutes).
For typed-only testing without the web UI: `python cli_demo.py "your question"`
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from voice_agent.nlp_module import get_capability_statement, load_qa_corpus
from voice_agent.pipeline import run_text_turn, run_voice_turn

LOG_FILE = Path(__file__).resolve().parent / "voice_agent" / "interaction_log.txt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_ui():
    load_qa_corpus()
    disclosure = get_capability_statement()

    def on_voice(audio):
        transcript, reply, audio_path, conf = run_voice_turn(audio, log_path=LOG_FILE)
        return transcript, reply, conf, audio_path

    def on_text(text):
        transcript, reply, audio_path, conf = run_text_turn(text, log_path=LOG_FILE)
        return transcript, reply, conf, audio_path

    with gr.Blocks(title="Voice Admissions Prototype") as demo:
        gr.Markdown(
            "## Voice-based conversational AI (prototype)\n\n"
            f"**Capability disclosure:** {disclosure}\n\n"
            "Record a question via microphone, or type below. The system uses **Whisper** (STT), "
            "**semantic retrieval** over a small Q&A set (NLP), and **pyttsx3** (TTS)."
        )
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Your voice")
            audio_out = gr.Audio(label="Agent voice", type="filepath")
        with gr.Row():
            btn_voice = gr.Button("Run voice turn")
        transcript = gr.Textbox(label="Transcript / input", lines=2)
        reply = gr.Textbox(label="Agent reply (with trust cues)", lines=6)
        conf = gr.Textbox(label="Confidence label", lines=1)
        text_in = gr.Textbox(label="Or type your question (skips STT)", lines=2)
        btn_text = gr.Button("Run text turn")

        btn_voice.click(on_voice, inputs=[audio_in], outputs=[transcript, reply, conf, audio_out])
        btn_text.click(on_text, inputs=[text_in], outputs=[transcript, reply, conf, audio_out])

    return demo


if __name__ == "__main__":
    logger.info("Starting Gradio server…")
    app = build_ui()
    app.launch()
