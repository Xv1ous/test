"""Text-only demo (no Gradio): prints reply and optional speaker output."""

from __future__ import annotations

import argparse
import logging
import sys

from voice_agent.pipeline import run_text_turn
from voice_agent.tts_module import speak_local

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Admissions assistant — typed input, STT skipped.")
    p.add_argument("question", nargs="*", help="Question (or pass via stdin)")
    p.add_argument("--speak", action="store_true", help="Also play TTS on local speakers (pyttsx3)")
    args = p.parse_args()
    text = " ".join(args.question).strip()
    if not text:
        text = input("Your question: ").strip()

    try:
        r = run_text_turn(text)
    except Exception as e:
        logger.exception("Pipeline raised unexpectedly")
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

    if not r.ok:
        print("Error:", r.error or "Unknown failure", file=sys.stderr)
        sys.exit(1)

    print("Confidence:", r.confidence)
    if r.similarity is not None:
        print("Similarity:", round(r.similarity, 4))
    print("Reply:\n", r.agent_text)

    if args.speak and r.agent_text:
        try:
            speak_local(r.agent_text)
        except Exception as e:
            print("TTS playback failed:", e, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
