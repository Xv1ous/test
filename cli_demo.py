"""Text-only demo (no Gradio): prints reply and optional speaker output."""

from __future__ import annotations

import argparse

from voice_agent.pipeline import run_text_turn
from voice_agent.tts_module import speak_local


def main() -> None:
    p = argparse.ArgumentParser(description="Admissions assistant — typed input, STT skipped.")
    p.add_argument("question", nargs="*", help="Question (or pass via stdin)")
    p.add_argument("--speak", action="store_true", help="Also play TTS on local speakers")
    args = p.parse_args()
    text = " ".join(args.question).strip()
    if not text:
        text = input("Your question: ").strip()

    transcript, reply, audio_path, conf = run_text_turn(text)
    print("Confidence:", conf)
    print("Reply:\n", reply)
    if args.speak and reply:
        speak_local(reply)


if __name__ == "__main__":
    main()
