"""
Client for Omnix TTS server.

Usage:
    1. Start Omnix:
        python app.py

    2. Install requirements:
        python -m pip install -r requirements_client.txt

    3. Run:
        python omnix_tts_client.py
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

import requests


BASE_URL = os.environ.get("OMNIX_URL", "http://127.0.0.1:5000")


def _decode_audio_payload(audio_value: str) -> bytes:
    """
    Supports both raw base64 and data URL format:
        UklGR...
        data:audio/wav;base64,UklGR...
    """
    if not audio_value:
        raise ValueError("Empty audio payload from server")

    match = re.match(r"^data:audio/[^;]+;base64,(.*)$", audio_value, flags=re.I | re.S)
    if match:
        audio_value = match.group(1)

    return base64.b64decode(audio_value)


def list_voices() -> list[dict[str, Any]]:
    url = f"{BASE_URL.rstrip('/')}/api/tts/speakers"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    speakers = data.get("speakers", [])

    print("\nProvider:", data.get("provider", "unknown"))
    print("Available voices:")

    if not speakers:
        print("  No speakers returned by server.")
        return []

    for item in speakers:
        speaker_id = item.get("id") or item.get("name") or str(item)
        speaker_name = item.get("name") or speaker_id
        print(f"  - {speaker_id} | {speaker_name}")

    return speakers


def speak(
    text: str,
    speaker: str = "default",
    language: str = "ru",
    out_file: str = "answer.wav",
    play_audio: bool = True,
) -> Path:
    url = f"{BASE_URL.rstrip('/')}/api/tts"

    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(data.get("error") or f"TTS request failed: {data}")

    audio_value = data.get("audio")
    audio_bytes = _decode_audio_payload(audio_value)

    out_path = Path(out_file)
    out_path.write_bytes(audio_bytes)

    print(f"Saved audio: {out_path.resolve()}")

    if play_audio and os.name == "nt":
        os.startfile(out_path)  # noqa: S606

    return out_path


def main() -> None:
    print(f"Omnix URL: {BASE_URL}")

    try:
        list_voices()
    except Exception as exc:
        print("\nCould not get speaker list from Omnix.")
        print("Check that Omnix is running and BASE_URL is correct.")
        print(f"Error: {exc}")
        return

    speaker = input("\nChoose speaker, for example default: ").strip() or "default"
    language = input("Language code, for example ru/en/zh [ru]: ").strip() or "ru"

    print("\nType text and press Enter.")
    print("To exit, type: q")

    counter = 1
    while True:
        text = input("\nText: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue

        out_file = f"answer_{counter:03d}.wav"
        counter += 1

        try:
            speak(
                text=text,
                speaker=speaker,
                language=language,
                out_file=out_file,
                play_audio=True,
            )
        except Exception as exc:
            print(f"TTS error: {exc}")


if __name__ == "__main__":
    main()
