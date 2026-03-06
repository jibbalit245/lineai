"""
Text-to-Speech engine with multi-provider support.

Supported backends:
  - gtts: Google Text-to-Speech (free, good quality)
  - pyttsx3: Offline system TTS (no API needed)
  - elevenlabs: ElevenLabs API (premium quality voices)

Produces a single narration audio file from the scene scripts.
"""

import logging
import os
from pathlib import Path

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


def synthesize_speech(ctx: PipelineContext) -> None:
    """Generate narration audio from script text."""
    backend = ctx.config.get("tts_backend", "gtts")
    output_path = ctx.output_dir / "narration.mp3"

    full_text = ctx.script_text
    if not full_text:
        full_text = " ".join(s["narration"] for s in ctx.scenes)

    if not full_text.strip():
        raise ValueError("No narration text to synthesize")

    if backend == "elevenlabs":
        _synthesize_elevenlabs(full_text, output_path, ctx.config)
    elif backend == "pyttsx3":
        _synthesize_pyttsx3(full_text, output_path, ctx.config)
    else:
        _synthesize_gtts(full_text, output_path, ctx.config)

    ctx.audio_narration_path = output_path

    # Compute duration
    from mutagen.mp3 import MP3
    audio_info = MP3(str(output_path))
    ctx.audio_duration = audio_info.info.length
    logger.info("Narration audio: %.1fs -> %s", ctx.audio_duration, output_path)


def _synthesize_gtts(text: str, output_path: Path, config: dict) -> None:
    from gtts import gTTS

    lang = config.get("tts_language", "en")
    slow = config.get("tts_slow", False)
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(str(output_path))


def _synthesize_pyttsx3(text: str, output_path: Path, config: dict) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    rate = config.get("tts_rate", 175)
    engine.setProperty("rate", rate)

    voices = engine.getProperty("voices")
    voice_index = config.get("tts_voice_index", 0)
    if voices and voice_index < len(voices):
        engine.setProperty("voice", voices[voice_index].id)

    engine.save_to_file(text, str(output_path))
    engine.runAndWait()


def _synthesize_elevenlabs(text: str, output_path: Path, config: dict) -> None:
    from elevenlabs import ElevenLabs

    api_key = config.get("elevenlabs_api_key") or os.environ.get("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)

    voice_id = config.get("elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")  # Rachel
    model_id = config.get("elevenlabs_model_id", "eleven_multilingual_v2")

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
    )

    with open(output_path, "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)
