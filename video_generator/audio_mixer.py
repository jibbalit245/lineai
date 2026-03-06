"""
Audio mixing engine.

Combines narration audio with background music, applying:
  - Volume ducking (lower music when narration is playing)
  - Fade in/out on background music
  - Normalization to prevent clipping
"""

import logging
from pathlib import Path

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


def mix_audio(ctx: PipelineContext) -> None:
    """Mix narration with background music and write the final audio track."""
    from pydub import AudioSegment
    from pydub.effects import normalize

    if not ctx.audio_narration_path or not ctx.audio_narration_path.exists():
        raise FileNotFoundError("Narration audio not found — run TTS first")

    narration = AudioSegment.from_file(str(ctx.audio_narration_path))

    bg_music_path = ctx.config.get("background_music_path")
    if bg_music_path and Path(bg_music_path).exists():
        music = AudioSegment.from_file(bg_music_path)
        music_volume = ctx.config.get("music_volume_db", -18)
        duck_amount = ctx.config.get("duck_amount_db", -12)
        fade_duration_ms = ctx.config.get("music_fade_ms", 2000)

        # Adjust music volume
        music = music + music_volume

        # Loop music if shorter than narration
        while len(music) < len(narration):
            music = music + music

        # Trim to narration length + fade out room
        music = music[: len(narration) + fade_duration_ms]

        # Apply fade in and fade out
        music = music.fade_in(fade_duration_ms).fade_out(fade_duration_ms)

        # Duck the music under the narration
        # Simple approach: lower music volume for the entire narration span
        ducked_music = music + duck_amount
        # Overlay: narration on top of ducked music
        mixed = ducked_music.overlay(narration)

        # Normalize to prevent clipping
        mixed = normalize(mixed)
    else:
        logger.info("No background music configured, using narration only")
        mixed = normalize(narration)

    output_path = ctx.output_dir / "mixed_audio.mp3"
    mixed.export(str(output_path), format="mp3", bitrate="192k")
    ctx.mixed_audio_path = output_path
    logger.info("Mixed audio -> %s (%.1fs)", output_path, len(mixed) / 1000)
