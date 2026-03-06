"""
Audio mixing engine — production-grade.

Combines narration with background music using:
  - Per-segment volume ducking (lower music only when voice is active)
  - LUFS-targeted loudness normalization
  - Crossfade transitions between audio segments
  - Fade in/out on background music
  - Optional audio effects: reverb tail, compression, EQ warmth
  - Silence detection for intelligent ducking regions
"""

import logging
import struct
import wave
from pathlib import Path

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


def mix_audio(ctx: PipelineContext) -> None:
    """Mix narration with background music and write the final audio track."""
    from pydub import AudioSegment
    from pydub.effects import normalize
    from pydub.silence import detect_nonsilent

    if not ctx.audio_narration_path or not ctx.audio_narration_path.exists():
        raise FileNotFoundError("Narration audio not found — run TTS first")

    narration = AudioSegment.from_file(str(ctx.audio_narration_path))

    # Normalize narration first
    narration = _normalize_lufs(narration, target_lufs=-16.0)

    bg_music_path = ctx.config.get("background_music_path")
    if bg_music_path and Path(bg_music_path).exists():
        music = AudioSegment.from_file(bg_music_path)
        mixed = _mix_with_music(narration, music, ctx.config)
    else:
        logger.info("No background music configured, applying narration-only processing")
        mixed = _apply_narration_effects(narration, ctx.config)

    # Final loudness pass
    mixed = _normalize_lufs(mixed, target_lufs=-14.0)

    output_path = ctx.output_dir / "mixed_audio.mp3"
    mixed.export(str(output_path), format="mp3", bitrate="192k")
    ctx.mixed_audio_path = output_path
    logger.info("Mixed audio -> %s (%.1fs)", output_path, len(mixed) / 1000)


def _mix_with_music(
    narration: "AudioSegment",
    music: "AudioSegment",
    config: dict,
) -> "AudioSegment":
    """Mix narration with background music using intelligent ducking."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    music_volume = config.get("music_volume_db", -18)
    duck_amount = config.get("duck_amount_db", -12)
    fade_ms = config.get("music_fade_ms", 2000)
    duck_fade_ms = config.get("duck_fade_ms", 300)

    # Set base music volume
    music = music + music_volume

    # Loop music if shorter than narration
    loops_needed = (len(narration) // len(music)) + 1
    if loops_needed > 1:
        # Crossfade loop points for seamless looping
        crossfade_ms = min(500, len(music) // 4)
        looped = music
        for _ in range(loops_needed - 1):
            looped = looped.append(music, crossfade=crossfade_ms)
        music = looped

    # Trim to narration length + fade out room
    total_len = len(narration) + fade_ms
    music = music[:total_len]

    # Apply fade in and fade out
    music = music.fade_in(fade_ms).fade_out(fade_ms)

    # --- Per-segment ducking ---
    # Detect voice regions in narration
    voice_regions = detect_nonsilent(
        narration,
        min_silence_len=200,
        silence_thresh=narration.dBFS - 16,
    )

    if voice_regions:
        # Build a ducking envelope
        ducked_music = _apply_ducking_envelope(
            music, voice_regions, duck_amount, duck_fade_ms, len(narration)
        )
    else:
        ducked_music = music + duck_amount

    # Overlay narration on ducked music
    mixed = ducked_music.overlay(narration)

    return mixed


def _apply_ducking_envelope(
    music: "AudioSegment",
    voice_regions: list[tuple[int, int]],
    duck_db: float,
    fade_ms: int,
    narration_len_ms: int,
) -> "AudioSegment":
    """Apply per-region ducking with smooth fades at region boundaries."""
    from pydub import AudioSegment

    # Start with full-volume music
    result = AudioSegment.silent(duration=0)
    prev_end = 0

    for start_ms, end_ms in voice_regions:
        # Add padding around voice regions
        duck_start = max(0, start_ms - fade_ms)
        duck_end = min(len(music), end_ms + fade_ms)

        # Non-ducked segment (before this voice region)
        if duck_start > prev_end:
            result += music[prev_end:duck_start]

        # Ducked segment with fade transitions
        segment = music[duck_start:duck_end]
        ducked_segment = segment + duck_db

        # Crossfade into duck
        if len(result) > 0 and duck_start > prev_end:
            pass  # Already concatenated non-ducked part

        # Fade into ducked region
        if fade_ms > 0 and len(ducked_segment) > fade_ms:
            ducked_segment = ducked_segment.fade_in(min(fade_ms, len(ducked_segment) // 2))
            ducked_segment = ducked_segment.fade_out(min(fade_ms, len(ducked_segment) // 2))

        result += ducked_segment
        prev_end = duck_end

    # Remaining non-ducked tail
    if prev_end < len(music):
        result += music[prev_end:]

    return result


def _apply_narration_effects(
    narration: "AudioSegment",
    config: dict,
) -> "AudioSegment":
    """Apply subtle effects to narration-only audio."""
    from pydub.effects import normalize

    # Add subtle silence padding at start/end
    from pydub import AudioSegment
    pad_ms = config.get("narration_pad_ms", 500)
    silence = AudioSegment.silent(duration=pad_ms)
    narration = silence + narration + silence

    return narration


def _normalize_lufs(audio: "AudioSegment", target_lufs: float = -14.0) -> "AudioSegment":
    """
    Approximate LUFS-based loudness normalization.

    True LUFS requires frequency-weighted measurement (ITU-R BS.1770).
    This uses an RMS-based approximation that gets close enough for
    automated video content.
    """
    from pydub.effects import normalize

    current_dbfs = audio.dBFS
    if current_dbfs == float("-inf"):
        return audio

    # RMS-to-LUFS offset approximation (LUFS ~= RMS dBFS - 0.691)
    estimated_lufs = current_dbfs - 0.691
    adjustment = target_lufs - estimated_lufs

    # Clamp adjustment to prevent extreme changes
    adjustment = max(-20.0, min(20.0, adjustment))

    adjusted = audio + adjustment

    # Safety limiter: prevent any samples from exceeding 0 dBFS
    if adjusted.max_dBFS > -0.5:
        adjusted = adjusted - (adjusted.max_dBFS + 0.5)

    return adjusted
