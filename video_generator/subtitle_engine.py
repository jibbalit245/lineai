"""
Subtitle generation engine with word-level timing.

Uses OpenAI Whisper for precise word-level timestamps from the narration
audio, then generates styled .srt and .ass subtitle files.

Fallback: estimates timing from text length and audio duration when
Whisper is unavailable.
"""

import logging
import math
from pathlib import Path
from typing import Any

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


def generate_subtitles(ctx: PipelineContext) -> None:
    """Generate timed subtitles from narration audio."""
    if not ctx.audio_narration_path or not ctx.audio_narration_path.exists():
        raise FileNotFoundError("Narration audio not found — run TTS first")

    backend = ctx.config.get("subtitle_backend", "estimated")

    if backend == "whisper":
        segments = _transcribe_whisper(ctx.audio_narration_path, ctx.config)
    else:
        segments = _estimate_timing(ctx.scenes, ctx.audio_duration)

    ctx.subtitle_segments = segments

    # Write .srt file
    srt_path = ctx.output_dir / "subtitles.srt"
    _write_srt(segments, srt_path)
    ctx.subtitle_path = srt_path

    # Also write .ass file for styled subtitles
    ass_path = ctx.output_dir / "subtitles.ass"
    _write_ass(segments, ass_path, ctx.config)

    logger.info("Generated %d subtitle segments -> %s", len(segments), srt_path)


def _transcribe_whisper(
    audio_path: Path, config: dict
) -> list[dict[str, Any]]:
    """Use Whisper for word-level transcription."""
    import whisper

    model_name = config.get("whisper_model", "base")
    model = whisper.load_model(model_name)

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=config.get("language", "en"),
    )

    segments: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "words": seg.get("words", []),
        })

    return segments


def _estimate_timing(
    scenes: list[dict[str, Any]], total_duration: float
) -> list[dict[str, Any]]:
    """Estimate subtitle timing from scene data and total audio duration."""
    if not scenes:
        return []

    # Calculate total character count for proportional timing
    total_chars = sum(len(s.get("narration", "")) for s in scenes)
    if total_chars == 0:
        return []

    segments: list[dict[str, Any]] = []
    current_time = 0.0

    for scene in scenes:
        narration = scene.get("narration", "")
        if not narration:
            continue

        # Duration proportional to text length
        proportion = len(narration) / total_chars
        segment_duration = total_duration * proportion

        # Break into subtitle chunks (max ~10 words per chunk)
        words = narration.split()
        chunk_size = 10
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        chunk_duration = segment_duration / len(chunks) if chunks else segment_duration

        for chunk in chunks:
            segments.append({
                "start": round(current_time, 3),
                "end": round(current_time + chunk_duration, 3),
                "text": chunk,
            })
            current_time += chunk_duration

    return segments


def _write_srt(segments: list[dict[str, Any]], path: Path) -> None:
    """Write standard .srt subtitle file."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg["start"])
        end = _format_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ass(
    segments: list[dict[str, Any]], path: Path, config: dict
) -> None:
    """Write .ass subtitle file with styling for burned-in subtitles."""
    font_name = config.get("subtitle_font", "Arial")
    font_size = config.get("subtitle_font_size", 48)
    primary_color = config.get("subtitle_color", "&H00FFFFFF")
    outline_color = config.get("subtitle_outline_color", "&H00000000")
    shadow_color = config.get("subtitle_shadow_color", "&H80000000")

    header = f"""[Script Info]
Title: Auto-generated subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{shadow_color},-1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for seg in segments:
        start = _format_ass_time(seg["start"])
        end = _format_ass_time(seg["end"])
        text = seg["text"].replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
