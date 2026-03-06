"""
Subtitle generation engine — word-level precision with style.

Uses OpenAI Whisper for precise word-level timestamps, then generates:
  - Standard .srt subtitles
  - Styled .ass subtitles with karaoke word-by-word highlighting
  - Color-timed word reveals (each word lights up as it's spoken)

Fallback: estimates timing from text length and audio duration.
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
    karaoke = ctx.config.get("subtitle_karaoke", True)

    if backend == "whisper":
        segments = _transcribe_whisper(ctx.audio_narration_path, ctx.config)
    else:
        segments = _estimate_timing(ctx.scenes, ctx.audio_duration)

    ctx.subtitle_segments = segments

    # Write .srt
    srt_path = ctx.output_dir / "subtitles.srt"
    _write_srt(segments, srt_path)
    ctx.subtitle_path = srt_path

    # Write .ass with optional karaoke
    ass_path = ctx.output_dir / "subtitles.ass"
    if karaoke and any(seg.get("words") for seg in segments):
        _write_ass_karaoke(segments, ass_path, ctx.config)
    else:
        _write_ass(segments, ass_path, ctx.config)

    logger.info("Generated %d subtitle segments -> %s", len(segments), srt_path)


# ---------------------------------------------------------------------------
# Whisper Transcription
# ---------------------------------------------------------------------------

def _transcribe_whisper(audio_path: Path, config: dict) -> list[dict[str, Any]]:
    """Use Whisper for word-level transcription with retry."""
    import whisper

    model_name = config.get("whisper_model", "base")

    logger.info("Loading Whisper model '%s'...", model_name)
    model = whisper.load_model(model_name)

    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=config.get("language", "en"),
    )

    segments: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": w.get("start", 0),
                "end": w.get("end", 0),
            })

        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "words": words,
        })

    logger.info("Whisper transcribed %d segments with word-level timing", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Estimated Timing (Fallback)
# ---------------------------------------------------------------------------

def _estimate_timing(scenes: list[dict[str, Any]], total_duration: float) -> list[dict[str, Any]]:
    """Estimate subtitle timing from scene data and total audio duration."""
    if not scenes:
        return []

    total_chars = sum(len(s.get("narration", "")) for s in scenes)
    if total_chars == 0:
        return []

    segments: list[dict[str, Any]] = []
    current_time = 0.0

    for scene in scenes:
        narration = scene.get("narration", "")
        if not narration:
            continue

        proportion = len(narration) / total_chars
        segment_duration = total_duration * proportion

        # Break into subtitle chunks (max ~8 words per chunk for readability)
        words = narration.split()
        chunk_size = 8
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

        chunk_duration = segment_duration / len(chunks) if chunks else segment_duration

        for chunk_words in chunks:
            chunk_text = " ".join(chunk_words)

            # Estimate per-word timing within chunk
            word_data = []
            words_in_chunk = len(chunk_words)
            word_dur = chunk_duration / words_in_chunk if words_in_chunk else chunk_duration
            word_time = current_time

            for w in chunk_words:
                word_data.append({
                    "word": w,
                    "start": round(word_time, 3),
                    "end": round(word_time + word_dur, 3),
                })
                word_time += word_dur

            segments.append({
                "start": round(current_time, 3),
                "end": round(current_time + chunk_duration, 3),
                "text": chunk_text,
                "words": word_data,
            })
            current_time += chunk_duration

    return segments


# ---------------------------------------------------------------------------
# .srt Writer
# ---------------------------------------------------------------------------

def _write_srt(segments: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt(seg['start'])} --> {_fmt_srt(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# .ass Writer (Standard)
# ---------------------------------------------------------------------------

def _write_ass(segments: list[dict[str, Any]], path: Path, config: dict) -> None:
    """Write styled .ass subtitle file."""
    header = _ass_header(config)

    events: list[str] = []
    for seg in segments:
        start = _fmt_ass(seg["start"])
        end = _fmt_ass(seg["end"])
        text = seg["text"].replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# .ass Writer (Karaoke Style)
# ---------------------------------------------------------------------------

def _write_ass_karaoke(segments: list[dict[str, Any]], path: Path, config: dict) -> None:
    """
    Write .ass with karaoke word-by-word highlighting.

    Each word lights up (changes color) at its exact timestamp using ASS
    override tags: {\\kf<duration>} for smooth fill and {\\k<duration>} for
    instant highlight.
    """
    header = _ass_header(config, karaoke=True)

    highlight_color = config.get("subtitle_highlight_color", "&H0000FFFF")  # Yellow
    style = config.get("subtitle_karaoke_style", "smooth")  # smooth | instant | glow

    events: list[str] = []
    for seg in segments:
        start = _fmt_ass(seg["start"])
        end = _fmt_ass(seg["end"])

        words = seg.get("words", [])
        if not words:
            text = seg["text"].replace("\n", "\\N")
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
            continue

        # Build karaoke line
        karaoke_parts = []
        for w in words:
            # Duration in centiseconds (ASS karaoke unit)
            dur_cs = max(1, int((w["end"] - w["start"]) * 100))

            if style == "smooth":
                tag = f"\\kf{dur_cs}"
            elif style == "glow":
                # Glow: highlight + slight border expansion
                tag = f"\\kf{dur_cs}\\bord4"
            else:
                tag = f"\\k{dur_cs}"

            karaoke_parts.append(f"{{{tag}}}{w['word']}")

        karaoke_text = " ".join(karaoke_parts)

        # Add color override for the highlight
        prefix = f"{{\\1c{highlight_color}}}"
        events.append(f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{prefix}{karaoke_text}")

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    logger.info("Wrote karaoke-style .ass subtitles with %s highlighting", style)


# ---------------------------------------------------------------------------
# .ass Header
# ---------------------------------------------------------------------------

def _ass_header(config: dict, karaoke: bool = False) -> str:
    font_name = config.get("subtitle_font", "Arial")
    font_size = config.get("subtitle_font_size", 48)
    primary = config.get("subtitle_color", "&H00FFFFFF")
    outline = config.get("subtitle_outline_color", "&H00000000")
    shadow = config.get("subtitle_shadow_color", "&H80000000")
    highlight = config.get("subtitle_highlight_color", "&H0000FFFF")

    header = f"""[Script Info]
Title: LineAI Auto-Generated Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},&H000000FF,{outline},{shadow},-1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1
"""

    if karaoke:
        # Karaoke style: starts dimmed, fills to highlight color
        karaoke_size = int(font_size * 1.1)
        header += f"Style: Karaoke,{font_name},{karaoke_size},&H80FFFFFF,{highlight},{outline},{shadow},-1,0,0,0,100,100,0,0,1,3,2,2,40,40,60,1\n"

    header += """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


# ---------------------------------------------------------------------------
# Time Formatters
# ---------------------------------------------------------------------------

def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
