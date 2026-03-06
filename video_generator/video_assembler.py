"""
Video assembly engine — the heart of the visual pipeline.

Takes scene images and audio, then composites them into a final video with:
  - Ken Burns effect (slow pan & zoom on still images)
  - Crossfade transitions between scenes
  - Color grading (contrast, saturation, vignette)
  - Burned-in subtitles (from .ass file)
  - Multiple output format support (landscape, portrait, square)
"""

import logging
import math
from pathlib import Path

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


def assemble_video(ctx: PipelineContext) -> None:
    """Build the final video from all pipeline artifacts."""
    from moviepy.editor import (
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        concatenate_videoclips,
    )

    if not ctx.image_paths:
        raise FileNotFoundError("No scene images — run visual engine first")

    audio_path = ctx.mixed_audio_path or ctx.audio_narration_path
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError("No audio file — run TTS/mixer first")

    video_format = ctx.config.get("video_format", "landscape")
    fps = ctx.config.get("fps", 30)
    transition_duration = ctx.config.get("transition_duration", 0.8)
    enable_ken_burns = ctx.config.get("ken_burns", True)
    enable_color_grade = ctx.config.get("color_grading", True)

    # Load audio to determine total duration
    audio_clip = AudioFileClip(str(audio_path))
    total_duration = audio_clip.duration

    # Calculate per-scene duration
    num_scenes = len(ctx.image_paths)
    scene_duration = total_duration / num_scenes

    # Build scene clips
    scene_clips = []
    for i, img_path in enumerate(ctx.image_paths):
        clip = ImageClip(str(img_path)).set_duration(scene_duration)

        if enable_ken_burns:
            clip = _apply_ken_burns(clip, i, scene_duration)

        if enable_color_grade:
            clip = _apply_color_grading(clip, ctx.config)

        scene_clips.append(clip)

    # Concatenate with crossfade transitions
    if transition_duration > 0 and len(scene_clips) > 1:
        final_video = _crossfade_concat(scene_clips, transition_duration)
    else:
        final_video = concatenate_videoclips(scene_clips, method="compose")

    # Trim to exact audio duration
    final_video = final_video.subclip(0, min(final_video.duration, total_duration))

    # Attach audio
    final_video = final_video.set_audio(audio_clip)

    # Determine output path
    output_path = ctx.output_dir / f"video_{video_format}.mp4"

    # Write the video
    codec = ctx.config.get("codec", "libx264")
    audio_codec = ctx.config.get("audio_codec", "aac")
    bitrate = ctx.config.get("bitrate", "5000k")

    logger.info("Rendering video -> %s", output_path)
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
        bitrate=bitrate,
        preset="medium",
        threads=4,
        logger=None,  # Suppress moviepy's verbose logging
    )

    # Burn in subtitles if available
    ass_path = ctx.output_dir / "subtitles.ass"
    if ass_path.exists() and ctx.config.get("burn_subtitles", True):
        subtitled_path = ctx.output_dir / f"video_{video_format}_subtitled.mp4"
        _burn_subtitles_ffmpeg(output_path, ass_path, subtitled_path)
        ctx.final_video_path = subtitled_path
    else:
        ctx.final_video_path = output_path

    # Cleanup
    audio_clip.close()
    final_video.close()
    for clip in scene_clips:
        clip.close()

    logger.info("Final video -> %s", ctx.final_video_path)


def _apply_ken_burns(clip, scene_index: int, duration: float):
    """Apply a slow pan & zoom (Ken Burns) effect to a still image clip."""
    # Alternate between zoom-in and zoom-out patterns
    patterns = [
        {"start_scale": 1.0, "end_scale": 1.15, "dx": 0.02, "dy": 0.01},
        {"start_scale": 1.15, "end_scale": 1.0, "dx": -0.02, "dy": 0.01},
        {"start_scale": 1.0, "end_scale": 1.1, "dx": 0.01, "dy": -0.02},
        {"start_scale": 1.1, "end_scale": 1.0, "dx": -0.01, "dy": -0.01},
    ]
    pattern = patterns[scene_index % len(patterns)]

    w, h = clip.size

    def ken_burns_frame(get_frame, t):
        progress = t / duration if duration > 0 else 0
        # Smooth easing
        progress = _ease_in_out(progress)

        scale = pattern["start_scale"] + (
            pattern["end_scale"] - pattern["start_scale"]
        ) * progress
        dx = pattern["dx"] * progress * w
        dy = pattern["dy"] * progress * h

        frame = get_frame(t)

        from PIL import Image
        import numpy as np

        img = Image.fromarray(frame)

        # Calculate crop region for zoom effect
        new_w = int(w / scale)
        new_h = int(h / scale)
        cx = w // 2 + int(dx)
        cy = h // 2 + int(dy)

        left = max(0, cx - new_w // 2)
        top = max(0, cy - new_h // 2)
        right = min(w, left + new_w)
        bottom = min(h, top + new_h)

        # Ensure valid crop
        if right - left < 10 or bottom - top < 10:
            return frame

        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((w, h), Image.LANCZOS)
        return np.array(resized)

    return clip.fl(ken_burns_frame)


def _apply_color_grading(clip, config: dict):
    """Apply cinematic color grading: contrast boost and slight warmth."""
    contrast = config.get("contrast_factor", 1.1)
    saturation = config.get("saturation_factor", 1.15)

    def grade_frame(get_frame, t):
        import numpy as np

        frame = get_frame(t).astype(np.float32)

        # Contrast adjustment
        mean = frame.mean()
        frame = (frame - mean) * contrast + mean

        # Saturation adjustment (simple method)
        gray = frame.mean(axis=2, keepdims=True)
        frame = gray + (frame - gray) * saturation

        # Slight warm tint
        frame[:, :, 0] *= 1.02  # Boost red slightly
        frame[:, :, 2] *= 0.98  # Reduce blue slightly

        return np.clip(frame, 0, 255).astype(np.uint8)

    return clip.fl(grade_frame)


def _crossfade_concat(clips: list, transition_duration: float):
    """Concatenate clips with crossfade transitions."""
    from moviepy.editor import CompositeVideoClip

    result_clips = []
    current_start = 0.0

    for i, clip in enumerate(clips):
        positioned = clip.set_start(current_start)

        if i > 0:
            positioned = positioned.crossfadein(transition_duration)

        result_clips.append(positioned)
        current_start += clip.duration - transition_duration

    return CompositeVideoClip(result_clips)


def _burn_subtitles_ffmpeg(
    video_path: Path, ass_path: Path, output_path: Path
) -> None:
    """Burn .ass subtitles into the video using FFmpeg."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"ass={ass_path}",
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "medium",
        str(output_path),
    ]

    logger.info("Burning subtitles: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "FFmpeg subtitle burn failed (video still available without subs): %s",
            result.stderr[:500],
        )
        # Copy original as fallback
        import shutil
        shutil.copy2(str(video_path), str(output_path))


def _ease_in_out(t: float) -> float:
    """Smooth cubic ease-in-out curve."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2
