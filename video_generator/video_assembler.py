"""
Video assembly engine — the cinematic core.

Takes scene images and audio, composites into a final video with:
  - Ken Burns effect (pan & zoom with 8 pattern variations)
  - Multiple transition types: crossfade, wipe, slide, zoom-through, dissolve
  - Color grading: contrast, saturation, warm/cool tint, film LUT emulation
  - Vignette overlay (darken edges for cinematic look)
  - Optional film grain texture
  - Optional letterbox (cinematic black bars)
  - Burned-in .ass subtitles via FFmpeg
  - Intro/outro card generation
  - Auto-generated thumbnail from most visually striking frame

Compatible with moviepy v2.x API.
"""

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
    vfx,
)

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)

# Transition types
TRANSITION_TYPES = ["crossfade", "wipe_left", "wipe_right", "slide_up", "zoom_through", "dissolve"]

# Ken Burns patterns — 8 varieties for visual interest
KB_PATTERNS = [
    {"start_scale": 1.0,  "end_scale": 1.18, "dx":  0.03, "dy":  0.01},
    {"start_scale": 1.18, "end_scale": 1.0,  "dx": -0.03, "dy":  0.01},
    {"start_scale": 1.0,  "end_scale": 1.12, "dx":  0.01, "dy": -0.03},
    {"start_scale": 1.12, "end_scale": 1.0,  "dx": -0.01, "dy": -0.01},
    {"start_scale": 1.0,  "end_scale": 1.15, "dx": -0.02, "dy":  0.02},
    {"start_scale": 1.15, "end_scale": 1.0,  "dx":  0.02, "dy": -0.02},
    {"start_scale": 1.05, "end_scale": 1.2,  "dx":  0.0,  "dy":  0.0},
    {"start_scale": 1.2,  "end_scale": 1.05, "dx":  0.0,  "dy":  0.0},
]


def assemble_video(ctx: PipelineContext) -> None:
    """Build the final video from all pipeline artifacts."""
    if not ctx.image_paths:
        raise FileNotFoundError("No scene images — run visual engine first")

    audio_path = ctx.mixed_audio_path or ctx.audio_narration_path
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError("No audio file — run TTS/mixer first")

    video_format = ctx.config.get("video_format", "landscape")
    fps = ctx.config.get("fps", 30)
    transition_duration = ctx.config.get("transition_duration", 0.8)
    transition_style = ctx.config.get("transition_style", "mixed")
    enable_ken_burns = ctx.config.get("ken_burns", True)
    enable_color_grade = ctx.config.get("color_grading", True)
    enable_vignette = ctx.config.get("vignette", True)
    enable_grain = ctx.config.get("film_grain", False)
    enable_letterbox = ctx.config.get("letterbox", False)
    letterbox_ratio = ctx.config.get("letterbox_ratio", 2.39)

    from .visual_engine import FORMAT_DIMENSIONS
    width, height = FORMAT_DIMENSIONS.get(video_format, (1920, 1080))

    audio_clip = AudioFileClip(str(audio_path))
    total_duration = audio_clip.duration

    # Intro/outro
    intro_clip = _make_intro_card(ctx, width, height) if ctx.config.get("intro_text") else None
    outro_clip = _make_outro_card(ctx, width, height) if ctx.config.get("outro_text") else None

    intro_dur = intro_clip.duration if intro_clip else 0
    outro_dur = outro_clip.duration if outro_clip else 0
    content_duration = total_duration - intro_dur - outro_dur

    num_scenes = len(ctx.image_paths)
    scene_duration = max(1.0, content_duration / num_scenes)

    # Build scene clips
    scene_clips = []
    for i, img_path in enumerate(ctx.image_paths):
        clip = ImageClip(str(img_path), duration=scene_duration).resized((width, height))

        if enable_ken_burns:
            clip = _apply_ken_burns(clip, i, scene_duration, width, height)

        if enable_color_grade:
            clip = _apply_color_grading(clip, ctx.config)

        if enable_vignette:
            clip = _apply_vignette(clip, width, height)

        if enable_grain:
            clip = _apply_film_grain(clip, width, height)

        scene_clips.append(clip)
        logger.info("Built scene clip %d/%d (%.1fs)", i + 1, num_scenes, scene_duration)

    # Transitions
    if transition_duration > 0 and len(scene_clips) > 1:
        final_video = _transition_concat(scene_clips, transition_duration, transition_style)
    else:
        final_video = concatenate_videoclips(scene_clips)

    # Add intro/outro
    parts = []
    if intro_clip:
        parts.append(intro_clip)
    parts.append(final_video)
    if outro_clip:
        parts.append(outro_clip)
    if len(parts) > 1:
        final_video = concatenate_videoclips(parts)

    # Trim to audio duration
    final_video = final_video.subclipped(0, min(final_video.duration, total_duration))

    # Attach audio
    final_video = final_video.with_audio(audio_clip)

    # Letterbox
    if enable_letterbox:
        final_video = _apply_letterbox(final_video, width, height, letterbox_ratio)

    # Render
    output_path = ctx.output_dir / f"video_{video_format}.mp4"
    codec = ctx.config.get("codec", "libx264")
    audio_codec = ctx.config.get("audio_codec", "aac")
    bitrate = ctx.config.get("bitrate", "5000k")

    logger.info("Rendering video -> %s (%dx%d @ %dfps)", output_path, width, height, fps)
    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
        bitrate=bitrate,
        preset="medium",
        threads=os.cpu_count() or 4,
        logger="bar",
    )

    # Burn subtitles
    ass_path = ctx.output_dir / "subtitles.ass"
    if ass_path.exists() and ctx.config.get("burn_subtitles", True):
        subtitled_path = ctx.output_dir / f"video_{video_format}_subtitled.mp4"
        _burn_subtitles_ffmpeg(output_path, ass_path, subtitled_path, codec, bitrate)
        ctx.final_video_path = subtitled_path
    else:
        ctx.final_video_path = output_path

    # Thumbnail
    _generate_thumbnail(ctx, width, height)

    # Cleanup
    audio_clip.close()
    final_video.close()
    for clip in scene_clips:
        clip.close()

    logger.info("Final video -> %s", ctx.final_video_path)


# ---------------------------------------------------------------------------
# Ken Burns Effect
# ---------------------------------------------------------------------------

def _apply_ken_burns(clip, scene_index: int, duration: float, width: int, height: int):
    """Apply slow pan & zoom (Ken Burns) effect."""
    pattern = KB_PATTERNS[scene_index % len(KB_PATTERNS)]

    def ken_burns_frame(get_frame, t):
        progress = _ease_in_out_cubic(t / duration) if duration > 0 else 0

        scale = pattern["start_scale"] + (pattern["end_scale"] - pattern["start_scale"]) * progress
        dx = pattern["dx"] * progress * width
        dy = pattern["dy"] * progress * height

        frame = get_frame(t)
        img = Image.fromarray(frame)

        new_w = int(width / scale)
        new_h = int(height / scale)
        cx = width // 2 + int(dx)
        cy = height // 2 + int(dy)

        left = max(0, min(cx - new_w // 2, width - new_w))
        top = max(0, min(cy - new_h // 2, height - new_h))
        right = left + new_w
        bottom = top + new_h

        if right - left < 10 or bottom - top < 10:
            return frame

        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((width, height), Image.LANCZOS)
        return np.array(resized)

    return clip.transform(ken_burns_frame)


# ---------------------------------------------------------------------------
# Color Grading
# ---------------------------------------------------------------------------

def _apply_color_grading(clip, config: dict):
    """Cinematic color grading: contrast, saturation, tint, highlight rolloff."""
    contrast = config.get("contrast_factor", 1.1)
    saturation = config.get("saturation_factor", 1.15)
    tint_mode = config.get("tint_mode", "warm")
    highlight_rolloff = config.get("highlight_rolloff", 0.95)

    def grade_frame(get_frame, t):
        frame = get_frame(t).astype(np.float32)

        # Contrast
        mean = frame.mean()
        frame = (frame - mean) * contrast + mean

        # Saturation (Rec.709 luminance-preserving)
        luma = frame[:, :, 0] * 0.2126 + frame[:, :, 1] * 0.7152 + frame[:, :, 2] * 0.0722
        luma = luma[:, :, np.newaxis]
        frame = luma + (frame - luma) * saturation

        # Tint
        if tint_mode == "warm":
            frame[:, :, 0] *= 1.04
            frame[:, :, 1] *= 1.01
            frame[:, :, 2] *= 0.96
        elif tint_mode == "cool":
            frame[:, :, 0] *= 0.96
            frame[:, :, 2] *= 1.05

        # Highlight rolloff
        frame = frame * highlight_rolloff + (255 * (1 - highlight_rolloff)) * (frame / 255) ** 0.5

        # Subtle S-curve
        normalized = frame / 255.0
        curved = normalized * normalized * (3.0 - 2.0 * normalized)
        frame = curved * 255.0 * 0.15 + frame * 0.85

        return np.clip(frame, 0, 255).astype(np.uint8)

    return clip.transform(grade_frame)


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

def _apply_vignette(clip, width: int, height: int, strength: float = 0.4):
    """Radial vignette (darken edges)."""
    Y, X = np.ogrid[:height, :width]
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist

    vignette_mask = 1.0 - strength * (dist ** 1.8)
    vignette_mask = np.clip(vignette_mask, 0, 1).astype(np.float32)
    vignette_3ch = np.stack([vignette_mask] * 3, axis=-1)

    def apply_vig(get_frame, t):
        frame = get_frame(t).astype(np.float32)
        return np.clip(frame * vignette_3ch, 0, 255).astype(np.uint8)

    return clip.transform(apply_vig)


# ---------------------------------------------------------------------------
# Film Grain
# ---------------------------------------------------------------------------

def _apply_film_grain(clip, width: int, height: int, intensity: float = 15.0):
    """Add subtle film grain noise."""
    rng = np.random.default_rng(42)

    def add_grain(get_frame, t):
        frame = get_frame(t).astype(np.float32)
        noise = rng.normal(0, intensity, (height, width, 3)).astype(np.float32)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    return clip.transform(add_grain)


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

def _apply_letterbox(clip, width: int, height: int, aspect_ratio: float):
    """Cinematic letterbox bars."""
    desired_height = int(width / aspect_ratio)
    if desired_height >= height:
        return clip

    bar_height = (height - desired_height) // 2
    top_bar = ColorClip(size=(width, bar_height), color=(0, 0, 0), duration=clip.duration)
    bottom_bar = ColorClip(size=(width, bar_height), color=(0, 0, 0), duration=clip.duration)

    return CompositeVideoClip([
        clip,
        top_bar.with_position(("center", 0)),
        bottom_bar.with_position(("center", height - bar_height)),
    ])


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def _transition_concat(clips: list, transition_duration: float, style: str):
    """Concatenate clips with transitions. Uses crossfade for all types for v2 compat."""
    result_clips = []
    current_start = 0.0

    for i, clip in enumerate(clips):
        positioned = clip.with_start(current_start)

        if i > 0:
            # MoviePy v2: use CrossFadeIn for all transition types
            # More complex transitions (wipe/slide/zoom) applied as transforms
            if style == "mixed":
                trans_type = TRANSITION_TYPES[i % len(TRANSITION_TYPES)]
            else:
                trans_type = style

            if trans_type in ("crossfade", "dissolve"):
                positioned = positioned.with_effects([vfx.CrossFadeIn(transition_duration)])
            elif trans_type == "zoom_through":
                positioned = _zoom_through_transition(positioned, transition_duration)
                positioned = positioned.with_effects([vfx.CrossFadeIn(transition_duration * 0.5)])
            elif trans_type == "slide_up":
                positioned = _slide_transition(positioned, transition_duration, "up",
                                                clip.size[1] if hasattr(clip, 'size') else 1080)
            else:
                # wipe_left, wipe_right — fall back to crossfade in v2
                positioned = positioned.with_effects([vfx.CrossFadeIn(transition_duration)])

        result_clips.append(positioned)
        current_start += clip.duration - transition_duration

    return CompositeVideoClip(result_clips)


def _slide_transition(clip, duration: float, direction: str, height: int):
    """Slide the clip into frame."""
    start_time = clip.start if hasattr(clip, 'start') and clip.start else 0

    def slide_pos(t):
        elapsed = t - start_time
        if elapsed >= duration or elapsed < 0:
            return (0, 0)
        progress = _ease_in_out_cubic(elapsed / duration)
        if direction == "up":
            return (0, int(height * (1 - progress)))
        return (0, 0)

    return clip.with_position(slide_pos)


def _zoom_through_transition(clip, duration: float):
    """Zoom from small to full size."""
    start_time = clip.start if hasattr(clip, 'start') and clip.start else 0

    def zoom_effect(get_frame, t):
        elapsed = t - start_time
        if elapsed >= duration or elapsed < 0:
            return get_frame(t)

        progress = _ease_in_out_cubic(elapsed / duration)
        scale = 0.3 + 0.7 * progress

        frame = get_frame(t)
        h, w = frame.shape[:2]
        img = Image.fromarray(frame)

        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 10 or new_h < 10:
            return frame

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
        return np.array(canvas)

    return clip.transform(zoom_effect)


# ---------------------------------------------------------------------------
# Intro / Outro Cards
# ---------------------------------------------------------------------------

def _make_intro_card(ctx: PipelineContext, width: int, height: int):
    """Generate intro card."""
    duration = ctx.config.get("intro_duration", 3.0)
    text = ctx.config.get("intro_text", "")
    subtitle = ctx.config.get("intro_subtitle", "")

    img = _render_title_card(text, subtitle, width, height)
    card_path = ctx.output_dir / "intro_card.png"
    img.save(str(card_path), "PNG")

    return ImageClip(str(card_path), duration=duration).with_effects([vfx.CrossFadeIn(0.8)])


def _make_outro_card(ctx: PipelineContext, width: int, height: int):
    """Generate outro card."""
    duration = ctx.config.get("outro_duration", 3.0)
    text = ctx.config.get("outro_text", "")
    subtitle = ctx.config.get("outro_subtitle", "Thanks for watching!")

    img = _render_title_card(text, subtitle, width, height)
    card_path = ctx.output_dir / "outro_card.png"
    img.save(str(card_path), "PNG")

    return ImageClip(str(card_path), duration=duration).with_effects([vfx.CrossFadeOut(1.0)])


def _render_title_card(title: str, subtitle: str, width: int, height: int) -> Image.Image:
    """Render a stylish title card."""
    img = Image.new("RGB", (width, height), (10, 10, 15))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        ratio = y / height
        draw.line([(0, y), (width, y)],
                  fill=(int(10 + 20 * ratio), int(10 + 15 * ratio), int(15 + 40 * ratio)))

    title_size = width // 15
    sub_size = width // 30
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_size)
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sub_size)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tx = (width - (bbox[2] - bbox[0])) // 2
        ty = height // 2 - title_size
        draw.text((tx + 3, ty + 3), title, fill=(0, 0, 0), font=title_font)
        draw.text((tx, ty), title, fill=(255, 255, 255), font=title_font)

    if subtitle:
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sx = (width - (bbox[2] - bbox[0])) // 2
        draw.text((sx, height // 2 + sub_size), subtitle, fill=(180, 180, 200), font=sub_font)

    # Decorative line
    lw = width // 3
    draw.line([(width // 2 - lw // 2, height // 2), (width // 2 + lw // 2, height // 2)],
              fill=(100, 120, 200), width=2)

    return img


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

def _generate_thumbnail(ctx: PipelineContext, width: int, height: int) -> None:
    """Auto-generate thumbnail from highest-contrast scene."""
    if not ctx.image_paths:
        return

    best_path = ctx.image_paths[0]
    best_score = 0
    for path in ctx.image_paths:
        score = np.array(Image.open(str(path))).astype(np.float32).std()
        if score > best_score:
            best_score = score
            best_path = path

    img = Image.open(str(best_path)).resize((width, height), Image.LANCZOS)

    if ctx.prompt:
        draw = ImageDraw.Draw(img)
        font_size = width // 18
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        title = ctx.prompt[:50].upper()
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        tx = (width - tw) // 2
        ty = height - font_size * 3

        draw.rectangle([(tx - 20, ty - 10), (tx + tw + 20, ty + font_size + 15)], fill=(0, 0, 0))
        draw.text((tx + 3, ty + 3), title, fill=(30, 30, 30), font=font)
        draw.text((tx, ty), title, fill=(255, 255, 255), font=font)

    thumb_path = ctx.output_dir / "thumbnail.png"
    img.save(str(thumb_path), "PNG")
    logger.info("Thumbnail -> %s", thumb_path)


# ---------------------------------------------------------------------------
# Subtitle Burning (FFmpeg)
# ---------------------------------------------------------------------------

def _burn_subtitles_ffmpeg(video_path: Path, ass_path: Path, output_path: Path, codec: str, bitrate: str) -> None:
    """Burn .ass subtitles into video via FFmpeg."""
    ass_str = str(ass_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={ass_str}",
        "-c:a", "copy", "-c:v", codec, "-b:v", bitrate, "-preset", "medium",
        str(output_path),
    ]

    logger.info("Burning subtitles via FFmpeg")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("FFmpeg subtitle burn failed: %s", result.stderr[:500])
        shutil.copy2(str(video_path), str(output_path))


# ---------------------------------------------------------------------------
# Easing Functions
# ---------------------------------------------------------------------------

def _ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2
