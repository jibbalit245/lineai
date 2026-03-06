"""
Visual asset generation engine — the art department.

Creates or fetches images for each scene. Supports:
  - Programmatic generation: multi-stop radial/linear gradients, particle systems,
    geometric patterns, mesh gradients, noise textures, text overlays
  - AI generation via DALL-E 3 with retry logic
  - Stock photo fetching from Pexels with fallback chain
  - Automatic resizing and format-aware composition

Each scene gets one high-res image sized to the target video format.
"""

import hashlib
import logging
import math
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)

# Preset dimensions for common video formats
FORMAT_DIMENSIONS = {
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
    "square": (1080, 1080),
}

# Extended color palettes — each is a curated cinematic palette
PALETTES = [
    {"name": "midnight_fire",   "colors": ["#0d0d2b", "#1a1a4e", "#2d1b69", "#e94560", "#ff6b6b"]},
    {"name": "ocean_depths",    "colors": ["#0a192f", "#0d2847", "#1a4a6e", "#2ecc71", "#00d2ff"]},
    {"name": "sunset_gold",     "colors": ["#1a0a2e", "#3d1a54", "#e67e22", "#f39c12", "#f1c40f"]},
    {"name": "arctic_aurora",   "colors": ["#0c1445", "#1a2980", "#26d0ce", "#a8e6cf", "#dcedc1"]},
    {"name": "neon_noir",       "colors": ["#0a0a0a", "#1a1a2e", "#e84393", "#6c5ce7", "#00cec9"]},
    {"name": "forest_mist",     "colors": ["#0d1b0e", "#1a3a1c", "#2d6a2e", "#a8d5a2", "#f0f7ee"]},
    {"name": "desert_storm",    "colors": ["#2c1810", "#5c3a28", "#c69c6d", "#e6d5b8", "#f5ebe0"]},
    {"name": "cyberpunk",       "colors": ["#0a0012", "#1a0033", "#ff00ff", "#00ffff", "#ffff00"]},
    {"name": "monochrome_lux",  "colors": ["#0d0d0d", "#1a1a1a", "#333333", "#888888", "#f5f5f5"]},
    {"name": "tropical_heat",   "colors": ["#1a0a0a", "#6b2d2d", "#ff7675", "#fab1a0", "#55efc4"]},
]

# Geometric pattern generators
PATTERN_TYPES = [
    "circles", "hexgrid", "diagonal_lines", "dots", "waves",
    "triangles", "concentric", "grid", "starburst", "noise_field",
]


def generate_visuals(ctx: PipelineContext) -> None:
    """Generate one image per scene and store paths in ctx.image_paths."""
    backend = ctx.config.get("visual_backend", "programmatic")
    video_format = ctx.config.get("video_format", "landscape")
    width, height = FORMAT_DIMENSIONS.get(video_format, (1920, 1080))

    images_dir = ctx.output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    paths: list[Path] = []

    for i, scene in enumerate(ctx.scenes):
        out_path = images_dir / f"scene_{i:03d}.png"

        if backend == "dalle":
            _generate_dalle(scene["visual"], out_path, width, height, ctx.config)
        elif backend == "pexels":
            _fetch_pexels(scene["visual"], out_path, width, height, ctx.config)
        else:
            _generate_programmatic(scene, i, out_path, width, height, len(ctx.scenes))

        paths.append(out_path)
        logger.info("Visual %d/%d -> %s", i + 1, len(ctx.scenes), out_path)

    ctx.image_paths = paths


# ---------------------------------------------------------------------------
# Programmatic Generation (Zero-API, High Quality)
# ---------------------------------------------------------------------------

def _generate_programmatic(
    scene: dict, index: int, out_path: Path, width: int, height: int, total_scenes: int,
) -> None:
    """Create a visually rich scene image using purely programmatic techniques."""
    seed = int(hashlib.md5(scene.get("narration", str(index)).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    palette = PALETTES[index % len(PALETTES)]
    colors = palette["colors"]
    pattern = PATTERN_TYPES[rng.randint(0, len(PATTERN_TYPES) - 1)]

    # Layer 1: Background gradient (radial or linear)
    if rng.random() > 0.5:
        img = _radial_gradient(width, height, colors, rng)
    else:
        img = _multi_stop_gradient(width, height, colors, rng)

    draw = ImageDraw.Draw(img, "RGBA")

    # Layer 2: Noise texture for depth
    _apply_noise_texture(img, intensity=rng.randint(5, 15), rng=rng)

    # Layer 3: Geometric pattern
    accent_rgb = _hex_to_rgb(colors[-1])
    secondary_rgb = _hex_to_rgb(colors[-2])
    _draw_pattern(draw, width, height, pattern, accent_rgb, secondary_rgb, rng)

    # Layer 4: Bokeh / light flare
    if rng.random() > 0.4:
        _draw_bokeh(img, width, height, colors, rng)

    # Layer 5: Scene text
    narration = scene.get("narration", "")
    _draw_text_block(draw, narration, width, height, accent_rgb, rng)

    # Layer 6: Scene indicator
    _draw_scene_indicator(draw, index + 1, total_scenes, width, height)

    # Post-processing: very subtle blur for depth
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    img.save(str(out_path), "PNG")


def _radial_gradient(width: int, height: int, colors: list[str], rng) -> Image.Image:
    """Radial gradient from offset center outward."""
    arr = np.zeros((height, width, 3), dtype=np.float32)
    cx = width * rng.uniform(0.3, 0.7)
    cy = height * rng.uniform(0.3, 0.7)
    max_dist = math.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)

    color_rgbs = [_hex_to_rgb(c) for c in colors]
    num_stops = len(color_rgbs)

    Y, X = np.mgrid[:height, :width]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist
    dist = np.clip(dist, 0, 0.9999)

    for ch in range(3):
        for s in range(num_stops - 1):
            lower = s / (num_stops - 1)
            upper = (s + 1) / (num_stops - 1)
            mask = (dist >= lower) & (dist < upper)
            local_t = (dist[mask] - lower) / (upper - lower)
            arr[:, :, ch][mask] = (
                color_rgbs[s][ch] + (color_rgbs[s + 1][ch] - color_rgbs[s][ch]) * local_t
            )

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _multi_stop_gradient(width: int, height: int, colors: list[str], rng) -> Image.Image:
    """Multi-stop linear gradient with random angle."""
    arr = np.zeros((height, width, 3), dtype=np.float32)
    angle_rad = rng.uniform(0, math.pi)
    color_rgbs = [_hex_to_rgb(c) for c in colors]
    num_stops = len(color_rgbs)

    Y, X = np.mgrid[:height, :width]
    proj = X * math.cos(angle_rad) + Y * math.sin(angle_rad)
    proj_min, proj_max = proj.min(), proj.max()
    t = (proj - proj_min) / (proj_max - proj_min) if proj_max > proj_min else np.zeros_like(proj)

    for ch in range(3):
        for s in range(num_stops - 1):
            lower = s / (num_stops - 1)
            upper = (s + 1) / (num_stops - 1)
            mask = (t >= lower) & (t < upper)
            local_t = (t[mask] - lower) / (upper - lower)
            arr[:, :, ch][mask] = (
                color_rgbs[s][ch] + (color_rgbs[s + 1][ch] - color_rgbs[s][ch]) * local_t
            )
        arr[:, :, ch][t >= 1.0] = color_rgbs[-1][ch]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_noise_texture(img: Image.Image, intensity: int = 10, rng=None) -> None:
    """Add subtle noise for organic feel."""
    arr = np.array(img).astype(np.float32)
    seed = rng.randint(0, 999999) if rng else 42
    noise = np.random.default_rng(seed).normal(0, intensity, arr.shape).astype(np.float32)
    img.paste(Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8)))


def _draw_pattern(
    draw: ImageDraw.ImageDraw,
    width: int, height: int,
    pattern: str,
    accent: tuple, secondary: tuple,
    rng,
) -> None:
    """Draw a geometric pattern overlay."""
    alpha = rng.randint(20, 60)
    accent_a = (*accent, alpha)

    if pattern == "circles":
        for _ in range(rng.randint(5, 12)):
            cx, cy = rng.randint(0, width), rng.randint(0, height)
            r = rng.randint(50, 300)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent_a, width=rng.randint(1, 3))

    elif pattern == "hexgrid":
        hex_size = rng.randint(60, 120)
        for row in range(0, height + hex_size, int(hex_size * 1.5)):
            offset = hex_size if (row // int(hex_size * 1.5)) % 2 else 0
            for col in range(offset, width + hex_size, hex_size * 2):
                _draw_hexagon(draw, col, row, hex_size // 2, accent_a)

    elif pattern == "diagonal_lines":
        spacing = rng.randint(30, 80)
        for offset in range(-height, width + height, spacing):
            draw.line([(offset, 0), (offset + height, height)], fill=accent_a, width=1)

    elif pattern == "dots":
        spacing = rng.randint(40, 80)
        dot_r = rng.randint(2, 6)
        for y in range(0, height, spacing):
            for x in range(0, width, spacing):
                jx, jy = x + rng.randint(-5, 5), y + rng.randint(-5, 5)
                draw.ellipse([jx - dot_r, jy - dot_r, jx + dot_r, jy + dot_r], fill=accent_a)

    elif pattern == "waves":
        for w in range(rng.randint(3, 8)):
            points = []
            amp = rng.randint(20, 80)
            freq = rng.uniform(0.005, 0.02)
            y_off = height * (w + 1) / (rng.randint(3, 8) + 1)
            for x in range(0, width, 3):
                points.append((x, int(y_off + amp * math.sin(freq * x + w * 0.5))))
            if len(points) > 1:
                draw.line(points, fill=accent_a, width=2)

    elif pattern == "triangles":
        for _ in range(rng.randint(3, 8)):
            cx, cy = rng.randint(0, width), rng.randint(0, height)
            size = rng.randint(50, 200)
            pts = [(cx, cy - size), (cx - int(size * 0.866), cy + size // 2),
                   (cx + int(size * 0.866), cy + size // 2)]
            draw.polygon(pts, outline=accent_a)

    elif pattern == "concentric":
        cx, cy = width // 2, height // 2
        spacing = rng.randint(30, 60)
        for r in range(spacing, max(width, height), spacing):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=accent_a, width=1)

    elif pattern == "grid":
        spacing = rng.randint(40, 100)
        for x in range(0, width, spacing):
            draw.line([(x, 0), (x, height)], fill=accent_a, width=1)
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=accent_a, width=1)

    elif pattern == "starburst":
        cx, cy = width // 2, height // 2
        for i in range(rng.randint(12, 24)):
            angle = (2 * math.pi * i) / 24
            length = rng.randint(200, max(width, height))
            draw.line([(cx, cy), (cx + int(length * math.cos(angle)),
                                   cy + int(length * math.sin(angle)))], fill=accent_a, width=1)

    elif pattern == "noise_field":
        for _ in range(rng.randint(20, 60)):
            x, y = rng.randint(0, width), rng.randint(0, height)
            w, h = rng.randint(5, 40), rng.randint(5, 40)
            draw.rectangle([x, y, x + w, y + h], outline=accent_a)


def _draw_hexagon(draw, cx, cy, radius, color):
    points = []
    for i in range(6):
        angle = math.pi / 3 * i - math.pi / 6
        points.append((cx + int(radius * math.cos(angle)), cy + int(radius * math.sin(angle))))
    draw.polygon(points, outline=color)


def _draw_bokeh(img: Image.Image, width: int, height: int, colors: list[str], rng) -> None:
    """Draw soft bokeh light flares for depth."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(3, 10)):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        radius = rng.randint(40, 200)
        rgb = _hex_to_rgb(colors[rng.randint(0, len(colors) - 1)])
        alpha = rng.randint(10, 35)

        for r in range(radius, 0, -5):
            a = int(alpha * (r / radius) ** 0.5)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*rgb, a))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=15))
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    img.paste(composited.convert("RGB"))


def _draw_text_block(
    draw: ImageDraw.ImageDraw, text: str, width: int, height: int, accent: tuple, rng,
) -> None:
    """Render narration text centered with backdrop and accent underline."""
    if not text.strip():
        return

    font_size = max(32, width // 25)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("Arial Bold", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    max_chars = max(20, width // (font_size // 2))
    lines = _word_wrap(text, max_chars)
    block = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), block, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (width - text_w) // 2, (height - text_h) // 2

    # Rounded backdrop
    pad = 20
    draw.rounded_rectangle(
        [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
        radius=15, fill=(0, 0, 0, 120),
    )

    draw.multiline_text((x + 3, y + 3), block, fill=(0, 0, 0, 200), font=font, align="center")
    draw.multiline_text((x, y), block, fill=(255, 255, 255, 240), font=font, align="center")

    # Accent underline
    ul_y = y + text_h + 10
    ul_w = min(text_w, width // 2)
    draw.line([(width // 2 - ul_w // 2, ul_y), (width // 2 + ul_w // 2, ul_y)],
              fill=(*accent, 150), width=3)


def _draw_scene_indicator(draw, scene_num: int, total: int, width: int, height: int) -> None:
    font_size = max(16, width // 60)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((width - font_size * 3, height - font_size * 2),
              f"{scene_num}/{total}", fill=(255, 255, 255, 80), font=font)


def _word_wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ---------------------------------------------------------------------------
# DALL-E Generation (with retry)
# ---------------------------------------------------------------------------

def _generate_dalle(
    visual_desc: str, out_path: Path, width: int, height: int, config: dict
) -> None:
    """Generate via DALL-E 3 with exponential backoff retry."""
    from openai import OpenAI

    client = OpenAI(api_key=config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY"))
    size = "1792x1024" if width > height else ("1024x1792" if height > width else "1024x1024")

    prompt = (
        f"Cinematic still frame, high production value, photorealistic, "
        f"professional lighting, depth of field: {visual_desc}"
    )

    for attempt in range(3):
        try:
            response = client.images.generate(
                model=config.get("dalle_model", "dall-e-3"),
                prompt=prompt, size=size, quality="hd", n=1,
            )
            urllib.request.urlretrieve(response.data[0].url, str(out_path))
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                logger.warning("DALL-E attempt %d failed, retrying in %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)
            else:
                logger.error("DALL-E failed after 3 attempts, falling back to programmatic")
                _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height, 1)
                return

    img = Image.open(str(out_path)).resize((width, height), Image.LANCZOS)
    img.save(str(out_path), "PNG")


# ---------------------------------------------------------------------------
# Pexels Stock Photos (with retry)
# ---------------------------------------------------------------------------

def _fetch_pexels(
    visual_desc: str, out_path: Path, width: int, height: int, config: dict
) -> None:
    """Fetch from Pexels with retry and graceful fallback."""
    import json as json_mod

    api_key = config.get("pexels_api_key") or os.environ.get("PEXELS_API_KEY")
    if not api_key:
        logger.warning("No Pexels API key, falling back to programmatic")
        _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height, 1)
        return

    query = " ".join(visual_desc.split()[:5])
    orientation = "landscape" if width > height else "portrait"
    url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page=3&orientation={orientation}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"Authorization": api_key})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json_mod.loads(resp.read())
            if not data.get("photos"):
                logger.warning("No Pexels results for '%s'", query)
                _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height, 1)
                return
            photo_url = data["photos"][0]["src"].get("large2x") or data["photos"][0]["src"]["original"]
            urllib.request.urlretrieve(photo_url, str(out_path))
            break
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                logger.warning("Pexels attempt %d failed, retrying in %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)
            else:
                logger.error("Pexels failed, falling back to programmatic")
                _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height, 1)
                return

    img = Image.open(str(out_path)).resize((width, height), Image.LANCZOS)
    img.save(str(out_path), "PNG")
