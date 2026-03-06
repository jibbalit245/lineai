"""
Visual asset generation engine.

Creates or fetches images for each scene. Supports:
  - AI generation via DALL-E or Stable Diffusion API
  - Stock photo fetching from Pexels/Unsplash
  - Programmatic solid/gradient backgrounds with text overlay

Each scene gets one high-res image sized to the target video format.
"""

import logging
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)

# Preset dimensions for common video formats
FORMAT_DIMENSIONS = {
    "landscape": (1920, 1080),
    "portrait": (1080, 1920),
    "square": (1080, 1080),
}

# Curated color palettes for generated backgrounds
PALETTES = [
    ("#1a1a2e", "#16213e", "#0f3460", "#e94560"),
    ("#0d0d0d", "#1a1a1a", "#333333", "#f5f5f5"),
    ("#2d3436", "#636e72", "#b2bec3", "#dfe6e9"),
    ("#6c5ce7", "#a29bfe", "#fd79a8", "#e84393"),
    ("#00b894", "#00cec9", "#0984e3", "#6c5ce7"),
    ("#ff7675", "#fab1a0", "#ffeaa7", "#55efc4"),
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
            _generate_programmatic(scene, i, out_path, width, height)

        paths.append(out_path)
        logger.info("Visual %d/%d -> %s", i + 1, len(ctx.scenes), out_path)

    ctx.image_paths = paths


def _generate_programmatic(
    scene: dict, index: int, out_path: Path, width: int, height: int
) -> None:
    """Create a visually rich gradient background with styled text overlay."""
    palette = PALETTES[index % len(PALETTES)]

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Draw a vertical gradient using the palette
    num_colors = len(palette)
    for y in range(height):
        ratio = y / height
        segment = ratio * (num_colors - 1)
        idx = int(segment)
        local_ratio = segment - idx
        if idx >= num_colors - 1:
            idx = num_colors - 2
            local_ratio = 1.0

        c1 = _hex_to_rgb(palette[idx])
        c2 = _hex_to_rgb(palette[idx + 1])
        r = int(c1[0] + (c2[0] - c1[0]) * local_ratio)
        g = int(c1[1] + (c2[1] - c1[1]) * local_ratio)
        b = int(c1[2] + (c2[2] - c1[2]) * local_ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add decorative geometric elements
    accent_color = _hex_to_rgb(palette[-1])
    _draw_decorations(draw, width, height, accent_color, index)

    # Add scene text
    narration = scene.get("narration", "")
    _draw_text_block(draw, narration, width, height, accent_color)

    img.save(str(out_path), "PNG")


def _draw_decorations(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    accent: tuple[int, int, int],
    seed: int,
) -> None:
    """Add subtle geometric decorations for visual interest."""
    import random

    rng = random.Random(seed * 42)

    # Draw circles
    for _ in range(5):
        cx = rng.randint(0, width)
        cy = rng.randint(0, height)
        radius = rng.randint(50, 200)
        opacity_color = (*accent, 40)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(*accent, 80),
            width=2,
        )

    # Draw diagonal lines
    for _ in range(3):
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = x1 + rng.randint(-400, 400)
        y2 = y1 + rng.randint(-400, 400)
        draw.line([(x1, y1), (x2, y2)], fill=(*accent, 50), width=1)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    height: int,
    accent: tuple[int, int, int],
) -> None:
    """Render narration text centered on the image with a shadow effect."""
    # Try to load a nice font, fall back to default
    font_size = max(32, width // 25)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("Arial Bold", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Word-wrap the text
    max_chars = max(20, width // (font_size // 2))
    lines = _word_wrap(text, max_chars)
    block = "\n".join(lines)

    bbox = draw.multiline_textbbox((0, 0), block, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) // 2
    y = (height - text_h) // 2

    # Draw shadow
    draw.multiline_text((x + 3, y + 3), block, fill=(0, 0, 0, 180), font=font, align="center")
    # Draw main text
    draw.multiline_text((x, y), block, fill=(255, 255, 255), font=font, align="center")


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


def _generate_dalle(
    visual_desc: str, out_path: Path, width: int, height: int, config: dict
) -> None:
    """Generate an image via OpenAI DALL-E API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )

    # DALL-E 3 supports specific sizes
    if width > height:
        size = "1792x1024"
    elif height > width:
        size = "1024x1792"
    else:
        size = "1024x1024"

    response = client.images.generate(
        model=config.get("dalle_model", "dall-e-3"),
        prompt=f"Cinematic, high quality, photorealistic: {visual_desc}",
        size=size,
        quality="hd",
        n=1,
    )

    import urllib.request
    image_url = response.data[0].url
    urllib.request.urlretrieve(image_url, str(out_path))

    # Resize to exact target dimensions
    img = Image.open(str(out_path))
    img = img.resize((width, height), Image.LANCZOS)
    img.save(str(out_path), "PNG")


def _fetch_pexels(
    visual_desc: str, out_path: Path, width: int, height: int, config: dict
) -> None:
    """Fetch a stock photo from Pexels API."""
    import urllib.request
    import json as json_mod

    api_key = config.get("pexels_api_key") or os.environ.get("PEXELS_API_KEY")
    if not api_key:
        logger.warning("No Pexels API key, falling back to programmatic")
        _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height)
        return

    query = " ".join(visual_desc.split()[:5])
    url = f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}&per_page=1&orientation={'landscape' if width > height else 'portrait'}"

    req = urllib.request.Request(url, headers={"Authorization": api_key})
    with urllib.request.urlopen(req) as resp:
        data = json_mod.loads(resp.read())

    if not data.get("photos"):
        logger.warning("No Pexels results for '%s', using programmatic", query)
        _generate_programmatic({"narration": visual_desc}, 0, out_path, width, height)
        return

    photo_url = data["photos"][0]["src"]["original"]
    urllib.request.urlretrieve(photo_url, str(out_path))

    img = Image.open(str(out_path))
    img = img.resize((width, height), Image.LANCZOS)
    img.save(str(out_path), "PNG")
