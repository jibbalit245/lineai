"""
Configuration management for the video generator.

Loads settings from YAML/JSON config files and merges with CLI overrides.
Provides sensible defaults for every parameter.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    # --- Pipeline ---
    "output_dir": "output",

    # --- Script Generation ---
    "script_backend": "offline",       # offline | anthropic | openai | groq
    "anthropic_model": "claude-sonnet-4-20250514",
    "openai_model": "gpt-4o",
    "groq_model": "llama-3.3-70b-versatile",

    # --- Text-to-Speech ---
    "tts_backend": "gtts",             # gtts | pyttsx3 | elevenlabs
    "tts_language": "en",
    "tts_slow": False,
    "tts_rate": 175,
    "elevenlabs_voice_id": "21m00Tcm4TlvDq8ikWAM",
    "elevenlabs_model_id": "eleven_multilingual_v2",

    # --- Visual Generation ---
    "visual_backend": "programmatic",  # programmatic | dalle | pexels
    "video_format": "landscape",       # landscape | portrait | square
    "dalle_model": "dall-e-3",

    # --- Subtitles ---
    "subtitle_backend": "estimated",   # estimated | whisper
    "whisper_model": "base",
    "language": "en",
    "subtitle_font": "Arial",
    "subtitle_font_size": 48,
    "subtitle_color": "&H00FFFFFF",
    "subtitle_outline_color": "&H00000000",
    "burn_subtitles": True,

    # --- Audio Mixing ---
    "background_music_path": None,
    "music_volume_db": -18,
    "duck_amount_db": -12,
    "music_fade_ms": 2000,

    # --- Video Assembly ---
    "fps": 30,
    "transition_duration": 0.8,
    "ken_burns": True,
    "color_grading": True,
    "contrast_factor": 1.1,
    "saturation_factor": 1.15,
    "codec": "libx264",
    "audio_codec": "aac",
    "bitrate": "5000k",
}


def load_config(config_path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load configuration from file and apply overrides."""
    config = DEFAULT_CONFIG.copy()

    if config_path:
        path = Path(config_path)
        if path.exists():
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    with open(path) as f:
                        file_config = yaml.safe_load(f) or {}
                    config.update(file_config)
                    logger.info("Loaded config from %s", path)
                except ImportError:
                    logger.warning("PyYAML not installed, skipping YAML config")
            elif path.suffix == ".json":
                with open(path) as f:
                    file_config = json.load(f)
                config.update(file_config)
                logger.info("Loaded config from %s", path)
        else:
            logger.warning("Config file not found: %s", path)

    if overrides:
        config.update(overrides)

    return config
