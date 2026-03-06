#!/usr/bin/env python3
"""
Command-line interface for the LineAI Automated Video Generator.

Usage:
    python -m video_generator "Your video topic here"
    python -m video_generator --config config.json "Explain quantum computing"
    python -m video_generator --format portrait --tts elevenlabs "Top 5 life hacks"

Full pipeline: prompt -> script -> speech -> visuals -> subtitles -> mix -> render
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline, PipelineContext
from .script_generator import generate_script
from .tts_engine import synthesize_speech
from .visual_engine import generate_visuals
from .subtitle_engine import generate_subtitles
from .audio_mixer import mix_audio
from .video_assembler import assemble_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_generator",
        description="LineAI Automated Video Generator — turn any prompt into a polished video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s "The history of the internet"
  %(prog)s --format portrait --tts gtts "5 Python tips"
  %(prog)s --config my_config.json --script-backend anthropic "AI explained"
  %(prog)s --visual-backend dalle --burn-subtitles "Space exploration"
        """,
    )

    parser.add_argument("prompt", help="The topic or script for the video")
    parser.add_argument(
        "-o", "--output-dir", default="output", help="Output directory (default: output)"
    )
    parser.add_argument(
        "-c", "--config", default=None, help="Path to config file (JSON or YAML)"
    )

    # Script options
    script_group = parser.add_argument_group("Script Generation")
    script_group.add_argument(
        "--script-backend",
        choices=["offline", "anthropic", "openai", "groq"],
        help="Script generation backend",
    )

    # TTS options
    tts_group = parser.add_argument_group("Text-to-Speech")
    tts_group.add_argument(
        "--tts", choices=["gtts", "pyttsx3", "elevenlabs"], help="TTS engine"
    )
    tts_group.add_argument("--tts-language", help="TTS language code (default: en)")

    # Visual options
    vis_group = parser.add_argument_group("Visuals")
    vis_group.add_argument(
        "--visual-backend",
        choices=["programmatic", "dalle", "pexels"],
        help="Visual generation backend",
    )
    vis_group.add_argument(
        "--format",
        choices=["landscape", "portrait", "square"],
        help="Video format / aspect ratio",
    )

    # Subtitle options
    sub_group = parser.add_argument_group("Subtitles")
    sub_group.add_argument(
        "--subtitle-backend",
        choices=["estimated", "whisper"],
        help="Subtitle timing backend",
    )
    sub_group.add_argument(
        "--no-subtitles", action="store_true", help="Disable subtitle burning"
    )

    # Audio options
    audio_group = parser.add_argument_group("Audio")
    audio_group.add_argument(
        "--music", help="Path to background music file"
    )
    audio_group.add_argument(
        "--music-volume", type=int, help="Background music volume in dB (default: -18)"
    )

    # Video options
    video_group = parser.add_argument_group("Video Output")
    video_group.add_argument("--fps", type=int, help="Frames per second (default: 30)")
    video_group.add_argument(
        "--no-ken-burns", action="store_true", help="Disable Ken Burns effect"
    )
    video_group.add_argument(
        "--no-color-grading", action="store_true", help="Disable color grading"
    )
    video_group.add_argument("--bitrate", help="Video bitrate (default: 5000k)")

    # General
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show pipeline stages without executing"
    )

    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict:
    """Convert CLI arguments into config overrides."""
    overrides = {}

    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.script_backend:
        overrides["script_backend"] = args.script_backend
    if args.tts:
        overrides["tts_backend"] = args.tts
    if args.tts_language:
        overrides["tts_language"] = args.tts_language
    if args.visual_backend:
        overrides["visual_backend"] = args.visual_backend
    if args.format:
        overrides["video_format"] = args.format
    if args.subtitle_backend:
        overrides["subtitle_backend"] = args.subtitle_backend
    if args.no_subtitles:
        overrides["burn_subtitles"] = False
    if args.music:
        overrides["background_music_path"] = args.music
    if args.music_volume is not None:
        overrides["music_volume_db"] = args.music_volume
    if args.fps:
        overrides["fps"] = args.fps
    if args.no_ken_burns:
        overrides["ken_burns"] = False
    if args.no_color_grading:
        overrides["color_grading"] = False
    if args.bitrate:
        overrides["bitrate"] = args.bitrate

    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    overrides = cli_overrides_from_args(args)
    config = load_config(args.config, overrides)

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_stage("generate_script", generate_script)
    pipeline.add_stage("synthesize_speech", synthesize_speech)
    pipeline.add_stage("generate_visuals", generate_visuals)
    pipeline.add_stage("generate_subtitles", generate_subtitles)
    pipeline.add_stage("mix_audio", mix_audio)
    pipeline.add_stage("assemble_video", assemble_video)

    if args.dry_run:
        print("\n Pipeline stages (dry run):")
        for i, (name, _) in enumerate(pipeline._stages, 1):
            print(f"  {i}. {name}")
        print(f"\n  Config: {config}")
        return 0

    # Build context
    ctx = PipelineContext(
        prompt=args.prompt,
        output_dir=Path(config.get("output_dir", "output")),
        config=config,
    )

    # Run pipeline
    print(f"\n{'='*60}")
    print(f"  LineAI Automated Video Generator v1.0")
    print(f"  Topic: {args.prompt}")
    print(f"  Format: {config.get('video_format', 'landscape')}")
    print(f"  TTS: {config.get('tts_backend', 'gtts')}")
    print(f"  Visuals: {config.get('visual_backend', 'programmatic')}")
    print(f"{'='*60}\n")

    results = pipeline.run(ctx)

    # Report
    print(f"\n{'='*60}")
    print("  Pipeline Results:")
    print(f"{'='*60}")

    all_ok = True
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  [{status}] {r.stage_name} ({r.duration_seconds}s)")
        if not r.success:
            print(f"         Error: {r.message}")
            all_ok = False

    if all_ok and ctx.final_video_path:
        print(f"\n  Output: {ctx.final_video_path}")
        print(f"{'='*60}\n")
        return 0
    else:
        print(f"\n  Pipeline failed. Check logs for details.")
        print(f"{'='*60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
