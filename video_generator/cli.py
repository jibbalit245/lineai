#!/usr/bin/env python3
"""
Command-line interface for the LineAI Automated Video Generator.

Usage:
    python -m video_generator "Your video topic here"
    python -m video_generator --config config.json "Explain quantum computing"
    python -m video_generator --format portrait --tts elevenlabs "Top 5 life hacks"
    python -m video_generator --batch topics.txt --format square

Full pipeline: prompt -> script -> speech -> visuals -> subtitles -> mix -> render
"""

import argparse
import json
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
        description="LineAI Automated Video Generator — turn any prompt into a polished, cinematic video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s "The history of the internet"
  %(prog)s --format portrait --tts gtts "5 Python tips"
  %(prog)s --config my_config.json --script-backend anthropic "AI explained"
  %(prog)s --visual-backend dalle --film-grain --letterbox "Space exploration"
  %(prog)s --batch topics.txt --format square --output-dir batch_output
  %(prog)s --intro-text "MY CHANNEL" --outro-text "SUBSCRIBE" "Crypto 101"
  %(prog)s --transition-style wipe_left --vignette --karaoke "Jazz history"
        """,
    )

    parser.add_argument("prompt", nargs="?", help="The topic or script for the video")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("-c", "--config", default=None, help="Config file path (JSON or YAML)")

    # Script options
    sg = parser.add_argument_group("Script Generation")
    sg.add_argument("--script-backend", choices=["offline", "anthropic", "openai", "groq"])
    sg.add_argument("--script-style", help="Script style: engaging, educational, dramatic, humorous")
    sg.add_argument("--target-audience", help="Target audience: general, technical, kids, professionals")

    # TTS options
    tg = parser.add_argument_group("Text-to-Speech")
    tg.add_argument("--tts", choices=["gtts", "pyttsx3", "elevenlabs"])
    tg.add_argument("--tts-language", help="TTS language code (default: en)")

    # Visual options
    vg = parser.add_argument_group("Visuals")
    vg.add_argument("--visual-backend", choices=["programmatic", "dalle", "pexels"])
    vg.add_argument("--format", choices=["landscape", "portrait", "square"])

    # Subtitle options
    subg = parser.add_argument_group("Subtitles")
    subg.add_argument("--subtitle-backend", choices=["estimated", "whisper"])
    subg.add_argument("--no-subtitles", action="store_true", help="Disable subtitle burning")
    subg.add_argument("--karaoke", action="store_true", help="Enable karaoke word-by-word highlighting")
    subg.add_argument("--karaoke-style", choices=["smooth", "instant", "glow"], help="Karaoke highlight style")

    # Audio options
    ag = parser.add_argument_group("Audio")
    ag.add_argument("--music", help="Path to background music file")
    ag.add_argument("--music-volume", type=int, help="Background music volume in dB (default: -18)")

    # Video effects
    eg = parser.add_argument_group("Video Effects")
    eg.add_argument("--no-ken-burns", action="store_true", help="Disable Ken Burns effect")
    eg.add_argument("--no-color-grading", action="store_true", help="Disable color grading")
    eg.add_argument("--no-vignette", action="store_true", help="Disable vignette effect")
    eg.add_argument("--film-grain", action="store_true", help="Enable film grain texture")
    eg.add_argument("--letterbox", action="store_true", help="Enable cinematic letterbox bars")
    eg.add_argument("--letterbox-ratio", type=float, help="Letterbox aspect ratio (default: 2.39)")
    eg.add_argument("--transition-style",
                     choices=["mixed", "crossfade", "wipe_left", "wipe_right", "slide_up", "zoom_through", "dissolve"],
                     help="Transition style between scenes")
    eg.add_argument("--tint", choices=["warm", "cool", "neutral"], help="Color tint mode")

    # Intro/Outro
    iog = parser.add_argument_group("Intro / Outro")
    iog.add_argument("--intro-text", help="Intro card title text")
    iog.add_argument("--intro-subtitle", help="Intro card subtitle")
    iog.add_argument("--outro-text", help="Outro card title text")
    iog.add_argument("--outro-subtitle", help="Outro card subtitle")

    # Video output
    vog = parser.add_argument_group("Video Output")
    vog.add_argument("--fps", type=int, help="Frames per second (default: 30)")
    vog.add_argument("--bitrate", help="Video bitrate (default: 5000k)")

    # Batch mode
    bg = parser.add_argument_group("Batch Mode")
    bg.add_argument("--batch", help="File with one topic per line for batch generation")

    # General
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Show stages without executing")

    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict:
    """Convert CLI arguments into config overrides."""
    overrides = {}

    mapping = {
        "output_dir": "output_dir",
        "script_backend": "script_backend",
        "script_style": "script_style",
        "target_audience": "target_audience",
        "tts": "tts_backend",
        "tts_language": "tts_language",
        "visual_backend": "visual_backend",
        "format": "video_format",
        "subtitle_backend": "subtitle_backend",
        "karaoke_style": "subtitle_karaoke_style",
        "music": "background_music_path",
        "music_volume": "music_volume_db",
        "fps": "fps",
        "bitrate": "bitrate",
        "letterbox_ratio": "letterbox_ratio",
        "transition_style": "transition_style",
        "tint": "tint_mode",
        "intro_text": "intro_text",
        "intro_subtitle": "intro_subtitle",
        "outro_text": "outro_text",
        "outro_subtitle": "outro_subtitle",
    }

    for arg_name, config_key in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[config_key] = val

    # Boolean flags
    if args.no_subtitles:
        overrides["burn_subtitles"] = False
    if args.karaoke:
        overrides["subtitle_karaoke"] = True
    if args.no_ken_burns:
        overrides["ken_burns"] = False
    if args.no_color_grading:
        overrides["color_grading"] = False
    if args.no_vignette:
        overrides["vignette"] = False
    if args.film_grain:
        overrides["film_grain"] = True
    if args.letterbox:
        overrides["letterbox"] = True

    return overrides


def _build_pipeline() -> Pipeline:
    """Build the standard video generation pipeline."""
    pipeline = Pipeline()
    pipeline.add_stage("generate_script", generate_script)
    pipeline.add_stage("synthesize_speech", synthesize_speech)
    pipeline.add_stage("generate_visuals", generate_visuals)
    pipeline.add_stage("generate_subtitles", generate_subtitles)
    pipeline.add_stage("mix_audio", mix_audio)
    pipeline.add_stage("assemble_video", assemble_video)
    return pipeline


def _print_banner(prompt: str, config: dict) -> None:
    effects = []
    if config.get("ken_burns", True):
        effects.append("Ken Burns")
    if config.get("color_grading", True):
        effects.append("Color Grade")
    if config.get("vignette", True):
        effects.append("Vignette")
    if config.get("film_grain", False):
        effects.append("Film Grain")
    if config.get("letterbox", False):
        effects.append("Letterbox")

    print(f"\n{'='*64}")
    print(f"  LineAI Automated Video Generator v2.0")
    print(f"{'='*64}")
    print(f"  Topic:       {prompt}")
    print(f"  Format:      {config.get('video_format', 'landscape')}")
    print(f"  TTS:         {config.get('tts_backend', 'gtts')}")
    print(f"  Visuals:     {config.get('visual_backend', 'programmatic')}")
    print(f"  Script:      {config.get('script_backend', 'offline')}")
    print(f"  Transitions: {config.get('transition_style', 'mixed')}")
    print(f"  Effects:     {', '.join(effects) or 'None'}")
    if config.get("subtitle_karaoke"):
        print(f"  Subtitles:   Karaoke ({config.get('subtitle_karaoke_style', 'smooth')})")
    print(f"{'='*64}\n")


def _run_single(prompt: str, config: dict, output_dir: str) -> int:
    """Run the pipeline for a single prompt."""
    config["output_dir"] = output_dir

    _print_banner(prompt, config)

    pipeline = _build_pipeline()
    ctx = PipelineContext(
        prompt=prompt,
        output_dir=Path(output_dir),
        config=config,
    )

    results = pipeline.run(ctx)

    # Report
    print(f"\n{'='*64}")
    print("  Pipeline Results:")
    print(f"{'='*64}")

    all_ok = True
    total_time = 0.0
    for r in results:
        status = " OK " if r.success else "FAIL"
        print(f"  [{status}] {r.stage_name} ({r.duration_seconds}s)")
        if not r.success:
            print(f"          Error: {r.message}")
            all_ok = False
        total_time += r.duration_seconds

    print(f"  {'─'*56}")
    print(f"  Total: {total_time:.1f}s")

    if all_ok and ctx.final_video_path:
        print(f"\n  Output: {ctx.final_video_path}")
        thumb = ctx.output_dir / "thumbnail.png"
        if thumb.exists():
            print(f"  Thumbnail: {thumb}")
        print(f"{'='*64}\n")
        return 0
    else:
        print(f"\n  Pipeline failed. Check logs for details.")
        print(f"{'='*64}\n")
        return 1


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

    # Dry run
    if args.dry_run:
        pipeline = _build_pipeline()
        print("\n Pipeline stages (dry run):")
        for i, (name, _) in enumerate(pipeline._stages, 1):
            print(f"  {i}. {name}")
        print(f"\n  Config: {json.dumps(config, indent=2, default=str)}")
        return 0

    # Batch mode
    if args.batch:
        return _run_batch(args.batch, config, args.output_dir)

    # Single mode
    if not args.prompt:
        parser.error("prompt is required (or use --batch for batch mode)")

    return _run_single(args.prompt, config, args.output_dir)


def _run_batch(batch_file: str, config: dict, base_output_dir: str) -> int:
    """Run the pipeline for each line in the batch file."""
    import json

    path = Path(batch_file)
    if not path.exists():
        print(f"Batch file not found: {batch_file}")
        return 1

    topics = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not topics:
        print("Batch file is empty")
        return 1

    print(f"\n Batch mode: {len(topics)} videos to generate\n")

    results_summary = []
    for i, topic in enumerate(topics, 1):
        print(f"\n{'#'*64}")
        print(f"  Video {i}/{len(topics)}: {topic}")
        print(f"{'#'*64}")

        # Create unique output dir per video
        safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:50].strip().replace(" ", "_")
        output_dir = str(Path(base_output_dir) / f"{i:03d}_{safe_name}")

        exit_code = _run_single(topic, config.copy(), output_dir)
        results_summary.append({"topic": topic, "success": exit_code == 0, "output_dir": output_dir})

    # Final batch report
    print(f"\n{'='*64}")
    print(f"  Batch Complete: {sum(1 for r in results_summary if r['success'])}/{len(topics)} succeeded")
    print(f"{'='*64}")
    for r in results_summary:
        status = " OK " if r["success"] else "FAIL"
        print(f"  [{status}] {r['topic'][:50]}")
    print()

    return 0 if all(r["success"] for r in results_summary) else 1


if __name__ == "__main__":
    sys.exit(main())
