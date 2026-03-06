"""
Core pipeline engine for the automated video generator.

Orchestrates all stages: script generation -> TTS -> visual assets ->
subtitle generation -> audio mixing -> video assembly -> final render.

Each stage is a self-contained module that receives a shared context dict
and writes its outputs back into it. Stages can be skipped, swapped, or
extended without touching the rest of the pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state that flows through every stage of the pipeline."""

    # --- Inputs ---
    prompt: str = ""
    script_text: str = ""
    output_dir: Path = field(default_factory=lambda: Path("output"))
    config: dict = field(default_factory=dict)

    # --- Intermediate artifacts ---
    scenes: list[dict[str, Any]] = field(default_factory=list)
    audio_narration_path: Path | None = None
    audio_duration: float = 0.0
    image_paths: list[Path] = field(default_factory=list)
    subtitle_path: Path | None = None
    subtitle_segments: list[dict[str, Any]] = field(default_factory=list)
    background_music_path: Path | None = None
    mixed_audio_path: Path | None = None

    # --- Final output ---
    final_video_path: Path | None = None

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StageResult:
    """Result returned by each pipeline stage."""

    success: bool
    stage_name: str
    duration_seconds: float
    message: str = ""


class Pipeline:
    """
    Executes an ordered list of stages against a shared PipelineContext.

    Usage:
        pipeline = Pipeline()
        pipeline.add_stage("generate_script", script_gen_fn)
        pipeline.add_stage("synthesize_speech", tts_fn)
        ...
        results = pipeline.run(context)
    """

    def __init__(self) -> None:
        self._stages: list[tuple[str, Callable[[PipelineContext], None]]] = []

    def add_stage(
        self, name: str, fn: Callable[[PipelineContext], None]
    ) -> "Pipeline":
        self._stages.append((name, fn))
        return self

    def run(self, ctx: PipelineContext) -> list[StageResult]:
        ctx.ensure_output_dir()
        results: list[StageResult] = []

        logger.info("Pipeline started with %d stages", len(self._stages))

        for name, fn in self._stages:
            logger.info(">>> Stage: %s", name)
            t0 = time.perf_counter()
            try:
                fn(ctx)
                elapsed = time.perf_counter() - t0
                result = StageResult(
                    success=True,
                    stage_name=name,
                    duration_seconds=round(elapsed, 2),
                )
                logger.info(
                    "<<< Stage %s completed in %.2fs", name, elapsed
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                result = StageResult(
                    success=False,
                    stage_name=name,
                    duration_seconds=round(elapsed, 2),
                    message=str(exc),
                )
                logger.error(
                    "<<< Stage %s FAILED after %.2fs: %s",
                    name,
                    elapsed,
                    exc,
                    exc_info=True,
                )
                results.append(result)
                break

            results.append(result)

        return results
