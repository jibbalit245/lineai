"""
Script generation module.

Generates a narration script broken into timed scenes from a user prompt.
Supports multiple backends:
  - Claude (Anthropic API)
  - OpenAI (GPT-4)
  - Groq (llama / mixtral)
  - Offline fallback (template-based)

Each scene has: narration text, visual description, and suggested duration.
"""

import json
import logging
import os
from typing import Any

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a world-class video script writer. Given a topic, produce a JSON array
of scenes for a short-form video. Each scene is an object with these fields:

- "narration": the spoken text for this scene (1-3 sentences, vivid and engaging)
- "visual": a detailed description of what the viewer should see (used for image generation)
- "duration_hint": suggested duration in seconds (3-8)

Rules:
- Total video should be 30-90 seconds depending on topic complexity.
- Use a hook in scene 1 to grab attention immediately.
- End with a strong call to action or memorable closing thought.
- Write in a conversational, energetic tone.
- Return ONLY the JSON array, no other text.
"""


def generate_script(ctx: PipelineContext) -> None:
    """Generate a scene-by-scene script and store it in ctx.scenes."""
    backend = ctx.config.get("script_backend", "offline")
    prompt = ctx.prompt or ctx.script_text

    if not prompt:
        raise ValueError("No prompt or script_text provided in context")

    if backend == "anthropic":
        scenes = _generate_anthropic(prompt, ctx.config)
    elif backend == "openai":
        scenes = _generate_openai(prompt, ctx.config)
    elif backend == "groq":
        scenes = _generate_groq(prompt, ctx.config)
    else:
        scenes = _generate_offline(prompt)

    ctx.scenes = scenes
    ctx.script_text = " ".join(s["narration"] for s in scenes)
    logger.info("Generated %d scenes totaling ~%ds", len(scenes),
                sum(s.get("duration_hint", 5) for s in scenes))


def _generate_anthropic(prompt: str, config: dict) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    )
    message = client.messages.create(
        model=config.get("anthropic_model", "claude-sonnet-4-20250514"),
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Create a video script about: {prompt}"}],
    )
    return _parse_scenes(message.content[0].text)


def _generate_openai(prompt: str, config: dict) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        model=config.get("openai_model", "gpt-4o"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a video script about: {prompt}"},
        ],
        max_tokens=2048,
    )
    return _parse_scenes(response.choices[0].message.content)


def _generate_groq(prompt: str, config: dict) -> list[dict[str, Any]]:
    from groq import Groq

    client = Groq(
        api_key=config.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    )
    response = client.chat.completions.create(
        model=config.get("groq_model", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a video script about: {prompt}"},
        ],
        max_tokens=2048,
    )
    return _parse_scenes(response.choices[0].message.content)


def _generate_offline(prompt: str) -> list[dict[str, Any]]:
    """Template-based fallback when no API keys are available."""
    words = prompt.split()
    topic = " ".join(words[:10])

    return [
        {
            "narration": f"Have you ever wondered about {topic}? Let me break it down for you.",
            "visual": f"Bold title text '{topic}' appearing with dynamic zoom effect on dark background",
            "duration_hint": 5,
        },
        {
            "narration": f"Here's what makes {topic} absolutely fascinating.",
            "visual": f"Split screen showing multiple perspectives related to {topic}",
            "duration_hint": 6,
        },
        {
            "narration": f"The key insight is this: {topic} changes everything about how we think.",
            "visual": f"Dramatic close-up with glowing highlights illustrating the core concept of {topic}",
            "duration_hint": 6,
        },
        {
            "narration": "And that's just the beginning. Follow for more deep dives like this.",
            "visual": "Call to action screen with subscribe button animation and channel branding",
            "duration_hint": 4,
        },
    ]


def _parse_scenes(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from LLM response, handling markdown fences."""
    text = raw.strip()
    if "```" in text:
        # Extract content between first ``` and last ```
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    scenes = json.loads(text)
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise ValueError("LLM returned invalid scene format")

    for scene in scenes:
        if "narration" not in scene or "visual" not in scene:
            raise ValueError(f"Scene missing required fields: {scene}")
        scene.setdefault("duration_hint", 5)

    return scenes
