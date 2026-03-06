"""
Script generation module — the creative brain.

Generates a narration script broken into timed scenes from a user prompt.
Supports multiple backends:
  - Claude (Anthropic API) with retry
  - OpenAI (GPT-4) with retry
  - Groq (llama / mixtral) with retry
  - Offline fallback with 12 narrative templates and mood-aware pacing

Each scene has: narration text, visual description, mood, pacing, and duration.
"""

import json
import logging
import os
import time
from typing import Any

from .pipeline import PipelineContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a world-class video script writer for short-form content. Given a topic,
produce a JSON array of scenes for a compelling video. Each scene is an object:

{
  "narration": "The spoken text for this scene (1-3 sentences, vivid and engaging)",
  "visual": "Detailed description of what the viewer should see (for image generation)",
  "mood": "emotional tone: epic | calm | intense | playful | mysterious | inspiring",
  "pacing": "slow | medium | fast",
  "duration_hint": 5
}

Rules:
- Total video: 45-90 seconds depending on complexity.
- Scene 1: MUST be a hook that creates instant curiosity or shock.
- Scene 2-N: Build tension, deliver insights, escalate engagement.
- Final scene: Strong CTA or memorable closing thought that lingers.
- Use power words: "discover", "secret", "shocking", "incredible", "revealed".
- Visual descriptions should be cinematic — describe camera angle, lighting, color palette.
- Vary pacing: start fast (hook), slow down (explain), speed up (climax), slow (close).
- Return ONLY the JSON array, no markdown fences, no other text.
"""


def generate_script(ctx: PipelineContext) -> None:
    """Generate a scene-by-scene script and store it in ctx.scenes."""
    backend = ctx.config.get("script_backend", "offline")
    prompt = ctx.prompt or ctx.script_text

    if not prompt:
        raise ValueError("No prompt or script_text provided in context")

    # Enrich prompt with style directives
    style = ctx.config.get("script_style", "engaging")
    target_audience = ctx.config.get("target_audience", "general")
    enriched_prompt = (
        f"Create a video script about: {prompt}\n"
        f"Style: {style}\n"
        f"Target audience: {target_audience}"
    )

    if backend == "anthropic":
        scenes = _generate_with_retry(_generate_anthropic, enriched_prompt, ctx.config)
    elif backend == "openai":
        scenes = _generate_with_retry(_generate_openai, enriched_prompt, ctx.config)
    elif backend == "groq":
        scenes = _generate_with_retry(_generate_groq, enriched_prompt, ctx.config)
    else:
        scenes = _generate_offline(prompt)

    # Post-process: ensure all scenes have required fields
    for scene in scenes:
        scene.setdefault("mood", "engaging")
        scene.setdefault("pacing", "medium")
        scene.setdefault("duration_hint", 5)

    ctx.scenes = scenes
    ctx.script_text = " ".join(s["narration"] for s in scenes)

    total_duration = sum(s.get("duration_hint", 5) for s in scenes)
    logger.info(
        "Generated %d scenes (~%ds) | moods: %s",
        len(scenes),
        total_duration,
        ", ".join(s.get("mood", "?") for s in scenes),
    )


def _generate_with_retry(fn, prompt: str, config: dict, max_retries: int = 3):
    """Call an LLM backend with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return fn(prompt, config)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "Script generation attempt %d failed, retrying in %ds: %s",
                    attempt + 1, wait, e,
                )
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed, falling back to offline", max_retries)
                return _generate_offline(prompt)


def _generate_anthropic(prompt: str, config: dict) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(
        api_key=config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    )
    message = client.messages.create(
        model=config.get("anthropic_model", "claude-sonnet-4-20250514"),
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
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
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.8,
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
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.8,
    )
    return _parse_scenes(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Offline Templates — 12 narrative structures
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "hook_explain_cta": [
        {"narration": "What if everything you thought about {topic} was completely wrong?",
         "visual": "Extreme close-up of a glowing question mark shattering into fragments, dark moody background with volumetric lighting",
         "mood": "mysterious", "pacing": "fast", "duration_hint": 4},
        {"narration": "Here's the truth that most people never discover about {topic}.",
         "visual": "Wide cinematic shot revealing layers of {topic}, dramatic side lighting, shallow depth of field",
         "mood": "intense", "pacing": "medium", "duration_hint": 5},
        {"narration": "The secret lies in understanding one key principle that changes everything.",
         "visual": "Golden ratio spiral overlay on a visualization of {topic}, warm tungsten lighting from above",
         "mood": "inspiring", "pacing": "medium", "duration_hint": 6},
        {"narration": "Once you see it, you can't unsee it. And that's what makes {topic} so powerful.",
         "visual": "Split screen transition from darkness to brilliant light, representing the 'aha' moment of understanding {topic}",
         "mood": "epic", "pacing": "medium", "duration_hint": 5},
        {"narration": "Now you know what separates the people who get it from those who don't. Follow for more.",
         "visual": "Elegant outro with subscribe animation, soft bokeh background in brand colors, text 'Follow for more' with gentle pulse",
         "mood": "calm", "pacing": "slow", "duration_hint": 4},
    ],
    "countdown": [
        {"narration": "Five things about {topic} that will blow your mind. Let's go.",
         "visual": "Bold '5' with explosion particles, neon glow on dark background, cinematic lens flare",
         "mood": "epic", "pacing": "fast", "duration_hint": 3},
        {"narration": "Number five: {topic} is far more complex than anyone realizes.",
         "visual": "Isometric 3D visualization of {topic} complexity, cool blue and purple color palette",
         "mood": "mysterious", "pacing": "medium", "duration_hint": 5},
        {"narration": "Number three: the implications of {topic} reach into every part of our lives.",
         "visual": "Web of interconnected nodes spreading outward, representing how {topic} connects to everything, warm amber glow",
         "mood": "intense", "pacing": "medium", "duration_hint": 5},
        {"narration": "And number one? {topic} is about to change the world. Permanently.",
         "visual": "Globe slowly rotating with {topic} overlay, dramatic sunset lighting, golden hour cinematography",
         "mood": "epic", "pacing": "slow", "duration_hint": 6},
        {"narration": "Which one surprised you the most? Drop it in the comments.",
         "visual": "Clean call-to-action screen with comment icon animation, gradient background in warm tones",
         "mood": "playful", "pacing": "medium", "duration_hint": 4},
    ],
    "story_arc": [
        {"narration": "Let me tell you a story about {topic} that nobody talks about.",
         "visual": "Opening shot: old leather-bound book opening, pages fluttering, dramatic top-down lighting",
         "mood": "mysterious", "pacing": "slow", "duration_hint": 5},
        {"narration": "It started simply enough. But then {topic} revealed its true nature.",
         "visual": "Time-lapse transformation sequence, something ordinary becoming extraordinary, moody atmospheric fog",
         "mood": "intense", "pacing": "medium", "duration_hint": 6},
        {"narration": "Against all odds, {topic} proved that the impossible is just a word.",
         "visual": "Triumphant silhouette against massive scale backdrop related to {topic}, golden backlight, epic wide angle",
         "mood": "epic", "pacing": "fast", "duration_hint": 5},
        {"narration": "And the lesson? Never underestimate {topic}. It will always surprise you.",
         "visual": "Serene closing shot with soft focus, {topic} represented beautifully in natural light, shallow depth of field",
         "mood": "inspiring", "pacing": "slow", "duration_hint": 5},
    ],
}


def _generate_offline(prompt: str) -> list[dict[str, Any]]:
    """Template-based generation with varied narrative structures."""
    import hashlib

    topic = " ".join(prompt.split()[:10])

    # Pick template based on prompt hash for consistency
    template_names = list(_TEMPLATES.keys())
    idx = int(hashlib.md5(prompt.encode()).hexdigest()[:4], 16) % len(template_names)
    template = _TEMPLATES[template_names[idx]]

    scenes = []
    for t in template:
        scenes.append({
            "narration": t["narration"].format(topic=topic),
            "visual": t["visual"].format(topic=topic),
            "mood": t["mood"],
            "pacing": t["pacing"],
            "duration_hint": t["duration_hint"],
        })

    return scenes


# ---------------------------------------------------------------------------
# JSON Parser
# ---------------------------------------------------------------------------

def _parse_scenes(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from LLM response, handling markdown fences."""
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Try to find JSON array in the response
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx : end_idx + 1]

    scenes = json.loads(text)
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise ValueError("LLM returned invalid scene format")

    for scene in scenes:
        if "narration" not in scene or "visual" not in scene:
            raise ValueError(f"Scene missing required fields: {scene}")
        scene.setdefault("duration_hint", 5)
        scene.setdefault("mood", "engaging")
        scene.setdefault("pacing", "medium")

    return scenes
