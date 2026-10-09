#!/usr/bin/env python3
"""
visual_planning.py — STORY → visual grouping → image generation prompts.

Single responsibility:
    plan_visuals(story: dict) -> {"visual_plan": list[dict]}

Each visual plan entry:
    {
        "sentence_indices": list[int],   # 0-based indices of story sentences covered by this visual
        "image_prompt": str              # Detailed prompt for image generation
    }

Rules:
- Every story sentence must be covered exactly once.
- sentence_indices must be valid, ordered, and partition the full range.
- Every image_prompt must be non-empty.
- Each image_prompt explicitly specifies:
  * 1080x1920 vertical composition
  * Pure white background
  * Historically appropriate visual elements
  * Visual continuity with all narration sentences assigned to that image
- NO scene IDs, descriptions, dimensions fields, timing, TTS, captions,
  Whiteboard Animator, Remotion, or image generation logic.
- Uses Gemini 3.1 Flash Lite by default via llm_client.
- Exactly ONE LLM call.
"""

import json
import re
from typing import Optional

import llm_client


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISUAL_PLAN_MAX_TOKENS = 20192

# Default model (Gemini 3.1 Flash Lite)
DEFAULT_VISUAL_MODEL_KEY = "gemini-31-flash-lite"

_VISUAL_SYSTEM_PROMPT = """\
You are a visual planning expert for historical documentary videos. Given a \
documentary narration (title + ordered sentences), you must group consecutive \
sentences into visual segments and write one detailed image-generation prompt \
per segment.

CRITICAL RULES:
1. GROUPING: Decide dynamically whether consecutive narration sentences should \
   share ONE visual or use SEPARATE visuals. Group sentences that describe the \
   SAME scene, event, location, or continuous action. Split when the narration \
   shifts to a new location, time period, subject, or visual concept.Do not merge\
   morethan 2 sentences togather.
2. COVERAGE: Every sentence must be covered EXACTLY ONCE. The sentence_indices \
   across all visuals must partition the full range [0, N-1] in order.
3. PROMPT CONTENT: Each image_prompt MUST explicitly include:
   - "1080x1920 vertical composition"
   - "pure white background"
   - Historically appropriate visual elements (period-accurate clothing, \
     architecture, tools, vehicles, landscapes, etc.)
   - Visual continuity tying together ALL sentences assigned to that visual \
     (describe how the elements connect/compose into one coherent scene)
   - WHITEBOARD SOURCE IMAGE CONSTRAINTS:
     * The image is a SOURCE IMAGE for an automatic stroke-by-stroke whiteboard drawing animator
     * Clean black/dark marker-style outlines
     * Simple hand-drawn educational illustration style
     * Clear continuous outlines around major objects
     * Distinct, separated visual components
     * Large recognizable shapes and silhouettes
     * Generous white space
     * Simple connected line art
     * Use a small number of flat color accents when useful (prefer simple bounded fills)
     * Keep important visual elements easy to trace independently
     * Make the major outlines visually dominant
     * Minimize unnecessary overlap
     * Simplify details when they do not improve historical understanding
     * The scene should look like something a person could reproduce sequentially with markers on a whiteboard
     * EXPLICITLY PROHIBIT: photorealism, photographic rendering, gradients, realistic lighting, complex shadows, watercolor, painterly texture, paper texture, atmospheric haze, cinematic effects, blur, glow, 3D rendering, dense backgrounds, excessive tiny details, intricate patterns, complex textures, realistic smoke/fire/ocean/terrain textures, unnecessary decorative elements
4. OUTPUT FORMAT: Return ONLY valid JSON:
   {
     "visual_plan": [
       {
         "sentence_indices": [0, 1, 2],
         "image_prompt": "..."
       },
       ...
     ]
   }
5. NO EXTRA FIELDS: No scene_id, description, dimensions, timing, TTS, \
   caption, whiteboard, remotion, or any metadata.
6. PROMPT STYLE: Write for an image generation model. Be specific, visual, \
   and descriptive. Use comma-separated visual details.
"""

_VISUAL_USER_PROMPT_TEMPLATE = """\
Title: {title}

Narration sentences (0-indexed):
{sentences}

Plan the visuals now. Return only the requested JSON.
"""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_visual_plan_contract(plan: dict, num_sentences: int) -> None:
    """Raise ValueError/TypeError if *plan* violates the output contract."""
    if not isinstance(plan, dict):
        raise TypeError(f"Visual plan must be a dict, got {type(plan).__name__}")

    try:
        json.dumps(plan)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Visual plan must be JSON-serialisable: {exc}") from exc

    if "visual_plan" not in plan:
        raise ValueError("Missing required key: 'visual_plan'")
    if len(plan) != 1:
        extra = set(plan.keys()) - {"visual_plan"}
        raise ValueError(f"Unexpected extra keys: {sorted(extra)}")

    visual_plan = plan["visual_plan"]
    if not isinstance(visual_plan, list):
        raise TypeError(f"visual_plan must be a list, got {type(visual_plan).__name__}")
    if not visual_plan:
        raise ValueError("visual_plan must be non-empty")

    # Check coverage: every sentence index 0..num_sentences-1 appears exactly once
    covered = []
    for i, entry in enumerate(visual_plan):
        if not isinstance(entry, dict):
            raise TypeError(f"Visual plan entry {i} must be a dict")
        if set(entry.keys()) != {"sentence_indices", "image_prompt"}:
            raise ValueError(
                f"Entry {i} must have exactly keys "
                f"{{'sentence_indices', 'image_prompt'}}, got {set(entry.keys())}"
            )

        indices = entry["sentence_indices"]
        prompt = entry["image_prompt"]

        if not isinstance(indices, list):
            raise TypeError(f"Entry {i} sentence_indices must be a list")
        if not indices:
            raise ValueError(f"Entry {i} sentence_indices must be non-empty")
        for idx in indices:
            if not isinstance(idx, int):
                raise TypeError(f"Entry {i} sentence_indices must contain integers")
            if idx < 0 or idx >= num_sentences:
                raise ValueError(
                    f"Entry {i} sentence index {idx} out of range [0, {num_sentences - 1}]"
                )

        # Check indices are strictly increasing within entry
        if indices != sorted(indices):
            raise ValueError(f"Entry {i} sentence_indices must be sorted ascending")

        # Check prompt
        if not isinstance(prompt, str):
            raise TypeError(f"Entry {i} image_prompt must be a string")
        if not prompt.strip():
            raise ValueError(f"Entry {i} image_prompt must be non-empty")

        # Check required keywords in prompt
        required_keywords = [
            "1080x1920 vertical composition",
            "pure white background",
        ]
        for kw in required_keywords:
            if kw.lower() not in prompt.lower():
                raise ValueError(
                    f"Entry {i} image_prompt missing required keyword: '{kw}'"
                )

        covered.extend(indices)

    # Verify full coverage exactly once
    if len(covered) != num_sentences:
        raise ValueError(
            f"Total covered indices ({len(covered)}) != number of sentences ({num_sentences})"
        )
    if sorted(covered) != list(range(num_sentences)):
        raise ValueError(
            f"Sentence indices not a perfect partition of [0..{num_sentences - 1}]: {covered}"
        )
    if len(set(covered)) != len(covered):
        raise ValueError("Duplicate sentence indices found")


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json … ``` fences accidentally wrapping a JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_visuals(
    story: dict,
    *,
    model_key: Optional[str] = None,
) -> dict:
    """Generate a visual plan from a story artifact.

    Performs exactly ONE LLM call.

    Parameters
    ----------
    story : dict
        Output from history_story_generator.generate_history_story:
        {"title": str, "sentences": list[str]}
    model_key : str, optional
        Key from llm_client.MODEL_REGISTRY. Defaults to gemini-31-flash-lite.

    Returns
    -------
    dict
        {"visual_plan": list[dict]} — enforced by validate_visual_plan_contract.

    Raises
    ------
    ValueError
        Story invalid or generated plan fails validation.
    TypeError
        Wrong argument types.
    RuntimeError
        LLM call fails or returns unparseable / invalid content.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(story, dict):
        raise TypeError(f"story must be a dict, got {type(story).__name__}")

    title = story.get("title")
    sentences = story.get("sentences")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("story['title'] must be a non-empty string")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("story['sentences'] must be a non-empty list")
    for i, s in enumerate(sentences):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"Sentence {i} must be a non-empty string")

    resolved_key = model_key or DEFAULT_VISUAL_MODEL_KEY

    # --- Build user prompt ---------------------------------------------------
    numbered = "\n".join(f"  [{i}] {s}" for i, s in enumerate(sentences))
    user_prompt = _VISUAL_USER_PROMPT_TEMPLATE.format(
        title=title.strip(),
        sentences=numbered,
    )

    # --- Single LLM call -----------------------------------------------------
    raw = llm_client.call_llm(
        messages=[
            {"role": "system", "content": _VISUAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model_key=resolved_key,
        temperature=0.4,  # Lower temperature for consistent structured output
        max_tokens=VISUAL_PLAN_MAX_TOKENS,
    )

    # --- Parse ---------------------------------------------------------------
    try:
        parsed = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Visual planner returned invalid JSON: {exc}\nRaw output:\n{raw[:500]}"
        ) from exc

    # --- Validate ------------------------------------------------------------
    validate_visual_plan_contract(parsed, len(sentences))

    return parsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python visual_planning.py <story_json_file> [--model MODEL_KEY]",
            file=sys.stderr,
        )
        sys.exit(1)

    story_path = sys.argv[1]
    model_key = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_key = sys.argv[idx + 1]

    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    plan = plan_visuals(story, model_key=model_key)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
