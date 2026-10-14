#!/usr/bin/env python3
"""
youtube_metadata.py — STORY → YouTube metadata (title, description, hashtags).

Single responsibility:
    generate_youtube_metadata(story: dict) -> {
        "title": str,
        "description": str,
        "hashtags": list[str]
    }

Rules:
- Title: compelling, ≤100 characters, no hashtags.
- Description: informative, ≤5000 characters, includes call-to-action optional.
- Hashtags: list of strings, each starting with '#', no spaces within tag, max 10 tags.
- Exactly ONE LLM call.
- Uses Gemini 3.1 Flash Lite by default via llm_client.
- No story generation, voice generation, visual planning, timing, asset, Whiteboard, Remotion logic.
"""

import json
import re
from typing import Optional

import llm_client


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOUTUBE_METADATA_MAX_TOKENS = 8192

# Default model (Gemini 3.1 Flash Lite)
DEFAULT_YOUTUBE_MODEL_KEY = "gemini-31-flash-lite"

_YOUTUBE_SYSTEM_PROMPT = """\
You are a YouTube metadata expert for educational historical videos. Given a \
documentary narration (title + ordered sentences), you must produce \
YouTube-friendly metadata: a title, a description, and a list of hashtags.

CRITICAL RULES:
1. TITLE: Must be compelling, accurate, and suitable for YouTube search.
   - Length: ≤100 characters.
   - No hashtags in the title.
   - Should reflect the core historical topic and hook the viewer.
2. DESCRIPTION: Must be informative and engaging.
   - Length: ≤5000 characters.
   - Can include a brief summary of the video content, key takeaways, and a call-to-action (e.g., "Subscribe for more history!").
   - Should not contain misleading information.
3. HASHTAGS: Must be a list of strings, each starting with '#' and containing no spaces.
   - Maximum 10 hashtags.
   - Use relevant historical keywords (e.g., #History, #AncientRome, #MaliEmpire).
   - Do not include duplicate hashtags.
4. OUTPUT FORMAT: Return ONLY valid JSON:
   {
     "title": "...",
     "description": "...",
     "hashtags": ["#History", "#Education", "..."]
   }
5. NO EXTRA FIELDS: No scene_id, description, dimensions, timing, TTS, \
   caption, whiteboard, remotion, or any metadata.
6. PROMPT STYLE: Be specific and follow the constraints exactly.
"""

_YOUTUBE_USER_PROMPT_TEMPLATE = """\
Title: {title}

Narration sentences (0-indexed):
{sentences}

Generate the YouTube metadata now. Return only the requested JSON.
"""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_youtube_metadata_contract(metadata: dict) -> None:
    """Raise ValueError/TypeError if *metadata* violates the output contract."""
    if not isinstance(metadata, dict):
        raise TypeError(f"YouTube metadata must be a dict, got {type(metadata).__name__}")

    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"YouTube metadata must be JSON-serialisable: {exc}") from exc

    required = {"title", "description", "hashtags"}
    got = set(metadata.keys())
    if got != required:
        extra = got - required
        missing = required - got
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected extra keys: {sorted(extra)}")
        raise ValueError(
            f"YouTube metadata keys do not match contract ({'; '.join(parts)}). "
            f"Expected exactly: {sorted(required)}"
        )

    title = metadata["title"]
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title).__name__}")
    if not title.strip():
        raise ValueError("title must be non-empty.")
    if len(title) > 100:
        raise ValueError(f"title must be ≤100 characters, got {len(title)}.")

    description = metadata["description"]
    if not isinstance(description, str):
        raise TypeError(f"description must be str, got {type(description).__name__}")
    if len(description) > 5000:
        raise ValueError(f"description must be ≤5000 characters, got {len(description)}.")

    hashtags = metadata["hashtags"]
    if not isinstance(hashtags, list):
        raise TypeError(f"hashtags must be a list, got {type(hashtags).__name__}")
    if len(hashtags) > 10:
        raise ValueError(f"hashtags must have at most 10 items, got {len(hashtags)}.")
    for i, tag in enumerate(hashtags):
        if not isinstance(tag, str):
            raise TypeError(f"hashtag {i} must be str, got {type(tag).__name__}")
        if not tag.startswith('#'):
            raise ValueError(f"hashtag {i} must start with '#': {tag!r}")
        if ' ' in tag:
            raise ValueError(f"hashtag {i} must not contain spaces: {tag!r}")
        if tag == '#':
            raise ValueError(f"hashtag {i} must have at least one character after '#': {tag!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def generate_youtube_metadata(
    story: dict,
    *,
    model_key: Optional[str] = None,
) -> dict:
    """Generate YouTube metadata from a story artifact.

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
        {"title": str, "description": str, "hashtags": list[str]} — enforced by
        validate_youtube_metadata_contract.

    Raises
    ------
    ValueError
        Story invalid or generated metadata fails validation.
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

    resolved_key = model_key or DEFAULT_YOUTUBE_MODEL_KEY

    # --- Build user prompt ---------------------------------------------------
    numbered = "\n".join(f"  [{i}] {s}" for i, s in enumerate(sentences))
    user_prompt = _YOUTUBE_USER_PROMPT_TEMPLATE.format(
        title=title.strip(),
        sentences=numbered,
    )

    # --- Single LLM call -----------------------------------------------------
    raw = llm_client.call_llm(
        messages=[
            {"role": "system", "content": _YOUTUBE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model_key=resolved_key,
        temperature=0.5,  # Balanced for creativity and consistency
        max_tokens=YOUTUBE_METADATA_MAX_TOKENS,
    )

    # --- Parse ---------------------------------------------------------------
    try:
        parsed = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"YouTube metadata generator returned invalid JSON: {exc}\nRaw output:\n{raw[:500]}"
        ) from exc

    # --- Validate ------------------------------------------------------------
    validate_youtube_metadata_contract(parsed)

    # Ensure hashtags are a list of strings (already validated)
    return {
        "title": parsed["title"].strip(),
        "description": parsed["description"].strip(),
        "hashtags": [tag.strip() for tag in parsed["hashtags"]],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python youtube_metadata.py <story_json_file> [--model MODEL_KEY]",
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

    metadata = generate_youtube_metadata(story, model_key=model_key)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))