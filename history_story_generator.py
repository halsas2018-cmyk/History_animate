#!/usr/bin/env python3
"""
history_story_generator.py — TOPIC → research → documentary story.

Single responsibility:
    generate_history_story(topic) -> {"title": str, "sentences": list[str]}

The function performs two LLM calls:
  1. Research call  — gather key verified facts / chronology for the topic.
  2. Story call     — turn the research notes into coherent documentary narration.

Output contract (enforced by validate_story_contract):
  {
      "title":     str   — short, compelling, non-sensational title
      "sentences": list[str]  — narration in spoken order; no metadata
  }

No scene, image, project, manifest, Whiteboard, Remotion, timing, audio,
or caption logic lives here.
"""

import json
import re
from typing import Optional

import llm_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_LENGTH_MODES = ("short", "long")

# Generous but not wasteful token budgets.
_RESEARCH_MAX_TOKENS = 10048   # Research notes (both modes share this)
_STORY_MAX_TOKENS_SHORT = 14096
_STORY_MAX_TOKENS_LONG  = 28384

# ---------------------------------------------------------------------------
# Metadata-leakage detection
# ---------------------------------------------------------------------------

_METADATA_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(?:narrator|voiceover|voice\s*over|v\.o\.|speaker(?:\s*\d+)?|host)\s*:\s*", re.IGNORECASE), "narration label"),
    (re.compile(r"(?i)\b(?:sfx|sound effect|bgm)\s*:", re.IGNORECASE), "sound effect label"),
    (re.compile(r"\[\s*(?:sfx|sound effect?|sound|music|audio|bgm)\b[^\]]*\]", re.IGNORECASE), "bracketed sound effect"),
    (re.compile(r"\(\s*(?:sfx|sound effect?|sound|music|audio|bgm)\b[^\)]*\)", re.IGNORECASE), "parenthetical sound effect"),
    (re.compile(r"(?i)\b(?:visual|camera|shot|scene)\s*:"), "visual direction label"),
    (re.compile(r"\[\s*(?:camera|visual|shot|scene|cut to|fade to|fade in|close[- ]up|wide shot|zoom)\b[^\]]*\]", re.IGNORECASE), "bracketed visual direction"),
    (re.compile(r"\(\s*(?:camera|visual|shot|scene|cut to|fade to|fade in|close[- ]up|wide shot|zoom)\b[^\)]*\)", re.IGNORECASE), "parenthetical visual direction"),
    (re.compile(r"^\s*\[.*\]\s*$"), "bracketed direction"),
    (re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]"), "bracketed timestamp"),
    (re.compile(r"\(\s*\d{1,2}:\d{2}(?::\d{2})?\s*\)"), "parenthetical timestamp"),
    (re.compile(r"(?i)\btimestamp\s*:\s*\d"), "timestamp label"),
    (re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*[:-]"), "leading timestamp"),
]

_LEAKAGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\bwhiteboard\b"), "whiteboard reference"),
    (re.compile(r"(?i)\bremotion\b"), "remotion reference"),
    (re.compile(r"(?i)\bscene[_\s]?\d+\b"), "scene reference"),
    (re.compile(r"(?i)\bmanifest\b"), "manifest reference"),
    (re.compile(r"(?i)\bproject\.json\b"), "project.json reference"),
    (re.compile(r"(?i)\bimage\.png\b"), "image.png reference"),
    (re.compile(r"(?i)\bcaption\b"), "caption reference"),
    (re.compile(r"(?i)\baudio\s*file\b"), "audio file reference"),
    (re.compile(r"(?i)\btts\b"), "TTS reference"),
    (re.compile(r"(?i)\bwhisper\b"), "whisper reference"),
]


def find_sentence_metadata_artifacts(sentence: str) -> list[str]:
    """Return list of forbidden metadata artifact descriptions found in *sentence*."""
    found = []
    for pattern, desc in _METADATA_PATTERNS:
        if pattern.search(sentence):
            found.append(desc)
    for pattern, desc in _LEAKAGE_PATTERNS:
        if pattern.search(sentence):
            found.append(desc)
    return found


def _clean_sentence(s: str) -> str:
    """Strip accidental narrator/speaker labels, preserving the spoken narration."""
    s = s.strip()
    s = re.sub(
        r"^\s*(?:narrator|voiceover|voice\s*over|v\.o\.|speaker(?:\s*\d+)?|host)\s*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    return s


# ---------------------------------------------------------------------------
# Output contract validator
# ---------------------------------------------------------------------------

def validate_story_contract(story: dict) -> None:
    """Raise ValueError/TypeError if *story* violates the output contract.

    Contract:
      - dict with exactly the keys {"title", "sentences"}.
      - title: non-empty str.
      - sentences: non-empty list[str], each non-empty, free of metadata leakage.
    """
    if not isinstance(story, dict):
        raise TypeError(f"Story must be a dict, got {type(story).__name__}")

    try:
        json.dumps(story)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"Story must be JSON-serialisable: {exc}") from exc

    required = {"title", "sentences"}
    got = set(story.keys())
    if got != required:
        extra = got - required
        missing = required - got
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected extra keys: {sorted(extra)}")
        raise ValueError(
            f"Story keys do not match contract ({'; '.join(parts)}). "
            f"Expected exactly: {sorted(required)}"
        )

    title = story["title"]
    if not isinstance(title, str):
        raise TypeError(f"title must be str, got {type(title).__name__}")
    if not title.strip():
        raise ValueError("title must be non-empty.")

    sentences = story["sentences"]
    if not isinstance(sentences, list):
        raise TypeError(f"sentences must be a list, got {type(sentences).__name__}")
    if not sentences:
        raise ValueError("sentences must be non-empty.")

    for i, s in enumerate(sentences):
        if not isinstance(s, str):
            raise TypeError(f"Sentence {i + 1} must be str, got {type(s).__name__}")
        if not s.strip():
            raise ValueError(f"Sentence {i + 1} is empty or whitespace-only.")
        artifacts = find_sentence_metadata_artifacts(s)
        if artifacts:
            raise ValueError(
                f"Sentence {i + 1} contains forbidden metadata "
                f"({', '.join(artifacts)}): {s!r}"
            )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT = """\
You are a meticulous historical researcher. Given a history topic, produce \
concise but complete research notes that a documentary narrator can turn into \
accurate narration.

Rules:
- Include only well-attested facts — dates, figures, causes, effects, key actors.
- Order notes chronologically or causally.
- Flag any genuinely uncertain claims with "(uncertain)" or "(estimated)".
- Do NOT invent statistics, quotes, or events.
- Do NOT write narration. Write bullet-point notes only.
- Keep notes under 600 words.

Respond with raw bullet-point text (no JSON).\
"""

_STORY_SYSTEM_SHORT = """\
You write concise, accurate, documentary-style historical narration for short \
educational videos (~100-170 words, 10-20 sentences).

Rules:
- Use ONLY facts from the research notes provided — add nothing unverified.
- Open with a strong hook that establishes historical significance immediately.
- Maintain chronological and causal coherence throughout.
- Write natural spoken sentences; vary length for rhythm.
- Do NOT include: visual directions, camera directions, sound effects, \
timestamps, stage directions, emojis, hashtags, narration labels, WhisperX \
references, scene/image/caption/manifest/project/audio/TTS/Whiteboard/Remotion \
metadata of any kind.
- The "title" field: short (≤35 chars), compelling, no hashtags.

Return ONLY valid JSON:
{
  "title": "...",
  "sentences": ["...", "..."]
}\
"""

_STORY_SYSTEM_LONG = """\
You write long-form, accurate, documentary-style historical narration (~1,000-2,000 \
words, 60-120 sentences).

Rules:
- Use ONLY facts from the research notes provided — add nothing unverified.
- Open with a strong hook. Build through logical historical beats. Close with \
  legacy/significance.
- Maintain strict chronological and causal coherence.
- Write in natural spoken sentences; mix longer descriptive passages with \
  punchy beats for rhythm.
- Do NOT include: visual directions, camera directions, sound effects, \
timestamps, stage directions, emojis, hashtags, narration labels, WhisperX \
references, scene/image/caption/manifest/project/audio/TTS/Whiteboard/Remotion \
metadata of any kind.
- The "title" field: short (≤35 chars), compelling, no hashtags.

Return ONLY valid JSON:
{
  "title": "...",
  "sentences": ["...", "..."]
}\
"""


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


def _validate_raw_story(parsed: object) -> list[str]:
    """Return a list of validation error strings (empty = OK)."""
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return ["Response is not a JSON object."]

    title = parsed.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("Missing or invalid 'title'.")

    sentences = parsed.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        errors.append("'sentences' must be a non-empty array.")
    else:
        for i, s in enumerate(sentences):
            if not isinstance(s, str) or not s.strip():
                errors.append(f"Sentence {i + 1} is empty or not a string.")
            else:
                artifacts = find_sentence_metadata_artifacts(s)
                if artifacts:
                    errors.append(
                        f"Sentence {i + 1} contains forbidden metadata "
                        f"({', '.join(artifacts)})."
                    )
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_history_story(
    topic: str,
    *,
    model_key: Optional[str] = None,
    length_mode: str = "short",
) -> dict:
    """Generate a documentary-style historical story for *topic*.

    Performs exactly two LLM calls:
      1. Research: gather verified facts about the topic.
      2. Story:    turn the research notes into narration JSON.

    Parameters
    ----------
    topic : str
        The historical topic (required, non-empty).
    model_key : str, optional
        Key from ``llm_client.MODEL_REGISTRY``. Defaults to
        ``llm_client.DEFAULT_MODEL_KEY``.
    length_mode : {"short", "long"}
        "short" ≈ 100-170 words; "long" ≈ 1,000-2,000 words.

    Returns
    -------
    dict
        ``{"title": str, "sentences": list[str]}`` — enforced by
        ``validate_story_contract``.

    Raises
    ------
    ValueError
        topic/length_mode invalid, or generated story fails validation.
    TypeError
        Wrong argument types.
    RuntimeError
        LLM call fails or returns unparseable / invalid content.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(
            f"topic is required and must be a non-empty string, got {topic!r}"
        )
    topic = topic.strip()

    if model_key is not None and not isinstance(model_key, str):
        raise TypeError(f"model_key must be str, got {type(model_key).__name__}")
    resolved_key = model_key or llm_client.DEFAULT_MODEL_KEY

    if (
        not isinstance(length_mode, str)
        or length_mode.strip().lower() not in ALLOWED_LENGTH_MODES
    ):
        raise ValueError(
            f"length_mode must be one of {ALLOWED_LENGTH_MODES!r}, got {length_mode!r}"
        )
    length_mode = length_mode.strip().lower()

    story_system = _STORY_SYSTEM_LONG if length_mode == "long" else _STORY_SYSTEM_SHORT
    story_max_tokens = (
        _STORY_MAX_TOKENS_LONG if length_mode == "long" else _STORY_MAX_TOKENS_SHORT
    )
    # Higher temperature for short punchy stories; lower for long analytical ones.
    temperature = 0.5 if length_mode == "long" else 0.7

    # --- Call 1: Research ---------------------------------------------------
    research_notes = llm_client.call_llm(
        messages=[
            {"role": "system", "content": _RESEARCH_PROMPT},
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        model_key=resolved_key,
        temperature=0.3,          # Low temperature for factual research
        max_tokens=_RESEARCH_MAX_TOKENS,
    )

    # --- Call 2: Story generation -------------------------------------------
    story_user_prompt = (
        f"Topic: {topic}\n\n"
        f"Research notes:\n{research_notes}\n\n"
        "Write the narration now and return only the requested JSON."
    )

    raw_story = llm_client.call_llm(
        messages=[
            {"role": "system", "content": story_system},
            {"role": "user", "content": story_user_prompt},
        ],
        model_key=resolved_key,
        temperature=temperature,
        max_tokens=story_max_tokens,
    )

    # --- Parse ---------------------------------------------------------------
    try:
        parsed = json.loads(_strip_markdown_fences(raw_story))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Story generator returned invalid JSON: {exc}\nRaw output:\n{raw_story[:500]}"
        ) from exc

    # Clean narrator labels that occasionally leak through
    if isinstance(parsed, dict) and isinstance(parsed.get("sentences"), list):
        parsed["sentences"] = [
            _clean_sentence(s) if isinstance(s, str) else s
            for s in parsed["sentences"]
        ]

    # Strip the "topic" key if the model echoes it back (not part of contract)
    if isinstance(parsed, dict) and "topic" in parsed:
        del parsed["topic"]

    errors = _validate_raw_story(parsed)
    if errors:
        raise RuntimeError(
            "Generated story failed validation:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    story: dict = {
        "title": parsed["title"].strip(),
        "sentences": [s.strip() for s in parsed["sentences"]],
    }

    # Final strict enforcement
    validate_story_contract(story)

    return story


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python history_story_generator.py <topic> "
            "[--model MODEL_KEY] [--length short|long]"
        )
        print(
            'Example: python history_story_generator.py '
            '"How Kano became a centre of trans-Saharan trade"'
        )
        sys.exit(1)

    _topic = sys.argv[1]
    _model_key: Optional[str] = None
    _length_mode = "short"

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            _model_key = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--length" and i + 1 < len(sys.argv):
            _length_mode = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    _story = generate_history_story(
        _topic, model_key=_model_key, length_mode=_length_mode
    )
    print(json.dumps(_story, indent=2, ensure_ascii=False))
