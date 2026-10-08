#!/usr/bin/env python3
"""
voice_generation.py — STORY → single continuous audio file via Edge TTS.

Single responsibility:
    generate_voice(story: dict, output_dir: Path) -> {"audio_path": str, "voice": str, "rate": str, ...}

The function:
- Consumes the story_generation artifact (title + sentences)
- Joins all sentences into ONE continuous narration text
- Sends to Edge TTS as ONE synthesis request
- Saves ONE audio file to the pipeline run's artifact directory
- Returns metadata for downstream stages

No sentence-level timing, captions, scenes, Whiteboard Animator, or Remotion logic.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import edge_tts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_TTS_VOICE = "en-US-AndrewNeural"
EDGE_TTS_RATE = "+20%"
EDGE_TTS_PITCH = "+0Hz"
EDGE_TTS_VOLUME = "+0%"

# Default output filename
AUDIO_FILENAME = "narration.mp3"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_story_for_voice(story: dict) -> None:
    """Raise ValueError/TypeError if story artifact is invalid for voice generation."""
    if not isinstance(story, dict):
        raise TypeError(f"Story must be a dict, got {type(story).__name__}")

    if "sentences" not in story:
        raise ValueError("Story artifact missing required key: 'sentences'")

    sentences = story["sentences"]
    if not isinstance(sentences, list):
        raise TypeError(f"story['sentences'] must be a list, got {type(sentences).__name__}")
    if not sentences:
        raise ValueError("story['sentences'] must be non-empty")

    for i, s in enumerate(sentences):
        if not isinstance(s, str):
            raise TypeError(f"Sentence {i} must be a string, got {type(s).__name__}")
        if not s.strip():
            raise ValueError(f"Sentence {i} is empty or whitespace-only")


def validate_audio_output(audio_path: Path) -> None:
    """Raise ValueError if the generated audio file is missing or unreadable."""
    if not audio_path.exists():
        raise ValueError(f"Generated audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Generated audio path is not a file: {audio_path}")
    size = audio_path.stat().st_size
    if size == 0:
        raise ValueError(f"Generated audio file is empty (0 bytes): {audio_path}")
    if size < 100:
        raise ValueError(f"Generated audio file suspiciously small ({size} bytes): {audio_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _synthesize_to_file(
    text: str,
    output_path: Path,
    voice: str = EDGE_TTS_VOICE,
    rate: str = EDGE_TTS_RATE,
    pitch: str = EDGE_TTS_PITCH,
    volume: str = EDGE_TTS_VOLUME,
) -> None:
    """Internal: synthesize text to a single audio file via Edge TTS."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(str(output_path))


def generate_voice(
    story: dict,
    output_dir: Path,
    *,
    voice: str = EDGE_TTS_VOICE,
    rate: str = EDGE_TTS_RATE,
    pitch: str = EDGE_TTS_PITCH,
    volume: str = EDGE_TTS_VOLUME,
    audio_filename: str = AUDIO_FILENAME,
) -> dict:
    """Generate a single continuous audio file from the story narration.

    Parameters
    ----------
    story : dict
        Output from history_story_generator.generate_history_story:
        {"title": str, "sentences": list[str]}
    output_dir : Path
        Pipeline run directory where the audio file will be saved.
    voice : str
        Edge TTS voice name (default: en-US-AndrewNeural).
    rate : str
        Speaking rate adjustment (default: +20%).
    pitch : str
        Pitch adjustment (default: +0Hz).
    volume : str
        Volume adjustment (default: +0%).
    audio_filename : str
        Output filename (default: narration.mp3).

    Returns
    -------
    dict
        {
            "audio_path": str,       # Relative path from output_dir
            "voice": str,
            "rate": str,
            "pitch": str,
            "volume": str,
            "sentence_count": int,
            "char_count": int
        }

    Raises
    ------
    ValueError
        Story artifact invalid or narration empty.
    RuntimeError
        Edge TTS synthesis fails or output validation fails.
    """
    # --- Input validation ---------------------------------------------------
    validate_story_for_voice(story)

    sentences = story["sentences"]

    # Join sentences preserving exact text and order
    narration_text = " ".join(sentences)

    if not narration_text.strip():
        raise ValueError("Joined narration text is empty")

    # --- Prepare output path ------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / audio_filename

    # --- Single Edge TTS synthesis call -------------------------------------
    try:
        asyncio.run(_synthesize_to_file(
            text=narration_text,
            output_path=audio_path,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        ))
    except Exception as e:
        raise RuntimeError(f"Edge TTS synthesis failed: {e}") from e

    # --- Validate output ----------------------------------------------------
    validate_audio_output(audio_path)

    # --- Return metadata ----------------------------------------------------
    return {
        "audio_path": audio_filename,
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "volume": volume,
        "sentence_count": len(sentences),
        "char_count": len(narration_text),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print(
            "Usage: python voice_generation.py <story_json_file> <output_dir> [--voice VOICE] [--rate RATE]",
            file=sys.stderr,
        )
        sys.exit(1)

    story_path = sys.argv[1]
    output_dir = Path(sys.argv[2])

    voice = EDGE_TTS_VOICE
    rate = EDGE_TTS_RATE

    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        if idx + 1 < len(sys.argv):
            voice = sys.argv[idx + 1]

    if "--rate" in sys.argv:
        idx = sys.argv.index("--rate")
        if idx + 1 < len(sys.argv):
            rate = sys.argv[idx + 1]

    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    result = generate_voice(story, output_dir, voice=voice, rate=rate)
    print(json.dumps(result, indent=2, ensure_ascii=False))