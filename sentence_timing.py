#!/usr/bin/env python3
"""
sentence_timing.py — STORY + WHISPERX TIMING → sentence-level timestamps (frame based).

Single responsibility:
    compute_sentence_timing(story_sentences: list[str], timing_data: dict) -> list[dict]
    where each dict contains:
        {
            "sentence_index": int,
            "text": str,          # original story sentence text (preserved exactly)
            "startFrame": int,    # start frame (inclusive) from first matched WhisperX word
            "endFrame": int,      # end frame (inclusive) from last matched WhisperX word
            "durationInFrames": int  # endFrame - startFrame
        }

The function:
- Takes the authoritative story sentences (from story_generation).
- Takes the WhisperX timing data (from whisperx_timing stage) containing word-level timestamps.
- Uses the same robust normalization and LCS/Needleman-Wunsch alignment as beat_generator.py:
    * lowercase
    * strip punctuation . , ! ? ; : " ' ( ) [ ] { }
    * tolerate punctuation/case differences
    * tolerate WhisperX word splits/merges and extra/missing tokens
    * interpolate/clamp unmatched script words exactly as the existing kinetic_typo_vid logic does.
- Builds a global script-word → WhisperX-word-index mapping for all sentences concatenated.
- For each authoritative story sentence:
    * determines its script-word range
    * maps the first and last script words through the global word map
    * uses WhisperX start from the first mapped word
    * uses WhisperX end from the last mapped word
    * converts seconds to frames using FPS = 30
    * preserves the exact original story sentence text.
- Produces sentence-level timing with sentence_index, text, startFrame, endFrame, durationInFrames.
- Keeps sentences ordered and ensures every story sentence is represented exactly once.
- Does not invent timing, pad the audio duration, or alter the WhisperX timestamps.
- Keeps the implementation small and focused; reuses the proven algorithm rather than creating a new alignment framework.

No story generation, voice generation, WhisperX timing, or controller logic is modified.
"""

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Constants (matching beat_generator.py)
# ---------------------------------------------------------------------------
FPS = 30


def seconds_to_frames(seconds: float) -> int:
    """Convert seconds to frame numbers using FPS."""
    return int(round(seconds * FPS))


def normalize_word(word: str) -> str:
    """Normalize word for comparison: lowercase, remove all punctuation, collapse whitespace.
    Matches the expected test behavior and handles punctuation anywhere in the word.
    """
    # Remove all punctuation
    no_punct = word.translate(str.maketrans("", "", string.punctuation))
    # Lowercase
    lower = no_punct.lower()
    # Collapse whitespace
    collapsed = " ".join(lower.split())
    return collapsed


def build_word_index_map(word_timestamps: List[Dict[str, Any]], script: str) -> List[int]:
    """
    Map each word in the script to its index in word_timestamps using
    a robust sequence alignment (dynamic programming / LCS-based).
    Returns list of word_timestamps indices, one per script word.
    This is a direct copy of the function from beat_generator.py.
    """
    script_words = script.strip().split()
    ts_words = [normalize_word(w["word"]) for w in word_timestamps]
    script_norm = [normalize_word(w) for w in script_words]

    # Use dynamic programming to find the longest common subsequence alignment
    # This is essentially the Needleman-Wunsch algorithm for sequence alignment
    n, m = len(script_norm), len(ts_words)

    # DP table for LCS length
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if script_norm[i - 1] == ts_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find alignment
    mapping = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        if script_norm[i - 1] == ts_words[j - 1]:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Fill in any unmatched script words by interpolating between matched neighbors
    last_matched_ts = -1
    last_matched_script = -1

    for i in range(n):
        if mapping[i] != -1:
            # This word is matched
            last_matched_ts = mapping[i]
            last_matched_script = i
        else:
            # Unmatched - find next matched word to interpolate
            next_matched_ts = -1
            next_matched_script = -1
            for k in range(i + 1, n):
                if mapping[k] != -1:
                    next_matched_ts = mapping[k]
                    next_matched_script = k
                    break

            if last_matched_ts >= 0 and next_matched_ts >= 0:
                # Interpolate proportionally
                script_span = next_matched_script - last_matched_script
                ts_span = next_matched_ts - last_matched_ts
                if script_span > 0:
                    ratio = (i - last_matched_script) / script_span
                    mapping[i] = last_matched_ts + int(round(ratio * ts_span))
                else:
                    mapping[i] = last_matched_ts
            elif last_matched_ts >= 0:
                mapping[i] = last_matched_ts
            elif next_matched_ts >= 0:
                mapping[i] = next_matched_ts
            else:
                mapping[i] = 0

    # Clamp to valid range
    mapping = [max(0, min(val, m - 1)) for val in mapping]

    return mapping


def word_idx_to_frame(word_timestamps: List[Dict[str, Any]], word_idx: int) -> int:
    """Convert word index to frame number using word timestamp start."""
    if not word_timestamps:
        return 0
    if word_idx >= len(word_timestamps):
        word_idx = len(word_timestamps) - 1
    if word_idx < 0:
        word_idx = 0
    return seconds_to_frames(word_timestamps[word_idx]["start"])


def word_idx_to_end_frame(word_timestamps: List[Dict[str, Any]], word_idx: int) -> int:
    """Convert word index to frame number using word timestamp end."""
    if not word_timestamps:
        return 0
    if word_idx >= len(word_timestamps):
        word_idx = len(word_timestamps) - 1
    if word_idx < 0:
        word_idx = 0
    return seconds_to_frames(word_timestamps[word_idx]["end"])


def compute_sentence_timing(story_sentences: List[str], timing_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map story sentences to WhisperX word-level timestamps (frame based).

    Parameters
    ----------
    story_sentences : list[str]
        Original story sentences (authoritative text, to be preserved exactly).
    timing_data : dict
        Output from whisperx_timing stage, containing:
            {
                "audio_path": str,
                "duration": float,
                "words": [{"word": str, "start": float, "end": float}, ...],
                "sentences": [...]  # ignored; we use words only
            }

    Returns
    -------
    list[dict]
        Sentence-level timing dicts as described in the module docstring.

    Notes
    -----
    - If a sentence has zero words after stripping, it is assigned zero duration
      (startFrame == endFrame == 0). This matches the previous behaviour of
      zero-length timing for empty sentences.
    """
    if not story_sentences:
        return []

    # ------------------------------------------------------------------
    # 1. Flatten all sentences into a single word list and record sentence
    #    word ranges (inclusive start/end indices in the flattened list).
    # ------------------------------------------------------------------
    all_script_words: List[str] = []
    sentence_word_ranges: List[tuple[int, int]] = []  # (start_word_idx, end_word_idx) inclusive

    for sentence in story_sentences:
        words = sentence.strip().split()
        start_idx = len(all_script_words)
        all_script_words.extend(words)
        end_idx = len(all_script_words) - 1
        sentence_word_ranges.append((start_idx, end_idx))

    # ------------------------------------------------------------------
    # 2. Build the global script-word → WhisperX-word-index mapping.
    # ------------------------------------------------------------------
    script_string = " ".join(all_script_words)
    word_map = build_word_index_map(timing_data["words"], script_string)
    # word_map length equals len(all_script_words)

    # ------------------------------------------------------------------
    # 3. Compute timing for each sentence.
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    for sent_idx, (start_w, end_w) in enumerate(sentence_word_ranges):
        sentence = story_sentences[sent_idx]  # original text preserved exactly

        # Handle empty sentence (no words)
        if start_w > end_w:
            # Zero duration sentence.
            # We'll set startFrame = endFrame = 0 (could also use previous word's end, but 0 is simple).
            start_frame = 0
            end_frame = 0
            duration_frames = 0
            results.append({
                "sentence_index": sent_idx,
                "text": sentence,
                "startFrame": start_frame,
                "endFrame": end_frame,
                "durationInFrames": duration_frames,
            })
            continue

        # Non-empty sentence: map first and last word indices.
        ts_start_idx = word_map[start_w]
        ts_end_idx = word_map[end_w]

        # Convert to frames.
        start_frame = word_idx_to_frame(timing_data["words"], ts_start_idx)
        end_frame = word_idx_to_end_frame(timing_data["words"], ts_end_idx)
        duration_frames = end_frame - start_frame  # matches beat_generator.py convention

        results.append({
            "sentence_index": sent_idx,
            "text": sentence,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "durationInFrames": duration_frames,
        })

    return results


# ---------------------------------------------------------------------------
# Pipeline stage function (unchanged except for calling the updated compute_*)
# ---------------------------------------------------------------------------
def stage_sentence_timing(config: Any, prev: dict) -> Any:
    """
    Pipeline stage: Sentence timing.

    Expects `prev` to be the story artifact (from story_generation).
    Loads whisperx_timing artifact from disk (config.output_dir / "whisperx_timing.json").
    Returns a PipelineArtifact with sentence-level timing data.
    """
    # Import PipelineArtifact from run_pipeline to avoid circular import issues.
    # If unavailable, define a minimal local version (should not happen in pipeline).
    try:
        from run_pipeline import PipelineArtifact
    except ImportError:
        # Minimal fallback
        class PipelineArtifact:
            def __init__(self, stage_name: str, data: dict, metadata: dict = None):
                self.stage_name = stage_name
                self.data = data
                self.metadata = metadata or {}
            def to_json(self) -> str:
                import json
                return json.dumps(
                    {"stage": self.stage_name, "data": self.data, "meta": self.metadata},
                    indent=2,
                    ensure_ascii=False,
                )

    # Extract story sentences from prev (story artifact)
    if not isinstance(prev, dict) or "sentences" not in prev:
        raise ValueError("Invalid story artifact received: missing 'sentences'")
    story_sentences = prev["sentences"]
    if not isinstance(story_sentences, list):
        raise TypeError("story['sentences'] must be a list")
    if not all(isinstance(s, str) for s in story_sentences):
        raise TypeError("all story sentences must be strings")

    # Load whisperx_timing artifact from disk
    timing_path = config.output_dir / "whisperx_timing.json"
    if not timing_path.exists():
        raise FileNotFoundError(
            f"WhisperX timing artifact not found: {timing_path}. "
            "Ensure the whisperx_timing stage has run and saved intermediate artifacts."
        )
    try:
        with open(timing_path, "r", encoding="utf-8") as f:
            timing_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse whisperx_timing artifact: {e}") from e

    # Unwrap pipeline artifact envelope ({stage, data, meta}) if present.
    # The artifact is saved via PipelineArtifact.to_json() which wraps the
    # raw timing dict inside a "data" key. Unwrap it so we work with the
    # raw timing dict regardless of whether the file was saved with or
    # without the envelope.
    if isinstance(timing_data, dict) and "data" in timing_data and "stage" in timing_data:
        timing_data = timing_data["data"]

    # Validate timing_data has words
    if not isinstance(timing_data, dict) or "words" not in timing_data:
        raise ValueError("WhisperX timing artifact missing 'words' field")
    words = timing_data.get("words", [])
    if not isinstance(words, list):
        raise TypeError("'words' must be a list")
    if not words:
        raise ValueError("WhisperX timing data contains no words")

    # Compute sentence timing
    sentence_timing = compute_sentence_timing(story_sentences, timing_data)

    # Build artifact
    artifact = PipelineArtifact(
        stage_name="sentence_timing",
        data={
            "sentence_timing": sentence_timing,
            # Optionally include audio_path and duration for convenience
            "audio_path": timing_data.get("audio_path"),
            "duration": timing_data.get("duration"),
        },
        metadata={
            "sentence_count": len(sentence_timing),
            "word_count": len(words),
        },
    )
    return artifact


# ---------------------------------------------------------------------------
# CLI for standalone testing (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow running this script directly for testing:
    #   python sentence_timing.py <story_json_file> <whisperx_timing_json_file> <output_dir>
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python sentence_timing.py <story_json_file> <whisperx_timing_json_file> <output_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    story_path = sys.argv[1]
    timing_path = sys.argv[2]
    output_dir = Path(sys.argv[3])

    with open(story_path, "r", encoding="utf-8") as f:
        story_artifact = json.load(f)
    # Expect story artifact to have "sentences" at top level or under "data"?
    # For simplicity, assume the file is the story artifact as stored by pipeline.
    if "data" in story_artifact and isinstance(story_artifact["data"], dict):
        story_sentences = story_artifact["data"].get("sentences", [])
    else:
        story_sentences = story_artifact.get("sentences", [])

    with open(timing_path, "r", encoding="utf-8") as f:
        timing_data = json.load(f)

    # Dummy config with output_dir
    class DummyConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    config = DummyConfig(output_dir)

    try:
        artifact = stage_sentence_timing(config, story_artifact if "data" in story_artifact else {"sentences": story_sentences})
        # Output the sentence timing list
        print(json.dumps(artifact.data["sentence_timing"], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)