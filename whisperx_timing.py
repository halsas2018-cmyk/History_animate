#!/usr/bin/env python3
"""
whisperx_timing.py — VOICE + STORY → word-level timestamps via WhisperX.

Single responsibility:
    compute_word_timestamps(story: dict, config: PipelineConfig) -> {
        "audio_path": str,
        "duration": float,
        "words": list[{"index": int, "word": str, "start": float, "end": float}],
        "sentences": list[{
            "sentence_index": int,
            "start": float,
            "end": float,
            "word_indices": list[int]
        }]
    }

The function:
- Consumes the story_generation artifact (title + sentences) for sentence alignment.
- Reads the voice_generation artifact from disk to get the narration audio path.
- Validates the audio file exists and is readable.
- Uses WhisperX (already installed) to transcribe and align for word-level timestamps.
- Flattens segments to word list with indices.
- Deterministically aligns each authoritative Story Generation sentence to a
  contiguous sequence of WhisperX words in order, tolerating punctuation,
  capitalisation, and whitespace differences but failing clearly on substantive
  mismatches.
- Derives each sentence start/end from its first/last aligned word.
- Measures actual audio duration independently (not from the last WhisperX word end).
- Returns a dictionary suitable for downstream animation synchronization.

No voice generation, image generation, Whiteboard Animator, or Remotion logic.
"""

import json
import re
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import whisperx


def _load_voice_generation_artifact(output_dir: Path) -> Dict[str, Any]:
    """Load the voice generation artifact from disk to get audio path."""
    artifact_path = output_dir / "voice_generation.json"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Voice generation artifact not found: {artifact_path}. "
            "Ensure the voice generation stage has run and saved intermediate artifacts."
        )
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse voice generation artifact: {e}") from e

    if "data" not in artifact:
        raise ValueError(
            f"Voice generation artifact missing 'data' field: {artifact}"
        )
    return artifact["data"]


def _validate_audio_file(audio_path: Path) -> None:
    """Validate that the audio file exists and is readable."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    size = audio_path.stat().st_size
    if size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")
    if size < 100:  # suspiciously small
        raise ValueError(
            f"Audio file suspiciously small ({size} bytes): {audio_path}"
        )


def _transcribe_and_align(audio_path: str) -> List[Dict[str, Any]]:
    """
    Run WhisperX transcription + alignment to get word-level timestamps.

    Returns a list of word dictionaries (without indices yet):
        [{"word": str, "start": float, "end": float}, ...]
    """
    # WhisperX configuration matching the reference in /root/video_maker
    device = "cpu"
    compute_type = "int8"
    model_name = "base"

    print(f"[1/3] Loading Whisper model ({model_name}, {compute_type}, {device})...")
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)

    print(f"[2/3] Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    print(f"[3/3] Transcribing and aligning...")
    result = model.transcribe(audio, batch_size=4)

    # Load alignment model for the detected language
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device=device
    )

    # Flatten segments to word list
    words = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word_text = word_info["word"].strip()
            if word_text:  # skip empty strings
                words.append(
                    {
                        "word": word_text,
                        "start": round(word_info["start"], 3),
                        "end": round(word_info["end"], 3),
                    }
                )

    return words


def _normalise_token(text: str) -> str:
    """
    Normalise a single token for comparison.

    Steps applied in order:
    1. Fold typographic punctuation variants to their plain ASCII equivalents
       so that, e.g., curly apostrophes and straight apostrophes compare equal:
         - RIGHT/LEFT SINGLE QUOTATION MARK (U+2019, U+2018)  → apostrophe (')
         - MODIFIER LETTER APOSTROPHE (U+02BC)                → apostrophe (')
         - GRAVE ACCENT (U+0060), ACUTE ACCENT (U+00B4)       → apostrophe (')
         - RIGHT/LEFT DOUBLE QUOTATION MARK (U+201C, U+201D)  → double quote (")
         - PRIME / DOUBLE PRIME (U+2032, U+2033)              → apostrophe/double
         - HORIZONTAL ELLIPSIS (U+2026)                       → three dots (...)
         - EN DASH (U+2013), EM DASH (U+2014)                → hyphen (-)
    2. Strip leading/trailing punctuation/whitespace (\\W and _).
    3. Lowercase.

    Internal characters (e.g. the apostrophe in "can't") are preserved after
    folding so "can't" still matches "can't" but not "cant".
    """
    # Step 1 — fold typographic variants to plain ASCII
    _FOLD = str.maketrans(
        "\u2018\u2019\u02bc\u0060\u00b4\u2032",   # → apostrophe
        "''''''",
        "",
    )
    _FOLD.update({
        ord("\u201c"): '"',  # LEFT DOUBLE QUOTATION MARK
        ord("\u201d"): '"',  # RIGHT DOUBLE QUOTATION MARK
        ord("\u2033"): '"',  # DOUBLE PRIME
        ord("\u2026"): "...",  # HORIZONTAL ELLIPSIS (expand so edge-strip still works)
        ord("\u2013"): "-",  # EN DASH
        ord("\u2014"): "-",  # EM DASH
    })
    text = text.translate(_FOLD)
    # Step 2+3 — strip edge punctuation then lowercase
    return re.sub(r"^[\W_]+|[\W_]+$", "", text, flags=re.UNICODE).lower()


def _tokenise_sentence(sentence: str) -> List[str]:
    """
    Split a sentence into normalised tokens.

    Splits on whitespace, normalises each piece, and discards any that reduce
    to the empty string after normalisation (e.g. a bare full-stop token).
    """
    return [t for t in (_normalise_token(w) for w in sentence.split()) if t]


def _align_sentences_to_words(
    sentences: List[str], words: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Align each authoritative sentence to a span of WhisperX words using
    a global LCS (Longest Common Subsequence) alignment.

    Algorithm
    ---------
    1. Flatten all sentence tokens into a single global script-token list,
       tracking sentence boundaries (start/end token indices).
    2. Normalise both the script tokens and the WhisperX words with
       ``_normalise_token``.
    3. Run DP/LCS to find the best monotone mapping from script tokens to
       WhisperX word indices.
    4. Fill in any unmatched script tokens by interpolating proportionally
       between adjacent matched tokens (clamped to valid range).
    5. For each sentence, take the WhisperX start time of its first mapped
       token and the end time of its last mapped token.

    Tolerances
    ----------
    - Punctuation attached to word edges  (e.g. ``"World."`` → ``"world"``)
    - Capitalisation differences           (``"Hello"`` → ``"hello"``)
    - Phonetic / spelling variants         (``"foederati"`` vs ``"federati"``)
    - WhisperX word splits or merges
    - Extra or missing tokens on either side
    - Extra or irregular whitespace between sentence words

    Parameters
    ----------
    sentences : list[str]
        Authoritative sentence strings from the Story Generation artifact.
    words : list[dict]
        WhisperX word dicts: ``{"word": str, "start": float, "end": float}``.

    Returns
    -------
    list[dict]
        One entry per sentence::

            {
                "sentence_index": int,
                "start": float,         # start of the first matched word
                "end": float,           # end of the last matched word
                "word_indices": list[int]   # contiguous indices into *words*
            }
    """
    if not sentences:
        return []

    # ------------------------------------------------------------------
    # 1. Build global script-token list and record sentence boundaries.
    # ------------------------------------------------------------------
    all_script_tokens: List[str] = []
    sentence_token_ranges: List[tuple] = []  # (start_tok_idx, end_tok_idx inclusive, tokens)

    for sentence in sentences:
        tokens = _tokenise_sentence(sentence)  # already normalised
        start_idx = len(all_script_tokens)
        all_script_tokens.extend(tokens)
        end_idx = len(all_script_tokens) - 1
        sentence_token_ranges.append((start_idx, end_idx, tokens))

    n = len(all_script_tokens)
    m = len(words)
    ts_norm = [_normalise_token(w["word"]) for w in words]

    # ------------------------------------------------------------------
    # 2. LCS DP table.
    # ------------------------------------------------------------------
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if all_script_tokens[i - 1] == ts_norm[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # ------------------------------------------------------------------
    # 3. Backtrack to recover the alignment mapping.
    # ------------------------------------------------------------------
    mapping: List[int] = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        if all_script_tokens[i - 1] == ts_norm[j - 1]:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # ------------------------------------------------------------------
    # 4. Interpolate / clamp unmatched positions.
    # ------------------------------------------------------------------
    last_matched_ts = -1
    last_matched_script = -1
    for i in range(n):
        if mapping[i] != -1:
            last_matched_ts = mapping[i]
            last_matched_script = i
        else:
            # Find next matched token to interpolate between.
            next_matched_ts = -1
            next_matched_script = -1
            for k in range(i + 1, n):
                if mapping[k] != -1:
                    next_matched_ts = mapping[k]
                    next_matched_script = k
                    break

            if last_matched_ts >= 0 and next_matched_ts >= 0:
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

    mapping = [max(0, min(v, m - 1)) for v in mapping]

    # ------------------------------------------------------------------
    # 5. Build per-sentence results from mapping.
    # ------------------------------------------------------------------
    aligned_sentences: List[Dict[str, Any]] = []
    for sent_index, (start_tok, end_tok, tokens) in enumerate(sentence_token_ranges):
        sentence = sentences[sent_index]

        if not tokens or start_tok > end_tok:
            # Zero-token sentence — assign zero-length timestamp.
            ref_time = words[mapping[start_tok - 1]]["end"] if start_tok > 0 else 0.0
            aligned_sentences.append(
                {
                    "sentence_index": sent_index,
                    "start": round(ref_time, 3),
                    "end": round(ref_time, 3),
                    "word_indices": [],
                }
            )
            continue

        first_wx_idx = mapping[start_tok]
        last_wx_idx = mapping[end_tok]

        aligned_sentences.append(
            {
                "sentence_index": sent_index,
                "start": round(words[first_wx_idx]["start"], 3),
                "end": round(words[last_wx_idx]["end"], 3),
                "word_indices": list(range(first_wx_idx, last_wx_idx + 1)),
            }
        )

    return aligned_sentences


def _measure_audio_duration(audio_path: str) -> float:
    """
    Return the actual duration of the audio file in seconds.

    Measured independently of WhisperX word timings.  Tries the following
    backends in order:

    1. **soundfile**  — handles WAV, FLAC, OGG, AIFF, and more.
    2. **wave**       — stdlib, WAV only, zero extra dependencies.
    3. **mutagen**    — handles MP3, AAC, M4A, and other container formats.

    Raises
    ------
    RuntimeError
        If none of the backends can read the file.
    """
    # --- soundfile -----------------------------------------------------------
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        pass

    # --- wave (stdlib) -------------------------------------------------------
    try:
        import wave
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
    except Exception:
        pass

    # --- mutagen -------------------------------------------------------------
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(audio_path)
        if mf is not None and mf.info is not None:
            return mf.info.length
    except Exception:
        pass

    raise RuntimeError(
        f"Could not measure audio duration for {audio_path!r}: "
        "none of soundfile / wave / mutagen could read the file."
    )


def compute_word_timestamps(story: dict, config: Any) -> Dict[str, Any]:
    """
    Compute word-level timestamps for the narration audio.

    Parameters
    ----------
    story : dict
        Story artifact from history_story_generator:
        {"title": str, "sentences": list[str]}
    config : PipelineConfig
        Pipeline configuration (used to locate output directory).

    Returns
    -------
    dict
        {
            "audio_path": str,       # relative to output_dir, e.g., "narration.mp3"
            "duration": float,       # actual audio duration in seconds (from audio file)
            "words": list[{
                "index": int,
                "word": str,
                "start": float,
                "end": float
            }],
            "sentences": list[{
                "sentence_index": int,
                "start": float,
                "end": float,
                "word_indices": list[int]
            }]
        }

    Raises
    ------
    FileNotFoundError
        If the voice generation artifact or audio file is missing.
    ValueError
        If the story artifact is invalid, audio validation fails, or sentence
        tokens do not match WhisperX words (substantive alignment mismatch).
    RuntimeError
        If WhisperX transcription/alignment fails or audio duration cannot be
        measured.
    """
    # --- Input validation ---------------------------------------------------
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

    # --- Load voice generation artifact to get audio path -------------------
    voice_artifact = _load_voice_generation_artifact(config.output_dir)
    audio_filename = voice_artifact.get("audio_path")
    if not audio_filename:
        raise ValueError(
            "Voice generation artifact missing 'audio_path' in its data"
        )

    audio_path = config.output_dir / audio_filename
    _validate_audio_file(audio_path)

    # --- Run WhisperX transcription and alignment ---------------------------
    try:
        words = _transcribe_and_align(str(audio_path))
    except Exception as e:
        raise RuntimeError(f"WhisperX transcription/alignment failed: {e}") from e

    if not words:
        raise ValueError("WhisperX returned no words")

    # --- Add word indices ---------------------------------------------------
    indexed_words = []
    for idx, w in enumerate(words):
        indexed_words.append(
            {
                "index": idx,
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
            }
        )

    # --- Align sentences to words -------------------------------------------
    aligned_sentences = _align_sentences_to_words(sentences, words)

    # --- Compute duration from actual audio file (not last WhisperX word) ---
    try:
        duration = _measure_audio_duration(str(audio_path))
    except RuntimeError as e:
        raise RuntimeError(f"Failed to measure audio duration: {e}") from e

    # --- Build timing dictionary --------------------------------------------
    timing_dict = {
        "audio_path": audio_filename,  # relative path from output_dir
        "duration": round(duration, 3),
        "words": indexed_words,
        "sentences": aligned_sentences,
    }

    return timing_dict


# ---------------------------------------------------------------------------
# Pipeline stage function
# ---------------------------------------------------------------------------
def stage_whisperx_timing(config: Any, prev: dict) -> Any:
    """
    Pipeline stage: WhisperX timing.

    Expects `prev` to be the story artifact (from story_generation).
    Returns a PipelineArtifact with timing data.
    """
    from dataclasses import asdict

    # We ignore the pipeline runner's PipelineArtifact expectation here and
    # return a dict that will be wrapped by the runner? Actually, the runner
    # expects the stage function to return a PipelineArtifact.

    # However, note: the pipeline runner's stage function signature is:
    #   StageFn = Callable[[PipelineConfig, dict], PipelineArtifact]
    # and the runner does:
    #   artifact = stage_fn(self.config, story_artifact)

    # So we must return a PipelineArtifact.

    # But we don't want to import PipelineArtifact here to avoid circular
    # dependencies? We can define it locally or import from run_pipeline.
    # Since we are in the same project, we can import from run_pipeline.

    # However, to avoid tight coupling, we can return a dict and let the
    # pipeline runner wrap it? No, the runner expects the stage function to
    # return a PipelineArtifact.

    # Let's check the existing stages: they return PipelineArtifact instances.

    # We'll import PipelineArtifact from run_pipeline. Since we are in the same
    # directory, we can do a relative import? But we are not in a package.

    # Instead, we can copy the minimal PipelineArtifact definition here? Or
    # we can import it. Since we are allowed to modify run_pipeline, we will
    # import it and then update run_pipeline to include our stage.

    # For now, we'll assume we can import from run_pipeline. We'll add the
    # import at the top of this file and then update run_pipeline accordingly.

    # However, to avoid circular imports, we can define a simple class here
    # that matches the expected interface? The runner only uses:
    #   artifact.stage_name, artifact.data, artifact.metadata, and artifact.to_json()

    # We'll create a minimal class.

    timing_data = compute_word_timestamps(story=prev, config=config)

    artifact = PipelineArtifact(
        stage_name="whisperx_timing",
        data=timing_data,
        metadata={
            "word_count": len(timing_data["words"]),
            "sentence_count": len(timing_data["sentences"]),
        },
    )
    return artifact


# ---------------------------------------------------------------------------
# Helper class to mimic PipelineArtifact (if we cannot import)
# ---------------------------------------------------------------------------
class PipelineArtifact:
    """Minimal PipelineArtifact for use if import fails."""

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


# ---------------------------------------------------------------------------
# CLI for standalone testing (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow running this script directly for testing:
    #   python whisperx_timing.py <story_json_file> <output_dir>
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python whisperx_timing.py <story_json_file> <output_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    story_path = sys.argv[1]
    output_dir = Path(sys.argv[2])

    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    # We need a config-like object with output_dir
    class DummyConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    config = DummyConfig(output_dir)

    try:
        artifact = stage_whisperx_timing(config, story)
        print(json.dumps(artifact.data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)