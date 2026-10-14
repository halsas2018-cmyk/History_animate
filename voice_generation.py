#!/usr/bin/env python3
"""
voice_generation.py — STORY → single continuous audio file via Piper TTS.

Single responsibility:
    generate_voice(story: dict, output_dir: Path) -> {"audio_path": str, "voice": str, "rate": str, ...}

The function:
- Consumes the story_generation artifact (title + sentences)
- Joins all sentences into ONE continuous narration text
- Sends to Piper TTS as ONE synthesis request (using en_US-ryan-medium model)
- Saves ONE audio file to the pipeline run's artifact directory
- Returns metadata for downstream stages

No sentence-level timing, captions, scenes, Whiteboard Animator, or Remotion logic.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
from pydub.silence import detect_silence


# ---------------------------------------------------------------------------
# Piper TTS Configuration
# ---------------------------------------------------------------------------

# Model directory (absolute path to the pre‑downloaded Piper model)
MODEL_DIR = Path("/root/video_maker/models/piper")
VOICE_MODEL = "en_US-ryan-medium"

# Default output filename
AUDIO_FILENAME = "narration.mp3"

# Pause compression defaults (in milliseconds)
DEFAULT_MIN_SILENCE_LEN = 180
DEFAULT_MAX_PAUSE_MS = 220
DEFAULT_TARGET_PAUSE_MS = 150
DEFAULT_EDGE_SILENCE_MS = 50
DEFAULT_CROSSFADE_MS = 10


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
# Helper: Piper TTS synthesis
# ---------------------------------------------------------------------------

def _parse_rate_to_length_scale(rate: str) -> float:
    """
    Convert Edge TTS rate string (e.g. "+25%", "-10%") to Piper length_scale.
    Piper length_scale = 1.0 / (1.0 + rate_percent/100)
    """
    # Remove leading + or - and trailing %
    if rate.endswith("%"):
        rate = rate[:-1]
    try:
        rate_percent = float(rate)
    except ValueError:
        # If parsing fails, assume no change
        rate_percent = 0.0
    # Avoid division by zero; length_scale should be positive
    if 1.0 + rate_percent/100.0 == 0:
        return 1.0
    return 1.0 / (1.0 + rate_percent/100.0)


def _synthesize_with_piper(
    text: str,
    wav_output_path: Path,
    voice: str = VOICE_MODEL,
    rate: str = "+25%",
) -> None:
    """Internal: synthesize text to a WAV file via local Piper TTS."""
    model_path = MODEL_DIR / f"{voice}.onnx"
    config_path = MODEL_DIR / f"{voice}.onnx.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Piper model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Piper model config not found: {config_path}")

    length_scale = _parse_rate_to_length_scale(rate)

    # Piper command: echo "<text>" | piper --model <model> --config <config> --output_file <wav> --length-scale <scale>
    cmd = [
        "piper",
        "--model", str(model_path),
        "--config", str(config_path),
        "--output-file", str(wav_output_path),
        "--length-scale", str(length_scale),
    ]

    # Run Piper with the text as stdin input
    result = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        timeout=120,  # 2 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {result.stderr}")

    # Verify the WAV file was created and is not empty
    if not wav_output_path.exists():
        raise RuntimeError(f"Piper did not generate output file: {wav_output_path}")

    if wav_output_path.stat().st_size == 0:
        raise RuntimeError(f"Piper generated empty output file: {wav_output_path}")


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert WAV file to MP3 using FFmpeg with libmp3lame."""
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",  # Good quality VBR
        str(mp3_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=60,  # 1 minute timeout for conversion
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Pause & Silence Compression
# ---------------------------------------------------------------------------

def compress_audio_silence(
    audio_path: Path,
    output_path: Optional[Path] = None,
    min_silence_len: int = DEFAULT_MIN_SILENCE_LEN,
    silence_thresh: Optional[float] = None,
    max_pause_ms: int = DEFAULT_MAX_PAUSE_MS,
    target_pause_ms: int = DEFAULT_TARGET_PAUSE_MS,
    edge_silence_ms: int = DEFAULT_EDGE_SILENCE_MS,
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
) -> Path:
    """Compress long pauses in speech to a target duration while preserving natural cadence.

    Parameters
    ----------
    audio_path : Path
        Input audio file path.
    output_path : Path, optional
        Destination audio file path (defaults to overwriting audio_path).
    min_silence_len : int
        Minimum silence duration in ms to be detected as a pause (default: 180ms).
    silence_thresh : float, optional
        Silence threshold in dBFS (defaults to dynamic: max(audio.dBFS - 16.0, -45.0)).
    max_pause_ms : int
        Pauses longer than this will be compressed (default: 220ms).
    target_pause_ms : int
        Target duration in ms for compressed internal pauses (default: 150ms).
    edge_silence_ms : int
        Maximum leading and trailing silence to keep in ms (default: 50ms).
    crossfade_ms : int
        Crossfade duration in ms when stitching chunks to eliminate click artifacts (default: 10ms).

    Returns
    -------
    Path
        Path to the resulting audio file.
    """
    if output_path is None:
        output_path = audio_path

    audio = AudioSegment.from_file(str(audio_path))
    if len(audio) == 0:
        return output_path

    if silence_thresh is None:
        # Dynamic threshold based on average audio loudness
        silence_thresh = max(audio.dBFS - 16.0, -45.0)

    silent_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    if not silent_ranges:
        if output_path != audio_path:
            audio.export(str(output_path), format="mp3", bitrate="192k")
        return output_path

    total_len = len(audio)
    segments = []
    curr = 0
    for s_start, s_end in silent_ranges:
        if s_start > curr:
            segments.append((False, curr, s_start))
        segments.append((True, s_start, s_end))
        curr = s_end
    if curr < total_len:
        segments.append((False, curr, total_len))

    output_audio = AudioSegment.empty()

    for idx, (is_silence, start_ms, end_ms) in enumerate(segments):
        duration = end_ms - start_ms
        if not is_silence:
            chunk = audio[start_ms:end_ms]
        else:
            is_start = (idx == 0)
            is_end = (idx == len(segments) - 1)

            if is_start or is_end:
                keep_len = min(duration, edge_silence_ms)
            else:
                if duration > max_pause_ms:
                    keep_len = target_pause_ms
                else:
                    keep_len = duration

            chunk = audio[start_ms:start_ms + keep_len]

        if len(chunk) > 0:
            if len(output_audio) == 0:
                output_audio = chunk
            else:
                xfade = min(crossfade_ms, len(output_audio), len(chunk))
                if xfade > 0:
                    output_audio = output_audio.append(chunk, crossfade=xfade)
                else:
                    output_audio = output_audio + chunk

    # Save to temp file first to prevent corruption if output_path == audio_path
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        output_audio.export(str(tmp_path), format="mp3", bitrate="192k")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_voice(
    story: dict,
    output_dir: Path,
    *,
    voice: str = "en-US-AndrewMultilingualNeural",  # kept for compatibility but ignored
    rate: str = "+25%",
    pitch: str = "+0Hz",  # ignored
    volume: str = "+0%",  # ignored
    audio_filename: str = AUDIO_FILENAME,
    compress_pauses: bool = True,
    min_silence_len: int = DEFAULT_MIN_SILENCE_LEN,
    max_pause_ms: int = DEFAULT_MAX_PAUSE_MS,
    target_pause_ms: int = DEFAULT_TARGET_PAUSE_MS,
) -> dict:
    """Generate a single continuous audio file from the story narration using Piper TTS.

    Parameters
    ----------
    story : dict
        Output from history_story_generator.generate_history_story:
        {"title": str, "sentences": list[str]}
    output_dir : Path
        Pipeline run directory where the audio file will be saved.
    voice : str
        Ignored; Piper uses the fixed en_US-ryan-medium model.
    rate : str
        Speaking rate adjustment (e.g. "+25%") passed to Piper as length_scale.
    pitch : str
        Ignored (Piper does not support pitch via CLI in this setup).
    volume : str
        Ignored (Piper does not support volume via CLI in this setup).
    audio_filename : str
        Output filename (default: narration.mp3).
    compress_pauses : bool
        Whether to automatically compress long silences and pauses (default: True).
    min_silence_len : int
        Minimum silence duration in ms to qualify as a pause (default: 180ms).
    max_pause_ms : int
        Pauses longer than this will be compressed (default: 220ms).
    target_pause_ms : int
        Target duration in ms for compressed internal pauses (default: 150ms).

    Returns
    -------
    dict
        {
            "audio_path": str,       # Relative path from output_dir
            "voice": str,            # The voice used (fixed)
            "rate": str,             # The rate passed through
            "pitch": str,            # Unused
            "volume": str,           # Unused
            "sentence_count": int,
            "char_count": int
        }

    Raises
    ------
    ValueError
        Story artifact invalid or narration empty.
    RuntimeError
        Piper TTS synthesis fails or output validation fails.
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

    # --- Create temporary WAV file -----------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = Path(tmp_wav.name)

    try:
        # --- Single Piper TTS synthesis call --------------------------------
        _synthesize_with_piper(
            text=narration_text,
            wav_output_path=wav_path,
            voice=VOICE_MODEL,  # fixed model
            rate=rate,
        )

        # --- Convert WAV to MP3 --------------------------------------------
        _convert_wav_to_mp3(wav_path, audio_path)

        # --- Pause / Silence compression -----------------------------------
        if compress_pauses:
            try:
                compress_audio_silence(
                    audio_path,
                    min_silence_len=min_silence_len,
                    max_pause_ms=max_pause_ms,
                    target_pause_ms=target_pause_ms,
                )
            except Exception as e:
                print(f"Warning: Pause compression skipped due to error: {e}", file=sys.stderr)

        # --- Validate output ------------------------------------------------
        validate_audio_output(audio_path)

        # --- Return metadata ------------------------------------------------
        return {
            "audio_path": audio_filename,
            "voice": VOICE_MODEL,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
            "sentence_count": len(sentences),
            "char_count": len(narration_text),
        }
    finally:
        # Clean up temporary WAV file
        if wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass  # Ignore cleanup errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 3:
        print(
            "Usage: python voice_generation.py <story_json_file> <output_dir> [--voice VOICE] [--rate RATE] [--no-compress-pauses]",
            file=sys.stderr,
        )
        sys.exit(1)

    story_path = sys.argv[1]
    output_dir = Path(sys.argv[2])

    voice = "en-US-AndrewMultilingualNeural"  # kept for compatibility but ignored
    rate = "+25%"
    compress_pauses = True

    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        if idx + 1 < len(sys.argv):
            voice = sys.argv[idx + 1]  # ignored, but we parse to maintain interface

    if "--rate" in sys.argv:
        idx = sys.argv.index("--rate")
        if idx + 1 < len(sys.argv):
            rate = sys.argv[idx + 1]

    if "--no-compress-pauses" in sys.argv:
        compress_pauses = False

    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    result = generate_voice(
        story,
        output_dir,
        voice=voice,
        rate=rate,
        compress_pauses=compress_pauses,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))