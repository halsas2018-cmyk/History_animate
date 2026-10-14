#!/usr/bin/env python3
"""
Unit tests for voice_generation.py — Edge TTS synthesis and pydub silence compression.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import voice_generation
from voice_generation import (
    validate_story_for_voice,
    validate_audio_output,
    compress_audio_silence,
    generate_voice,
)


def create_dummy_audio_with_pauses(
    tone_duration_ms: int = 500,
    pause_duration_ms: int = 600,
    num_segments: int = 3,
) -> AudioSegment:
    """Helper to create a synthetic AudioSegment with known tones and silence pauses."""
    tone = Sine(440).to_audio_segment(duration=tone_duration_ms)
    silence = AudioSegment.silent(duration=pause_duration_ms)

    audio = AudioSegment.silent(duration=200) # leading silence
    for i in range(num_segments):
        audio = audio + tone
        if i < num_segments - 1:
            audio = audio + silence
    audio = audio + AudioSegment.silent(duration=300) # trailing silence
    return audio


class TestVoiceValidators:
    """Tests for input and output validation."""

    def test_validate_story_valid(self):
        story = {
            "title": "Samurai Life",
            "sentences": ["Before dawn, the samurai woke up.", "He prepared his swords."],
        }
        validate_story_for_voice(story) # Should not raise

    def test_validate_story_not_a_dict(self):
        with pytest.raises(TypeError, match="Story must be a dict"):
            validate_story_for_voice(["not", "a", "dict"])

    def test_validate_story_missing_sentences(self):
        with pytest.raises(ValueError, match="missing required key: 'sentences'"):
            validate_story_for_voice({"title": "Test"})

    def test_validate_story_empty_sentences(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            validate_story_for_voice({"sentences": []})

    def test_validate_story_non_string_sentence(self):
        with pytest.raises(TypeError, match="Sentence 0 must be a string"):
            validate_story_for_voice({"sentences": [123]})

    def test_validate_story_whitespace_sentence(self):
        with pytest.raises(ValueError, match="Sentence 0 is empty or whitespace-only"):
            validate_story_for_voice({"sentences": ["   "]})


class TestSilenceCompression:
    """Tests for pydub-based pause and silence compression."""

    def test_compress_audio_silence_reduces_long_pauses(self):
        """Verify pauses > max_pause_ms are clamped to target_pause_ms."""
        audio = create_dummy_audio_with_pauses(
            tone_duration_ms=400,
            pause_duration_ms=600,
            num_segments=3,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
            in_path = Path(tmp_in.name)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
            out_path = Path(tmp_out.name)

        try:
            audio.export(str(in_path), format="mp3")
            initial_len = len(audio)

            compress_audio_silence(
                in_path,
                output_path=out_path,
                min_silence_len=180,
                silence_thresh=-35.0,
                max_pause_ms=200,
                target_pause_ms=150,
                edge_silence_ms=50,
            )

            result_audio = AudioSegment.from_file(str(out_path))
            # Output duration should be noticeably shorter than input
            assert len(result_audio) < initial_len
            # In our test case: 3 tones (1200ms) + 2 pauses (~300ms) + edges (~100ms) ~= 1600ms (+ crossfade)
            assert len(result_audio) < 2000
        finally:
            if in_path.exists():
                in_path.unlink()
            if out_path.exists():
                out_path.unlink()

    def test_compress_audio_silence_in_place(self):
        """Verify in-place file modification when output_path is None."""
        audio = create_dummy_audio_with_pauses(
            tone_duration_ms=300,
            pause_duration_ms=500,
            num_segments=2,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            file_path = Path(tmp.name)

        try:
            audio.export(str(file_path), format="mp3")
            initial_size = len(audio)

            compress_audio_silence(
                file_path,
                min_silence_len=150,
                silence_thresh=-35.0,
                max_pause_ms=200,
                target_pause_ms=150,
            )

            result_audio = AudioSegment.from_file(str(file_path))
            assert len(result_audio) < initial_size
        finally:
            if file_path.exists():
                file_path.unlink()


class TestGenerateVoice:
    """Tests for generate_voice orchestration."""

    @patch("voice_generation._synthesize_to_file")
    def test_generate_voice_success(self, mock_synth):
        """Test successful synthesis and silence compression."""
        async def fake_synth(text, output_path, **kwargs):
            dummy = create_dummy_audio_with_pauses(300, 500, 2)
            dummy.export(str(output_path), format="mp3")

        mock_synth.side_effect = fake_synth

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            story = {
                "title": "Test Title",
                "sentences": ["Sentence one.", "Sentence two."],
            }

            result = generate_voice(story, output_dir, compress_pauses=True)

            assert result["audio_path"] == "narration.mp3"
            assert result["sentence_count"] == 2
            assert (output_dir / "narration.mp3").exists()

            # Verify audio is valid
            audio = AudioSegment.from_file(str(output_dir / "narration.mp3"))
            assert len(audio) > 0
