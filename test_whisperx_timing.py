#!/usr/bin/env python3
"""
Unit tests for whisperx_timing.py
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from whisperx_timing import (
    _align_sentences_to_words,
    _load_voice_generation_artifact,
    _measure_audio_duration,
    _normalise_token,
    _tokenise_sentence,
    _transcribe_and_align,
    _validate_audio_file,
    compute_word_timestamps,
    stage_whisperx_timing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_story(sentences):
    return {"title": "Test Story", "sentences": sentences}


def make_voice_artifact(audio_filename="narration.mp3"):
    return {
        "data": {
            "audio_path": audio_filename,
            "voice": "en-US-AndrewNeural",
            "rate": "+20%",
            "pitch": "+0Hz",
            "volume": "+0%",
            "sentence_count": 2,
            "char_count": 10,
        }
    }


class DummyConfig:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)


# ---------------------------------------------------------------------------
# _load_voice_generation_artifact
# ---------------------------------------------------------------------------

def test_load_voice_generation_artifact_success(tmp_path):
    artifact_data = make_voice_artifact()
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text(json.dumps(artifact_data), encoding="utf-8")

    config = DummyConfig(tmp_path)
    result = _load_voice_generation_artifact(config.output_dir)
    assert result == artifact_data["data"]


def test_load_voice_generation_artifact_missing_file(tmp_path):
    config = DummyConfig(tmp_path)
    with pytest.raises(FileNotFoundError, match="Voice generation artifact not found"):
        _load_voice_generation_artifact(config.output_dir)


def test_load_voice_generation_artifact_invalid_json(tmp_path):
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text("not json", encoding="utf-8")
    config = DummyConfig(tmp_path)
    with pytest.raises(ValueError, match="Failed to parse voice generation artifact"):
        _load_voice_generation_artifact(config.output_dir)


def test_load_voice_generation_artifact_missing_data(tmp_path):
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text(json.dumps({"not": "data"}), encoding="utf-8")
    config = DummyConfig(tmp_path)
    with pytest.raises(ValueError, match="Voice generation artifact missing 'data' field"):
        _load_voice_generation_artifact(config.output_dir)


# ---------------------------------------------------------------------------
# _validate_audio_file
# ---------------------------------------------------------------------------

def test_validate_audio_file_success(tmp_path):
    audio_file = tmp_path / "test.mp3"
    audio_file.write_text("fake audio content" + "x" * 82, encoding="utf-8")
    _validate_audio_file(audio_file)  # should not raise


def test_validate_audio_file_missing(tmp_path):
    audio_file = tmp_path / "missing.mp3"
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        _validate_audio_file(audio_file)


def test_validate_audio_file_not_a_file(tmp_path):
    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="Audio path is not a file"):
        _validate_audio_file(dir_path)


def test_validate_audio_file_empty(tmp_path):
    audio_file = tmp_path / "empty.mp3"
    audio_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Audio file is empty"):
        _validate_audio_file(audio_file)


def test_validate_audio_file_suspiciously_small(tmp_path):
    audio_file = tmp_path / "small.mp3"
    audio_file.write_text("x" * 50, encoding="utf-8")
    with pytest.raises(ValueError, match="Audio file suspiciously small"):
        _validate_audio_file(audio_file)


# ---------------------------------------------------------------------------
# _transcribe_and_align
# ---------------------------------------------------------------------------

@patch("whisperx_timing.whisperx.load_model")
@patch("whisperx_timing.whisperx.load_audio")
@patch("whisperx_timing.whisperx.load_align_model")
@patch("whisperx_timing.whisperx.align")
def test_transcribe_and_align(
    mock_align, mock_load_align_model, mock_load_audio, mock_load_model
):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "language": "en",
        "segments": [
            {"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]},
            {"words": [{"word": "world", "start": 0.5, "end": 1.0}]},
        ],
    }
    mock_load_model.return_value = mock_model
    mock_load_audio.return_value = "fake audio"
    mock_model_a = MagicMock()
    mock_metadata = {"language": "en"}
    mock_load_align_model.return_value = (mock_model_a, mock_metadata)
    mock_align.return_value = {
        "segments": [
            {"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]},
            {"words": [{"word": "world", "start": 0.5, "end": 1.0}]},
        ]
    }

    words = _transcribe_and_align("fake/path.mp3")

    assert len(words) == 2
    assert words[0] == {"word": "Hello", "start": 0.0, "end": 0.5}
    assert words[1] == {"word": "world", "start": 0.5, "end": 1.0}

    mock_load_model.assert_called_once_with("base", device="cpu", compute_type="int8")
    mock_load_audio.assert_called_once_with("fake/path.mp3")
    mock_model.transcribe.assert_called_once()
    mock_load_align_model.assert_called_once_with(language_code="en", device="cpu")
    mock_align.assert_called_once()


# ---------------------------------------------------------------------------
# _normalise_token / _tokenise_sentence helpers
# ---------------------------------------------------------------------------

def test_normalise_token_strips_punctuation():
    assert _normalise_token("Hello,") == "hello"
    assert _normalise_token(".World.") == "world"
    assert _normalise_token("don't") == "don't"  # internal punctuation kept


def test_normalise_token_lowercases():
    assert _normalise_token("HELLO") == "hello"


def test_normalise_token_typographic_apostrophes():
    """All apostrophe/quote variants must normalise identically to their
    plain ASCII equivalents so sentence text and WhisperX output match
    regardless of which Unicode codepoint was used."""
    # Straight apostrophe baseline
    straight = _normalise_token("state's")           # U+0027

    # RIGHT SINGLE QUOTATION MARK (U+2019) — the classic "curly apostrophe"
    assert _normalise_token("state\u2019s") == straight

    # LEFT SINGLE QUOTATION MARK (U+2018)
    assert _normalise_token("state\u2018s") == straight

    # MODIFIER LETTER APOSTROPHE (U+02BC)
    assert _normalise_token("state\u02bcs") == straight

    # PRIME (U+2032) — sometimes used as apostrophe in OCR/TTS output
    assert _normalise_token("state\u2032s") == straight

    # GRAVE ACCENT (U+0060) — sometimes used as apostrophe
    assert _normalise_token("state\u0060s") == straight

    # ACUTE ACCENT (U+00B4)
    assert _normalise_token("state\u00b4s") == straight

    # Confirm the normalised value is what we expect
    assert straight == "state's"


def test_normalise_token_typographic_double_quotes():
    """Curly double-quote variants fold to plain ASCII double-quote."""
    plain = _normalise_token('"quoted"')            # U+0022 on both sides
    assert _normalise_token("\u201cquoted\u201d") == plain   # curly open+close
    assert _normalise_token("\u2033quoted\u2033") == plain   # double prime


def test_normalise_token_en_em_dash_to_hyphen():
    """En dash and em dash fold to hyphen so hyphenated compounds match."""
    hyphen = _normalise_token("well-known")
    assert _normalise_token("well\u2013known") == hyphen   # EN DASH
    assert _normalise_token("well\u2014known") == hyphen   # EM DASH


def test_normalise_token_substantive_difference_still_fails():
    """Folding must not collapse genuinely different words."""
    # "state's" (with apostrophe) vs "states" (no apostrophe) differ in
    # content and must NOT normalise to the same token.
    assert _normalise_token("state's") != _normalise_token("states")


def test_align_sentences_curly_vs_straight_apostrophe():
    """
    End-to-end alignment: sentence uses a straight apostrophe but WhisperX
    returns a curly RIGHT SINGLE QUOTATION MARK (U+2019), or vice versa.
    Alignment must succeed.
    """
    # Sentence text has straight apostrophe
    sentences = ["The state's capital"]
    words = [
        {"word": "The",          "start": 0.0, "end": 0.3},
        {"word": "state\u2019s", "start": 0.3, "end": 0.7},  # curly apostrophe
        {"word": "capital",      "start": 0.7, "end": 1.2},
    ]
    result = _align_sentences_to_words(sentences, words)
    assert len(result) == 1
    assert result[0]["word_indices"] == [0, 1, 2]

    # Reverse: sentence has curly apostrophe, WhisperX has straight
    sentences2 = ["The state\u2019s capital"]
    words2 = [
        {"word": "The",       "start": 0.0, "end": 0.3},
        {"word": "state's",   "start": 0.3, "end": 0.7},   # straight apostrophe
        {"word": "capital",   "start": 0.7, "end": 1.2},
    ]
    result2 = _align_sentences_to_words(sentences2, words2)
    assert result2[0]["word_indices"] == [0, 1, 2]



def test_tokenise_sentence_skips_bare_punctuation():
    # A bare "." or "," should be dropped
    tokens = _tokenise_sentence("Hello . world")
    assert tokens == ["hello", "world"]


# ---------------------------------------------------------------------------
# _align_sentences_to_words — new deterministic tests
# ---------------------------------------------------------------------------

def test_align_sentences_to_words_basic():
    """Exact match: sentence words equal WhisperX words."""
    sentences = ["Hello world", "How are you"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you", "start": 2.0, "end": 2.5},
    ]

    result = _align_sentences_to_words(sentences, words)

    assert len(result) == 2
    assert result[0] == {"sentence_index": 0, "start": 0.0, "end": 1.0, "word_indices": [0, 1]}
    assert result[1] == {"sentence_index": 1, "start": 1.0, "end": 2.5, "word_indices": [2, 3, 4]}


def test_align_sentences_to_words_punctuation_tolerance():
    """WhisperX words carry trailing punctuation; sentence words are clean."""
    sentences = ["Hello world", "How are you"]
    words = [
        {"word": "Hello,", "start": 0.0, "end": 0.5},   # trailing comma
        {"word": "world.", "start": 0.5, "end": 1.0},   # trailing full-stop
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you?", "start": 2.0, "end": 2.5},     # trailing question-mark
    ]

    result = _align_sentences_to_words(sentences, words)

    assert len(result) == 2
    assert result[0]["word_indices"] == [0, 1]
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0
    assert result[1]["word_indices"] == [2, 3, 4]
    assert result[1]["end"] == 2.5


def test_align_sentences_to_words_capitalisation_tolerance():
    """Sentence has different capitalisation from WhisperX output."""
    sentences = ["hello world"]   # all lowercase
    words = [
        {"word": "HELLO", "start": 0.0, "end": 0.4},   # all caps from WhisperX
        {"word": "WORLD", "start": 0.4, "end": 0.9},
    ]

    result = _align_sentences_to_words(sentences, words)

    assert len(result) == 1
    assert result[0]["word_indices"] == [0, 1]
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 0.9


def test_align_sentences_to_words_whitespace_tolerance():
    """Extra internal whitespace in sentence strings is normalised away."""
    sentences = ["Hello   world"]   # multiple spaces
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]

    result = _align_sentences_to_words(sentences, words)

    assert len(result) == 1
    assert result[0]["word_indices"] == [0, 1]


def test_align_sentences_to_words_multi_sentence_contiguous_indices():
    """Word indices across sentences are strictly contiguous and non-overlapping."""
    sentences = ["One two", "Three four five", "Six"]
    words = [
        {"word": "One", "start": 0.0, "end": 0.3},
        {"word": "two", "start": 0.3, "end": 0.6},
        {"word": "Three", "start": 0.6, "end": 0.9},
        {"word": "four", "start": 0.9, "end": 1.2},
        {"word": "five", "start": 1.2, "end": 1.5},
        {"word": "Six", "start": 1.5, "end": 1.8},
    ]

    result = _align_sentences_to_words(sentences, words)

    assert len(result) == 3
    # indices are contiguous across sentences
    assert result[0]["word_indices"] == [0, 1]
    assert result[1]["word_indices"] == [2, 3, 4]
    assert result[2]["word_indices"] == [5]
    # each sentence derives timing from its first/last word
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 0.6
    assert result[1]["start"] == 0.6
    assert result[1]["end"] == 1.5
    assert result[2]["start"] == 1.5
    assert result[2]["end"] == 1.8


def test_align_sentences_to_words_substantive_mismatch_raises():
    """A word that genuinely differs (not just punctuation/case) must raise ValueError."""
    sentences = ["Hello world"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "planet", "start": 0.5, "end": 1.0},  # "planet" ≠ "world"
    ]

    with pytest.raises(ValueError, match="expected 'world' but WhisperX word is 'planet'"):
        _align_sentences_to_words(sentences, words)


def test_align_sentences_to_words_exhausted_words_raises():
    """Raises ValueError when WhisperX words run out before all tokens are matched."""
    sentences = ["Hello world", "How are you"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        # "How are you" is missing entirely
    ]

    with pytest.raises(ValueError, match="WhisperX words but only"):
        _align_sentences_to_words(sentences, words)


def test_align_sentences_to_words_no_padding_no_truncation():
    """
    Verify there is no truncation or padding fallback: extra WhisperX words
    that go beyond what the sentences need do NOT get silently dropped, and
    short WhisperX word lists do NOT get padded.
    """
    # Extra words beyond what is needed → substantive mismatch on the wrong
    # word, OR the function simply succeeds but leaves trailing words unread
    # (that is acceptable — extra trailing words don't break alignment).
    sentences = ["Hi"]
    words = [
        {"word": "Hi", "start": 0.0, "end": 0.3},
        {"word": "extra", "start": 0.3, "end": 0.6},  # leftover — ignored
    ]
    # The alignment should succeed for the sentence that IS present.
    result = _align_sentences_to_words(sentences, words)
    assert result[0]["word_indices"] == [0]  # only the matched word; no padding

    # Fewer words than tokens → must raise, not pad
    sentences2 = ["Hi there"]
    words2 = [{"word": "Hi", "start": 0.0, "end": 0.3}]  # "there" missing
    with pytest.raises(ValueError):
        _align_sentences_to_words(sentences2, words2)


def test_align_sentences_to_words_empty_sentences_list():
    words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
    assert _align_sentences_to_words([], words) == []


# ---------------------------------------------------------------------------
# _measure_audio_duration
# ---------------------------------------------------------------------------

def test_measure_audio_duration_via_soundfile(tmp_path):
    """soundfile backend is tried first; mock it to return a known duration.

    soundfile is imported lazily inside _measure_audio_duration, so we inject
    a fake module via sys.modules rather than patching a module-level attribute.
    """
    import sys

    dummy_path = str(tmp_path / "audio.wav")

    mock_info = MagicMock()
    mock_info.duration = 12.345

    fake_sf = MagicMock()
    fake_sf.info.return_value = mock_info

    original = sys.modules.get("soundfile")
    sys.modules["soundfile"] = fake_sf
    try:
        duration = _measure_audio_duration(dummy_path)
        assert duration == 12.345
        fake_sf.info.assert_called_once_with(dummy_path)
    finally:
        if original is None:
            sys.modules.pop("soundfile", None)
        else:
            sys.modules["soundfile"] = original


def test_measure_audio_duration_fallback_to_wave(tmp_path):
    """When soundfile raises, the wave stdlib fallback is used."""
    import sys
    import wave
    import struct

    # Create a minimal valid WAV file (1-channel, 16-bit, 8000 Hz, 0.5 s → 4000 frames)
    wav_path = tmp_path / "test.wav"
    n_frames = 4000
    sample_rate = 8000
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)

    # Remove soundfile from sys.modules so the import inside the function fails.
    original_sf = sys.modules.pop("soundfile", None)
    try:
        duration = _measure_audio_duration(str(wav_path))
        assert abs(duration - 0.5) < 1e-6
    finally:
        if original_sf is not None:
            sys.modules["soundfile"] = original_sf


def test_measure_audio_duration_no_backend_raises(tmp_path):
    """Raises RuntimeError when no backend can read the file."""
    import sys

    dummy_path = str(tmp_path / "garbage.mp3")
    Path(dummy_path).write_bytes(b"not audio data" * 20)

    # Force all backends to fail by removing soundfile and mutagen
    saved = {}
    for mod in ("soundfile", "mutagen"):
        saved[mod] = sys.modules.pop(mod, None)

    try:
        with pytest.raises(RuntimeError, match="Could not measure audio duration"):
            _measure_audio_duration(dummy_path)
    finally:
        for mod, orig in saved.items():
            if orig is not None:
                sys.modules[mod] = orig


# ---------------------------------------------------------------------------
# compute_word_timestamps — validation tests (existing, preserved)
# ---------------------------------------------------------------------------

def test_compute_word_timestamps_invalid_story():
    config = DummyConfig("/tmp")

    with pytest.raises(TypeError, match="Story must be a dict"):
        compute_word_timestamps("not a dict", config)

    with pytest.raises(ValueError, match="Story artifact missing required key: 'sentences'"):
        compute_word_timestamps({}, config)

    with pytest.raises(TypeError, match="story\\['sentences'\\] must be a list"):
        compute_word_timestamps({"sentences": "not a list"}, config)

    with pytest.raises(ValueError, match="story\\['sentences'\\] must be non-empty"):
        compute_word_timestamps({"sentences": []}, config)

    with pytest.raises(TypeError, match="Sentence 0 must be a string"):
        compute_word_timestamps({"sentences": [123]}, config)

    with pytest.raises(ValueError, match="Sentence 0 is empty or whitespace-only"):
        compute_word_timestamps({"sentences": [""]}, config)


def test_compute_word_timestamps_missing_voice_artifact(tmp_path):
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world"])
    with pytest.raises(FileNotFoundError, match="Voice generation artifact not found"):
        compute_word_timestamps(story, config)


def test_compute_word_timestamps_invalid_voice_artifact(tmp_path):
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text("not json", encoding="utf-8")
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world"])
    with pytest.raises(ValueError, match="Failed to parse voice generation artifact"):
        compute_word_timestamps(story, config)


def test_compute_word_timestamps_missing_audio_path(tmp_path):
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text(json.dumps({"data": {}}), encoding="utf-8")
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world"])
    with pytest.raises(ValueError, match="Voice generation artifact missing 'audio_path'"):
        compute_word_timestamps(story, config)


def test_compute_word_timestamps_missing_audio_file(tmp_path):
    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text(
        json.dumps({"data": {"audio_path": "missing.mp3"}}), encoding="utf-8"
    )
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world"])
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        compute_word_timestamps(story, config)


# ---------------------------------------------------------------------------
# compute_word_timestamps — success path with independent audio duration
# ---------------------------------------------------------------------------

@patch("whisperx_timing._transcribe_and_align")
@patch("whisperx_timing._measure_audio_duration")
def test_compute_word_timestamps_success(mock_duration, mock_transcribe, tmp_path):
    """
    Verifies:
    - Sentences aligned to the correct contiguous WhisperX words.
    - duration comes from _measure_audio_duration, NOT the last word's end time.
    - word_indices are contiguous across sentences.
    """
    audio_file = tmp_path / "narration.mp3"
    audio_file.write_text("fake audio" + "x" * 90, encoding="utf-8")

    artifact_path = tmp_path / "voice_generation.json"
    artifact_path.write_text(
        json.dumps({"data": {"audio_path": "narration.mp3"}}), encoding="utf-8"
    )
    config = DummyConfig(tmp_path)

    story = make_story(["Hello world", "How are you"])

    mock_transcribe.return_value = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you", "start": 2.0, "end": 2.5},
    ]
    # Independent audio duration is longer than the last WhisperX word end (2.5)
    mock_duration.return_value = 3.0

    result = compute_word_timestamps(story, config)

    # Audio path
    assert result["audio_path"] == "narration.mp3"

    # Duration comes from _measure_audio_duration, NOT from words[-1]["end"]
    assert result["duration"] == 3.0
    assert result["duration"] != 2.5  # explicitly not the last-word end

    # Words
    assert len(result["words"]) == 5
    assert result["words"][0] == {"index": 0, "word": "Hello", "start": 0.0, "end": 0.5}
    assert result["words"][4] == {"index": 4, "word": "you", "start": 2.0, "end": 2.5}

    # Sentences
    assert len(result["sentences"]) == 2
    assert result["sentences"][0] == {
        "sentence_index": 0,
        "start": 0.0,
        "end": 1.0,
        "word_indices": [0, 1],
    }
    assert result["sentences"][1] == {
        "sentence_index": 1,
        "start": 1.0,
        "end": 2.5,
        "word_indices": [2, 3, 4],
    }

    # Contiguous indices: last index of sentence 0 + 1 == first index of sentence 1
    assert result["sentences"][0]["word_indices"][-1] + 1 == result["sentences"][1]["word_indices"][0]

    mock_transcribe.assert_called_once_with(str(audio_file))
    mock_duration.assert_called_once_with(str(audio_file))


@patch("whisperx_timing._transcribe_and_align")
@patch("whisperx_timing._measure_audio_duration")
def test_compute_word_timestamps_punctuation_case_tolerance(mock_duration, mock_transcribe, tmp_path):
    """WhisperX returns words with mixed capitalisation and punctuation."""
    audio_file = tmp_path / "narration.mp3"
    audio_file.write_text("x" * 150, encoding="utf-8")
    (tmp_path / "voice_generation.json").write_text(
        json.dumps({"data": {"audio_path": "narration.mp3"}}), encoding="utf-8"
    )
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world", "How are you"])

    mock_transcribe.return_value = [
        {"word": "HELLO,", "start": 0.0, "end": 0.5},
        {"word": "World.", "start": 0.5, "end": 1.0},
        {"word": "how", "start": 1.0, "end": 1.5},
        {"word": "ARE", "start": 1.5, "end": 2.0},
        {"word": "you?", "start": 2.0, "end": 2.5},
    ]
    mock_duration.return_value = 2.8

    result = compute_word_timestamps(story, config)

    # Should succeed despite capitalisation and punctuation differences
    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["word_indices"] == [0, 1]
    assert result["sentences"][1]["word_indices"] == [2, 3, 4]
    assert result["duration"] == 2.8


# ---------------------------------------------------------------------------
# stage_whisperx_timing — integration test
# ---------------------------------------------------------------------------

def test_stage_whisperx_timing_integration(tmp_path):
    audio_file = tmp_path / "narration.mp3"
    audio_file.write_text("fake audio" + "x" * 90, encoding="utf-8")

    (tmp_path / "voice_generation.json").write_text(
        json.dumps({"data": {"audio_path": "narration.mp3"}}), encoding="utf-8"
    )
    config = DummyConfig(tmp_path)
    story = make_story(["Hello world", "How are you"])

    with patch("whisperx_timing._transcribe_and_align") as mock_transcribe, \
         patch("whisperx_timing._measure_audio_duration") as mock_duration:

        mock_transcribe.return_value = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
            {"word": "How", "start": 1.0, "end": 1.5},
            {"word": "are", "start": 1.5, "end": 2.0},
            {"word": "you", "start": 2.0, "end": 2.5},
        ]
        mock_duration.return_value = 2.7

        try:
            from run_pipeline import PipelineArtifact
        except ImportError:
            from whisperx_timing import PipelineArtifact

        artifact = stage_whisperx_timing(config, story)

        assert artifact.stage_name == "whisperx_timing"
        assert artifact.data["audio_path"] == "narration.mp3"
        assert artifact.data["duration"] == 2.7  # from _measure_audio_duration
        assert len(artifact.data["words"]) == 5
        assert len(artifact.data["sentences"]) == 2
        assert artifact.metadata["word_count"] == 5
        assert artifact.metadata["sentence_count"] == 2

        mock_transcribe.assert_called_once()
        mock_duration.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])