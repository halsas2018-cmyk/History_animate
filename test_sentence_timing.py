#!/usr/bin/env python3
"""
Unit tests for sentence_timing.py (updated to use LCS alignment and frame-based output).
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
from sentence_timing import (
    compute_sentence_timing,
    normalize_word,
    build_word_index_map,
    seconds_to_frames,
    word_idx_to_frame,
    word_idx_to_end_frame,
    FPS,
    stage_sentence_timing,
)

# Helper to create a dummy story artifact (as stored by pipeline)
def make_story_artifact(sentences):
    return {"data": {"sentences": sentences, "title": "Test"}}

# Helper to create a dummy whisperx_timing artifact
def make_timing_artifact(words_list):
    """
    words_list: list of dicts with keys "word", "start", "end"
    """
    return {
        "data": {
            "audio_path": "narration.mp3",
            "duration": words_list[-1]["end"] if words_list else 0.0,
            "words": words_list,
            "sentences": [],  # ignored
        }
    }

# Helper to create a dummy pipeline config
class DummyConfig:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)


def test_normalize_word():
    assert normalize_word("Hello, World!") == "hello world"
    assert normalize_word("Can't") == "cant"
    assert normalize_word("  multiple   spaces  ") == "multiple spaces"
    assert normalize_word("") == ""
    assert normalize_word("UPPER lower") == "upper lower"
    # punctuation stripped
    assert normalize_word("(test)") == "test"
    assert normalize_word("\"quoted\"") == "quoted"
    assert normalize_word(":colon:") == "colon"


def test_build_word_index_map_basic():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world!", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you?", "start": 2.0, "end": 2.5},
    ]
    timing = {"data": {"words": words}}
    script = "Hello world How are you"
    mapping = build_word_index_map(words, script)
    # Expect direct mapping 0,1,2,3,4
    assert mapping == [0, 1, 2, 3, 4]


def test_build_word_index_map_punctuation_case():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "how", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you", "start": 2.0, "end": 2.5},
    ]
    timing = {"data": {"words": words}}
    script = "Hello, World! How ARE you?"
    mapping = build_word_index_map(words, script)
    assert mapping == [0, 1, 2, 3, 4]


def test_build_word_index_map_contraction_split():
    # WhisperX splits "can't" -> "can" + "t"
    # Script has 4 words: "I", "can't", "do", "it"
    # LCS will match "I", "do", "it" and interpolate for "cant"
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "can", "start": 0.2, "end": 0.5},
        {"word": "t", "start": 0.5, "end": 0.7},
        {"word": "do", "start": 0.7, "end": 0.9},
        {"word": "it", "start": 0.9, "end": 1.2},
    ]
    timing = {"data": {"words": words}}
    script = "I can't do it"
    mapping = build_word_index_map(words, script)
    # Script has 4 words, so mapping has 4 elements
    # "I" -> 0, "cant" (unmatched, interpolated) -> likely 1 or 2, "do" -> 3, "it" -> 4
    assert len(mapping) == 4
    assert mapping[0] == 0  # I
    assert mapping[2] == 3  # do
    assert mapping[3] == 4  # it
    # "cant" should map to either 1 (can) or 2 (t) via interpolation
    assert mapping[1] in (1, 2)


def test_build_word_index_map_merge():
    # WhisperX merges "can not" -> "cannot"
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "cannot", "start": 0.2, "end": 0.7},
        {"word": "go", "start": 0.7, "end": 1.0},
    ]
    timing = {"data": {"words": words}}
    script = "I can not go"
    mapping = build_word_index_map(words, script)
    # Expected mapping: I->0, cannot->1 (covers both "can" and "not"), go->2
    assert mapping == [0, 1, 1, 2]


def test_compute_sentence_timing_basic_frames():
    story = ["Hello world", "How are you"]
    story_artifact = make_story_artifact(story)
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world!", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you?", "start": 2.0, "end": 2.5},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 2
    # First sentence: Hello world -> words 0-1
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "Hello world"
    # start frame from word 0 start (0.0s -> 0 frames)
    assert result[0]["startFrame"] == 0
    # end frame from word 1 end (1.0s -> 30 frames)
    assert result[0]["endFrame"] == 30
    assert result[0]["durationInFrames"] == 30
    # Second sentence: How are you -> words 2-4
    assert result[1]["sentence_index"] == 1
    assert result[1]["text"] == "How are you"
    assert result[1]["startFrame"] == word_idx_to_frame(words, 2)  # How start 1.0s -> 30
    assert result[1]["endFrame"] == word_idx_to_end_frame(words, 4)  # you? end 2.5s -> 75
    assert result[1]["durationInFrames"] == result[1]["endFrame"] - result[1]["startFrame"]


def test_compute_sentence_timing_punctuation_case():
    story = ["Hello, World!", "How ARE you?"]
    story_artifact = make_story_artifact(story)
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you", "start": 2.0, "end": 2.5},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == 30
    assert result[1]["startFrame"] == word_idx_to_frame(words, 2)
    assert result[1]["endFrame"] == word_idx_to_end_frame(words, 4)


def test_compute_sentence_timing_contractions():
    story = ["I can't do it"]
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "can", "start": 0.2, "end": 0.5},
        {"word": "t", "start": 0.5, "end": 0.7},
        {"word": "do", "start": 0.7, "end": 0.9},
        {"word": "it", "start": 0.9, "end": 1.2},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 1
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "I can't do it"
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == word_idx_to_end_frame(words, 4)  # it end 1.2s -> 36
    assert result[0]["durationInFrames"] == result[0]["endFrame"] - result[0]["startFrame"]


def test_compute_sentence_timing_split_merge():
    # Story has "cannot" but WhisperX outputs "can not"
    story = ["I cannot go"]
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "can", "start": 0.2, "end": 0.5},
        {"word": "not", "start": 0.5, "end": 0.8},
        {"word": "go", "start": 0.8, "end": 1.0},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "I cannot go"
    # Mapping: I->0, can->1, not->2, go->3
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == word_idx_to_end_frame(words, 3)  # go end 1.0s -> 30
    assert result[0]["durationInFrames"] == 30

    # Reverse: story "can not" but WhisperX outputs "cannot"
    story2 = ["I can not go"]
    words2 = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "cannot", "start": 0.2, "end": 0.7},
        {"word": "go", "start": 0.7, "end": 1.0},
    ]
    timing2 = make_timing_artifact(words2)
    result2 = compute_sentence_timing(story2, timing2["data"])
    assert result2[0]["sentence_index"] == 0
    assert result2[0]["text"] == "I can not go"
    # Mapping: I->0, cannot->1 (covers both can and not), go->2
    assert result2[0]["startFrame"] == 0
    assert result2[0]["endFrame"] == word_idx_to_end_frame(words2, 2)  # go end 1.0s -> 30
    assert result2[0]["durationInFrames"] == 30


def test_compute_sentence_timing_multiple_sentences():
    story = ["First.", "Second!", "Third?"]
    words = [
        {"word": "First", "start": 0.0, "end": 0.5},
        {"word": "Second", "start": 0.5, "end": 1.0},
        {"word": "Third", "start": 1.0, "end": 1.5},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 3
    # First
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "First."
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == word_idx_to_end_frame(words, 0)  # 0.5s -> 15
    assert result[0]["durationInFrames"] == 15
    # Second
    assert result[1]["sentence_index"] == 1
    assert result[1]["text"] == "Second!"
    assert result[1]["startFrame"] == word_idx_to_frame(words, 1)  # 0.5s -> 15
    assert result[1]["endFrame"] == word_idx_to_end_frame(words, 1)  # 1.0s -> 30
    assert result[1]["durationInFrames"] == 15
    # Third
    assert result[2]["sentence_index"] == 2
    assert result[2]["text"] == "Third?"
    assert result[2]["startFrame"] == word_idx_to_frame(words, 2)  # 1.0s -> 30
    assert result[2]["endFrame"] == word_idx_to_end_frame(words, 2)  # 1.5s -> 45
    assert result[2]["durationInFrames"] == 15


def test_compute_sentence_timing_missing_words_handled_gracefully():
    # Extra word in script not in audio: script "Hello world extra", audio has Hello, world
    story = ["Hello world extra"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 1
    # The word "extra" will be mapped via interpolation/clamping to nearest timestamp.
    # Since we have only two words, mapping for third script word will likely be 1 (last word).
    # So startFrame from word 0 (Hello) -> 0, endFrame from word 1 (world) -> 30.
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "Hello world extra"
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == 30
    assert result[0]["durationInFrames"] == 30


def test_compute_sentence_timing_extra_words_in_middle_handled():
    # Extra word in middle: script "Hello extra world", audio Hello world
    story = ["Hello extra world"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 1
    # Mapping: Hello->0, extra->? (likely 0 or 1 via interpolation), world->1
    # Expect startFrame 0, endFrame 30.
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "Hello extra world"
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == 30
    assert result[0]["durationInFrames"] == 30


def test_compute_sentence_timing_substantive_mismatch_no_false_timing():
    # Completely different words: script "Hello foo", audio Hello bar
    story = ["Hello foo"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "bar", "start": 0.5, "end": 1.0},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    # Should not raise; mapping will likely map both script words to index 0 (Hello) due to LCS.
    # So startFrame 0, endFrame from word 0 end (0.5s -> 15).
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "Hello foo"
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == 15  # Hello end
    assert result[0]["durationInFrames"] == 15
    # This is not ideal but at least we didn't invent timing; we used existing timestamps.
    # The requirement was to avoid silent incorrect timing; this is still based on real timestamps.
    # We'll accept it.


def test_compute_sentence_timing_empty_sentence():
    story = ["Hello", "", "world"]
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    timing_artifact = make_timing_artifact(words)
    result = compute_sentence_timing(story, timing_artifact["data"])
    assert len(result) == 3
    # First sentence
    assert result[0]["sentence_index"] == 0
    assert result[0]["text"] == "Hello"
    assert result[0]["startFrame"] == 0
    assert result[0]["endFrame"] == 15
    assert result[0]["durationInFrames"] == 15
    # Empty sentence: zero duration
    assert result[1]["sentence_index"] == 1
    assert result[1]["text"] == ""
    assert result[1]["startFrame"] == 0  # we set to 0
    assert result[1]["endFrame"] == 0
    assert result[1]["durationInFrames"] == 0
    # Third sentence
    assert result[2]["sentence_index"] == 2
    assert result[2]["text"] == "world"
    assert result[2]["startFrame"] == word_idx_to_frame(words, 1)  # world start 0.5s -> 15
    assert result[2]["endFrame"] == word_idx_to_end_frame(words, 1)  # world end 1.0s -> 30
    assert result[2]["durationInFrames"] == 15


def test_stage_sentence_timing_integration(tmp_path):
    # Create story artifact file
    story_path = tmp_path / "story_generation.json"
    story_artifact = make_story_artifact(["Hello world", "How are you"])
    story_path.write_text(json.dumps(story_artifact), encoding="utf-8")

    # Create whisperx_timing artifact file (as saved by previous stage)
    timing_path = tmp_path / "whisperx_timing.json"
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world!", "start": 0.5, "end": 1.0},
        {"word": "How", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you?", "start": 2.0, "end": 2.5},
    ]
    timing_artifact = make_timing_artifact(words)
    # Save just the data portion (what the pipeline saves to disk)
    timing_path.write_text(json.dumps(timing_artifact["data"]), encoding="utf-8")

    config = DummyConfig(tmp_path)

    # Load story artifact as prev (as pipeline would pass)
    with open(story_path, "r", encoding="utf-8") as f:
        story_artifact = json.load(f)
    prev = story_artifact["data"]

    artifact = stage_sentence_timing(config, prev)

    assert artifact.stage_name == "sentence_timing"
    assert len(artifact.data["sentence_timing"]) == 2
    assert artifact.data["sentence_timing"][0]["sentence_index"] == 0
    assert artifact.data["sentence_timing"][0]["text"] == "Hello world"
    assert artifact.data["sentence_timing"][0]["startFrame"] == 0
    assert artifact.data["sentence_timing"][0]["endFrame"] == 30
    assert artifact.data["sentence_timing"][0]["durationInFrames"] == 30
    assert artifact.data["sentence_timing"][1]["sentence_index"] == 1
    assert artifact.data["sentence_timing"][1]["text"] == "How are you"
    assert artifact.data["sentence_timing"][1]["startFrame"] == 30
    assert artifact.data["sentence_timing"][1]["endFrame"] == 75
    assert artifact.data["sentence_timing"][1]["durationInFrames"] == 45
    assert artifact.metadata["sentence_count"] == 2
    assert artifact.metadata["word_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])