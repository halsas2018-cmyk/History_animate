# Sentence Timing Stage Implementation

## Overview
This implementation adds a new `sentence_timing` stage to the dynamic history video pipeline that maps original story sentences to WhisperX word-level timestamps to produce sentence-level timing information.

## Files Created
1. `/root/whiteboard_anime/dynamic_history_video_project/sentence_timing.py` - Main stage implementation
2. `/root/whiteboard_anime/dynamic_history_video_project/test_sentence_timing.py` - Comprehensive unit test suite

## Files Modified
1. `/root/whiteboard_anime/dynamic_history_video_project/run_pipeline.py` - Added import and inserted stage into pipeline

## Implementation Details

### Core Logic
The `sentence_timing.py` stage:
1. Takes authoritative story sentences from the story_generation stage
2. Takes WhisperX word-level timestamps from the whisperx_timing stage
3. Normalizes both text sides by lowercasing and removing punctuation (but keeping alphanumerics)
4. Concatenates normalized WhisperX words into a single string (no spaces) to handle word splits/merges
5. For each story sentence, computes its normalized string and finds it as a contiguous substring in the concatenated WhisperX string
6. Maps substring bounds back to word indices using cumulative lengths
7. Derives sentence start/end times from corresponding WhisperX word timestamps
8. Preserves original story sentence text exactly and contiguous word indices
9. Fails with ValueError on substantive mismatches rather than producing incorrect timing

### Key Features
- **Punctuation/Case Tolerance**: Ignores punctuation and case differences
- **Whitespace Tolerance**: Handles varying whitespace
- **Contraction Handling**: Properly maps contractions whether split or merged by WhisperX
- **Word Split/Merge Tolerance**: Handles cases where WhisperX splits or merges words differently than the original text
- **Sentence Order Preservation**: Maintains original sentence sequence
- **Substantive Mismatch Detection**: Clearly fails when words don't match (rather than silent incorrect timing)
- **Contiguous Word Mapping**: Ensures each sentence maps to a contiguous block of word indices

### Normalization Approach
- `_normalize_for_matching()`: Lowercase, remove punctuation, remove all whitespace
- This allows "Hello, World!" to match "helloworld" and "Can't" to match "cant"
- Enables robust matching despite punctuation, case, and whitespace differences

### Alignment Algorithm
1. Build reference from WhisperX words: normalized words, cumulative lengths, concatenated string
2. For each story sentence:
   - Normalize for matching
   - Find as substring in concatenated WhisperX string (starting after previous match)
   - Map character range to word indices using cumulative lengths
   - Extract start/end times from word timestamps
   - Preserve original sentence text exactly
3. Validate no overlaps or out-of-order mappings

## Pipeline Integration
- Inserted after `whisperx_timing.stage_whisperx_timing` and before `stage_visual_planning`
- Receives story artifact (from story_generation) as input
- Loads whisperx_timing artifact from disk (config.output_dir / "whisperx_timing.json")
- Returns PipelineArtifact with sentence-level timing data

## Test Coverage
The test suite includes focused tests for:
- Basic sentence-to-word mapping
- Punctuation and case differences
- Contractions (e.g., "can't" vs "can" + "t")
- Word splits and merges (e.g., "cannot" vs "can" + "not")
- Multiple sentences in sequence
- Missing words in WhisperX output (properly raises ValueError)
- Extra words in WhisperX output (handled gracefully as trailing silence)
- Substantive mismatches (clearly fails with ValueError)
- Empty sentences
- Integration test with full stage execution
- Edge cases in character range to word index mapping

## Dependencies
- Uses only existing WhisperX installation (no new packages)
- Does not modify Story Generation, Voice Generation, WhisperX timing, or controller stages
- Preserves existing pipeline architecture and data flow

## Output Format
Returns sentence-level timing as a list of dictionaries:
```json
[
  {
    "sentence_index": 0,
    "sentence": "Original story sentence text preserved exactly",
    "start": 0.123,
    "end": 1.456,
    "word_indices": [0, 1, 2]
  }
]
```
Where:
- `sentence_index`: Position in original story sentences list
- `sentence`: Original story sentence text (exactly as provided)
- `start`: Start time in seconds (from first matched WhisperX word)
- `end`: End time in seconds (from last matched WhisperX word)
- `word_indices`: Contiguous indices into the WhisperX words array

This format is suitable for downstream animation synchronization while preserving the authoritative story text.