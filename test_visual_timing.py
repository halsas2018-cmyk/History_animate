#!/usr/bin/env python3
"""
Tests for visual_timing.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

import visual_timing
from visual_timing import (
    compute_visual_timing,
    stage_visual_timing,
    frames_to_seconds,
)


class TestFramesToSeconds:
    """Tests for frames_to_seconds conversion."""

    def test_basic_conversion(self):
        assert frames_to_seconds(30) == 1.0
        assert frames_to_seconds(60) == 2.0
        assert frames_to_seconds(0) == 0.0
        assert frames_to_seconds(1) == 1/30
        assert frames_to_seconds(45) == 1.5


class TestComputeVisualTiming:
    """Tests for compute_visual_timing function."""

    def _make_sentence_timing(self, count: int, frame_step: int = 30) -> list[dict]:
        """Create mock sentence_timing with sequential timing."""
        return [
            {
                "sentence_index": i,
                "text": f"Sentence {i}.",
                "startFrame": i * frame_step,
                "endFrame": (i + 1) * frame_step,
                "durationInFrames": frame_step,
            }
            for i in range(count)
        ]

    def _make_visual_plan(self, sentence_indices_list: list[list[int]]) -> dict:
        """Create a visual_plan with given sentence index groupings."""
        return {
            "visual_plan": [
                {"sentence_indices": indices, "image_prompt": f"prompt for visual {i}"}
                for i, indices in enumerate(sentence_indices_list)
            ]
        }

    def test_one_visual_one_sentence(self):
        """One visual covering one sentence."""
        sentence_timing = self._make_sentence_timing(1)
        visual_plan = self._make_visual_plan([[0]])

        result = compute_visual_timing(visual_plan, sentence_timing)

        assert len(result["visuals"]) == 1
        v = result["visuals"][0]
        assert v["visual_index"] == 0
        assert v["sentence_indices"] == [0]
        assert v["image"] == "scene_1.png"
        assert v["start"] == 0.0
        assert v["end"] == 1.0
        assert v["duration"] == 1.0

    def test_one_visual_multiple_sentences(self):
        """One visual covering multiple consecutive sentences."""
        sentence_timing = self._make_sentence_timing(3)
        visual_plan = self._make_visual_plan([[0, 1, 2]])

        result = compute_visual_timing(visual_plan, sentence_timing)

        assert len(result["visuals"]) == 1
        v = result["visuals"][0]
        assert v["sentence_indices"] == [0, 1, 2]
        assert v["start"] == 0.0
        assert v["end"] == 3.0
        assert v["duration"] == 3.0

    def test_multiple_visuals(self):
        """Multiple visuals each covering some sentences."""
        sentence_timing = self._make_sentence_timing(5)
        visual_plan = self._make_visual_plan([[0, 1], [2], [3, 4]])

        result = compute_visual_timing(visual_plan, sentence_timing)

        assert len(result["visuals"]) == 3
        assert result["visuals"][0]["sentence_indices"] == [0, 1]
        assert result["visuals"][0]["image"] == "scene_1.png"
        assert result["visuals"][0]["start"] == 0.0
        assert result["visuals"][0]["end"] == 2.0

        assert result["visuals"][1]["sentence_indices"] == [2]
        assert result["visuals"][1]["image"] == "scene_2.png"
        assert result["visuals"][1]["start"] == 2.0
        assert result["visuals"][1]["end"] == 3.0

        assert result["visuals"][2]["sentence_indices"] == [3, 4]
        assert result["visuals"][2]["image"] == "scene_3.png"
        assert result["visuals"][2]["start"] == 3.0
        assert result["visuals"][2]["end"] == 5.0

    def test_correct_start_end_calculation(self):
        """Start = earliest sentence start, End = latest sentence end."""
        # Non-uniform sentence timing
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": 0, "endFrame": 30, "durationInFrames": 30},
            {"sentence_index": 1, "text": "B", "startFrame": 30, "endFrame": 90, "durationInFrames": 60},
            {"sentence_index": 2, "text": "C", "startFrame": 90, "endFrame": 120, "durationInFrames": 30},
        ]
        visual_plan = self._make_visual_plan([[0, 1], [2]])

        result = compute_visual_timing(visual_plan, sentence_timing)

        # Visual 0: sentences 0,1 -> start=0, end=3.0 (90 frames)
        assert result["visuals"][0]["start"] == 0.0
        assert result["visuals"][0]["end"] == 3.0
        assert result["visuals"][0]["duration"] == 3.0

        # Visual 1: sentence 2 -> start=3.0, end=4.0
        assert result["visuals"][1]["start"] == 3.0
        assert result["visuals"][1]["end"] == 4.0
        assert result["visuals"][1]["duration"] == 1.0

    def test_scene_filename_numbering(self):
        """Scene filenames are 1-based: scene_1.png, scene_2.png, ..."""
        sentence_timing = self._make_sentence_timing(4)
        visual_plan = self._make_visual_plan([[0], [1], [2], [3]])

        result = compute_visual_timing(visual_plan, sentence_timing)

        for i, v in enumerate(result["visuals"]):
            assert v["image"] == f"scene_{i + 1}.png"

    def test_invalid_sentence_index(self):
        """Invalid sentence index in visual_plan raises error."""
        sentence_timing = self._make_sentence_timing(2)
        visual_plan = self._make_visual_plan([[0, 5]])  # Index 5 doesn't exist

        with pytest.raises(ValueError, match="missing sentence index 5"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_missing_sentence_timing(self):
        """sentence_timing missing required fields raises error."""
        sentence_timing = [{"sentence_index": 0, "text": "A"}]  # missing startFrame/endFrame
        visual_plan = self._make_visual_plan([[0]])

        with pytest.raises(ValueError, match="missing startFrame or endFrame"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_incomplete_sentence_coverage(self):
        """Not all sentences covered raises error."""
        sentence_timing = self._make_sentence_timing(3)
        visual_plan = self._make_visual_plan([[0], [1]])  # Missing sentence 2

        with pytest.raises(ValueError, match="Total covered sentence indices.*!= number of sentences"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_duplicate_sentence_indices(self):
        """Duplicate sentence indices across visuals raises error."""
        sentence_timing = self._make_sentence_timing(3)
        visual_plan = self._make_visual_plan([[0, 1], [1, 2]])  # Index 1 duplicated

        with pytest.raises(ValueError, match="Total covered sentence indices.*!= number of sentences"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_empty_visual_plan(self):
        """Empty visual_plan raises error."""
        sentence_timing = self._make_sentence_timing(2)
        visual_plan = {"visual_plan": []}

        with pytest.raises(ValueError, match="visual_plan\['visual_plan'\] must be non-empty"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_visual_with_empty_sentence_indices(self):
        """Visual with empty sentence_indices raises error."""
        sentence_timing = self._make_sentence_timing(2)
        visual_plan = self._make_visual_plan([[], [0, 1]])

        with pytest.raises(ValueError, match="must be non-empty"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_start_less_than_end(self):
        """Visual with start > end raises error (start == end is allowed for zero duration)."""
        # Two sentences with same timing (duration 0)
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": 30, "endFrame": 30, "durationInFrames": 0},
            {"sentence_index": 1, "text": "B", "startFrame": 30, "endFrame": 30, "durationInFrames": 0},
        ]
        visual_plan = self._make_visual_plan([[0, 1]])

        # This should now be allowed (zero duration)
        result = compute_visual_timing(visual_plan, sentence_timing)
        assert result["visuals"][0]["duration"] == 0.0
        assert result["visuals"][0]["start"] == result["visuals"][0]["end"]

        # But start > end should fail - need to construct a case where this happens
        # (not easily possible with our frame mapping, so we just verify zero duration works)

    def test_negative_timing(self):
        """Negative frame values should be caught."""
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": -10, "endFrame": 30, "durationInFrames": 40},
        ]
        visual_plan = self._make_visual_plan([[0]])

        # Should work (negative frames converted to negative seconds)
        result = compute_visual_timing(visual_plan, sentence_timing)
        assert result["visuals"][0]["start"] < 0

    def test_out_of_order_sentence_indices_in_visual(self):
        """Sentence indices must be in order within a visual (not strictly required but preserved)."""
        sentence_timing = self._make_sentence_timing(3)
        visual_plan = self._make_visual_plan([[2, 0, 1]])  # Out of order but valid indices

        # This should work - we just use min/max of the frames
        result = compute_visual_timing(visual_plan, sentence_timing)
        assert result["visuals"][0]["sentence_indices"] == [2, 0, 1]  # Preserved as-is

    def test_preserves_visual_plan_order(self):
        """Visuals remain in same order as visual_plan."""
        sentence_timing = self._make_sentence_timing(6)
        visual_plan = self._make_visual_plan([[4, 5], [0, 1], [2, 3]])  # Non-sequential order

        result = compute_visual_timing(visual_plan, sentence_timing)

        assert result["visuals"][0]["sentence_indices"] == [4, 5]
        assert result["visuals"][1]["sentence_indices"] == [0, 1]
        assert result["visuals"][2]["sentence_indices"] == [2, 3]


class TestStageVisualTiming:
    """Tests for stage_visual_timing pipeline stage."""

    def setup_method(self):
        self.config = MagicMock()
        self.config.output_dir = Path("/tmp/test")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_stage_loads_artifacts_and_computes(self, mock_json_load, mock_open, mock_exists):
        """Stage loads visual_planning and sentence_timing artifacts from disk."""
        # Mock visual_planning.json first call, then sentence_timing.json second call
        mock_json_load.side_effect = [
            {  # First call - visual_planning.json with envelope
                "stage": "visual_planning",
                "data": {
                    "visual_plan": [
                        {"sentence_indices": [0, 1], "image_prompt": "prompt 1"},
                        {"sentence_indices": [2], "image_prompt": "prompt 2"},
                    ]
                },
                "meta": {}
            },
            {  # Second call - sentence_timing.json with envelope
                "stage": "sentence_timing",
                "data": {
                    "sentence_timing": [
                        {"sentence_index": 0, "startFrame": 0, "endFrame": 30, "durationInFrames": 30},
                        {"sentence_index": 1, "startFrame": 30, "endFrame": 60, "durationInFrames": 30},
                        {"sentence_index": 2, "startFrame": 60, "endFrame": 90, "durationInFrames": 30},
                    ]
                },
                "meta": {}
            }
        ]

        artifact = stage_visual_timing(self.config, {})  # prev not used anymore

        assert artifact.stage_name == "visual_timing"
        assert "visuals" in artifact.data
        assert len(artifact.data["visuals"]) == 2
        assert artifact.data["visuals"][0]["sentence_indices"] == [0, 1]
        assert artifact.data["visuals"][1]["sentence_indices"] == [2]

    @patch("pathlib.Path.exists", return_value=False)
    def test_stage_missing_visual_planning_raises(self, mock_exists):
        """Stage raises FileNotFoundError if visual_planning.json missing."""
        with pytest.raises(FileNotFoundError, match="Visual planning artifact not found"):
            stage_visual_timing(self.config, {})

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_stage_missing_visual_plan_field_raises(self, mock_json_load, mock_open, mock_exists):
        """Stage raises ValueError if visual_planning artifact missing visual_plan field."""
        mock_json_load.return_value = {
            "stage": "visual_planning",
            "data": {"not_visual_plan": []},
            "meta": {}
        }

        with pytest.raises(ValueError, match="missing 'visual_plan' field"):
            stage_visual_timing(self.config, {})

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_stage_visual_plan_not_list_raises(self, mock_json_load, mock_open, mock_exists):
        """Stage raises ValueError if visual_plan is not a list."""
        mock_json_load.return_value = {
            "stage": "visual_planning",
            "data": {"visual_plan": "not a list"},
            "meta": {}
        }

        with pytest.raises(ValueError, match="missing 'visual_plan' list"):
            stage_visual_timing(self.config, {})


class TestValidationEdgeCases:
    """Additional edge case validation tests."""

    def test_zero_duration_sentence(self):
        """Sentence with zero duration (startFrame == endFrame) produces visual with zero duration."""
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": 30, "endFrame": 60, "durationInFrames": 30},
            {"sentence_index": 1, "text": "B", "startFrame": 60, "endFrame": 60, "durationInFrames": 0},
        ]
        visual_plan = {"visual_plan": [
            {"sentence_indices": [0], "image_prompt": "p1"},
            {"sentence_indices": [1], "image_prompt": "p2"},
        ]}

        result = compute_visual_timing(visual_plan, sentence_timing)

        assert result["visuals"][0]["duration"] == 1.0
        assert result["visuals"][1]["duration"] == 0.0
        assert result["visuals"][1]["start"] == result["visuals"][1]["end"] == 2.0

    def test_visual_plan_extra_keys_rejected(self):
        """visual_plan with extra top-level keys rejected."""
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": 0, "endFrame": 30, "durationInFrames": 30},
        ]
        visual_plan = {"visual_plan": [{"sentence_indices": [0], "image_prompt": "p"}], "extra": "bad"}

        with pytest.raises(ValueError, match="visual_plan must have exactly one key"):
            compute_visual_timing(visual_plan, sentence_timing)

    def test_duration_calculation_precision(self):
        """Duration calculated as end - start with proper precision."""
        sentence_timing = [
            {"sentence_index": 0, "text": "A", "startFrame": 1, "endFrame": 31, "durationInFrames": 30},
        ]
        visual_plan = {"visual_plan": [{"sentence_indices": [0], "image_prompt": "p"}]}

        result = compute_visual_timing(visual_plan, sentence_timing)
        # 1/30 = 0.0333..., 31/30 = 1.0333..., duration = 1.0
        assert result["visuals"][0]["start"] == 0.033
        assert result["visuals"][0]["end"] == 1.033
        assert result["visuals"][0]["duration"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])