#!/usr/bin/env python3
"""
Tests for visual_planning.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

import visual_planning
from visual_planning import plan_visuals, validate_visual_plan_contract


class TestVisualPlanning:
    """Tests for the visual_planning module."""

    def test_validate_visual_plan_contract_valid(self):
        """Test that a valid visual plan passes validation."""
        plan = {
            "visual_plan": [
                {
                    "sentence_indices": [0, 1],
                    "image_prompt": "1080x1920 vertical composition, pure white background, a simple historical sailing ship illustration with clean black outlines, suitable for whiteboard animation"
                },
                {
                    "sentence_indices": [2],
                    "image_prompt": "1080x1920 vertical composition, pure white background, a diagram of Pompeii with Mount Vesuvius, simple line art, flat colors, whiteboard style"
                }
            ]
        }
        # Should not raise
        validate_visual_plan_contract(plan, 3)

    def test_validate_visual_plan_contract_missing_required_keywords(self):
        """Test that validation fails if required keywords are missing."""
        plan = {
            "visual_plan": [
                {
                    "sentence_indices": [0],
                    "image_prompt": "a beautiful painting of a ship"  # Missing required keywords
                }
            ]
        }
        with pytest.raises(ValueError, match="missing required keyword"):
            validate_visual_plan_contract(plan, 1)

    def test_plan_visuals_mock_llm_returns_valid_plan(self):
        """Test that plan_visuals returns a valid plan when LLM returns expected format."""
        # Mock story input
        story = {
            "title": "Test Story",
            "sentences": [
                "First sentence about a sailing ship.",
                "Second sentence continuing the ship topic.",
                "Third sentence about Pompeii."
            ]
        }

        # Mock LLM response that includes whiteboard constraints
        mock_llm_response = json.dumps({
            "visual_plan": [
                {
                    "sentence_indices": [0, 1],
                    "image_prompt": "1080x1920 vertical composition, pure white background, simple hand-drawn illustration of a historical sailing ship, clean black marker-style outlines, clear continuous outlines, distinct separated components, large recognizable shapes, generous white space, simple connected line art, small number of flat color accents, easy to trace elements, major outlines visually dominant, minimized unnecessary overlap, suitable for whiteboard stroke-by-stroke animation"
                },
                {
                    "sentence_indices": [2],
                    "image_prompt": "1080x1920 vertical composition, pure white background, educational diagram of Pompeii and Mount Vesuvius, simple line art style, clear outlines, separated visual components, flat color accents, whiteboard source image for stroke-by-stroke animation, no photorealism, no gradients, no complex textures"
                }
            ]
        })

        with patch('visual_planning.llm_client.call_llm', return_value=mock_llm_response):
            plan = plan_visuals(story)

            # Validate the plan structure
            assert "visual_plan" in plan
            assert len(plan["visual_plan"]) == 2
            assert plan["visual_plan"][0]["sentence_indices"] == [0, 1]
            assert plan["visual_plan"][1]["sentence_indices"] == [2]

            # Check that the prompts contain the required whiteboard constraints
            prompt0 = plan["visual_plan"][0]["image_prompt"].lower()
            prompt1 = plan["visual_plan"][1]["image_prompt"].lower()

            # Check for core whiteboard constraints
            assert "white background" in prompt0
            assert "white background" in prompt1
            assert "simple hand-drawn" in prompt0 or "simple line art" in prompt0
            assert "clean black" in prompt0 or "marker-style" in prompt0
            assert "clear continuous outlines" in prompt0
            assert ("distinct separated" in prompt0 or "separated visual components" in prompt0 or
                   "distinct, separated" in prompt0)
            assert "large recognizable shapes" in prompt0
            assert "generous white space" in prompt0
            assert "simple connected line art" in prompt0
            assert ("flat color accents" in prompt0 or "flat color" in prompt0)
            assert "easy to trace" in prompt0
            assert "major outlines visually dominant" in prompt0
            assert "minimized unnecessary overlap" in prompt0
            assert ("suitable for whiteboard" in prompt0 or "whiteboard stroke-by-stroke" in prompt0 or
                   "whiteboard source image" in prompt0)

            # Check for prohibitions
            assert ("no photorealism" in prompt1 or "no photographic" in prompt1 or
                   "no photorealistic" in prompt1)
            assert "no gradients" in prompt1
            assert ("no complex textures" in prompt1 or "complex textures" not in prompt1)

    def test_plan_visuals_preserves_existing_behavior(self):
        """Test that the existing required keywords are still present."""
        story = {
            "title": "Test",
            "sentences": ["Single sentence test."]
        }

        mock_llm_response = json.dumps({
            "visual_plan": [
                {
                    "sentence_indices": [0],
                    "image_prompt": "1080x1920 vertical composition, pure white background, a simple historical illustration"
                }
            ]
        })

        with patch('visual_planning.llm_client.call_llm', return_value=mock_llm_response):
            plan = plan_visuals(story)
            prompt = plan["visual_plan"][0]["image_prompt"].lower()
            assert "1080x1920 vertical composition" in prompt
            assert "pure white background" in prompt

    def test_plan_visuals_handles_empty_sentences(self):
        """Test that empty sentences list raises ValueError (as per existing validation)."""
        story = {
            "title": "Test",
            "sentences": []
        }
        with pytest.raises(ValueError, match="story\\['sentences'\\] must be a non-empty list"):
            plan_visuals(story)

    def test_plan_visuals_handles_invalid_story_structure(self):
        """Test that invalid story structure raises appropriate errors."""
        # Missing title
        story = {"sentences": ["test"]}
        with pytest.raises(ValueError, match="story\\['title'\\] must be a non-empty string"):
            plan_visuals(story)

        # Missing sentences
        story = {"title": "test"}
        with pytest.raises(ValueError, match="story\\['sentences'\\] must be a non-empty list"):
            plan_visuals(story)

        # Non-string title
        story = {"title": 123, "sentences": ["test"]}
        with pytest.raises(ValueError, match="story\\['title'\\] must be a non-empty string"):
            plan_visuals(story)

        # Non-string sentence
        story = {"title": "test", "sentences": [123]}
        with pytest.raises(ValueError, match="Sentence 0 must be a non-empty string"):
            plan_visuals(story)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])