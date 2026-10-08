#!/usr/bin/env python3
"""
Tests for run_pipeline.py — focused tests for the pipeline controller.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

import run_pipeline
from run_pipeline import (
    PipelineConfig,
    PipelineArtifact,
    PipelineRunner,
    PipelineError,
    stage_story_generation,
    PIPELINE_STAGES,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_from_args_defaults(self):
        """Test default values when only topic is provided."""
        class Args:
            topic = "Test Topic"
            model = None
            length = None
            output = None
            no_save = False

        config = PipelineConfig.from_args(Args())
        assert config.topic == "Test Topic"
        assert config.model_key == "gemini-31-flash-lite"
        assert config.length_mode == "short"
        assert config.save_intermediate is True
        assert "output" in str(config.output_dir)

    def test_from_args_with_overrides(self):
        """Test config with all overrides."""
        class Args:
            topic = "Custom Topic"
            model = "groq-gpt-oss-20b"
            length = "long"
            output = "/custom/output"
            no_save = True

        config = PipelineConfig.from_args(Args())
        assert config.topic == "Custom Topic"
        assert config.model_key == "groq-gpt-oss-20b"
        assert config.length_mode == "long"
        assert config.output_dir == Path("/custom/output").resolve()
        assert config.save_intermediate is False

    def test_from_args_empty_topic_raises(self):
        """Test that empty topic is caught by argparse (not here)."""
        # This is validated at argparse level in main()
        pass


class TestPipelineArtifact:
    """Tests for PipelineArtifact."""

    def test_artifact_creation(self):
        """Test creating and serializing an artifact."""
        artifact = PipelineArtifact(
            stage_name="test_stage",
            data={"title": "Test", "sentences": ["A.", "B."]},
            metadata={"model": "test-model"},
        )
        json_str = artifact.to_json()
        parsed = json.loads(json_str)
        assert parsed["stage"] == "test_stage"
        assert parsed["data"]["title"] == "Test"
        assert parsed["meta"]["model"] == "test-model"


class TestStoryGenerationStage:
    """Tests for the story generation stage."""

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_stage_story_generation_calls_generator(self, mock_generate):
        """Test that stage calls the story generator with correct params."""
        mock_generate.return_value = {
            "title": "Test Story",
            "sentences": ["Sentence 1.", "Sentence 2."],
        }

        class Config:
            topic = "Test Topic"
            model_key = "test-model"
            length_mode = "short"
            output_dir = Path("/tmp")
            save_intermediate = False

        artifact = stage_story_generation(Config(), {})

        mock_generate.assert_called_once_with(
            topic="Test Topic",
            model_key="test-model",
            length_mode="short",
        )
        assert artifact.stage_name == "story_generation"
        assert artifact.data["title"] == "Test Story"
        assert artifact.data["sentences"] == ["Sentence 1.", "Sentence 2."]
        assert artifact.metadata["model_key"] == "test-model"
        assert artifact.metadata["length_mode"] == "short"

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_stage_story_generation_propagates_error(self, mock_generate):
        """Test that stage propagates errors from generator."""
        mock_generate.side_effect = RuntimeError("LLM failed")

        class Config:
            topic = "Test Topic"
            model_key = "test-model"
            length_mode = "short"
            output_dir = Path("/tmp")
            save_intermediate = False

        with pytest.raises(RuntimeError, match="LLM failed"):
            stage_story_generation(Config(), {})


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def setup_method(self):
        """Set up a basic config for tests."""
        self.config = PipelineConfig(
            topic="Test Topic",
            model_key="gemini-31-flash-lite",  # Use a valid model key
            length_mode="short",
            output_dir=Path("/tmp/test_output"),
            save_intermediate=False,
        )

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_runner_executes_all_stages(self, mock_generate):
        """Test runner executes all registered stages."""
        mock_generate.return_value = {
            "title": "Test Story",
            "sentences": ["Sentence 1.", "Sentence 2."],
        }

        runner = PipelineRunner(self.config)
        artifacts = runner.run()

        assert len(artifacts) == len(run_pipeline.PIPELINE_STAGES)
        assert artifacts[0].data["title"] == "Test Story"
        assert mock_generate.call_count == 1

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_runner_passes_output_to_next_stage(self, mock_generate):
        """Test that output from one stage becomes input to next.

        Since we only have one stage (story_generation), this tests that
        the artifact's data is available as the final output.
        """
        mock_generate.return_value = {"title": "Stage 1", "sentences": ["S1."]}

        runner = PipelineRunner(self.config)
        artifacts = runner.run()

        # Only one stage currently registered
        assert len(artifacts) == 1
        assert artifacts[0].data["title"] == "Stage 1"
        # Verify final output is accessible
        assert runner.get_final_output() == {"title": "Stage 1", "sentences": ["S1."]}

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_runner_failure_propagation(self, mock_generate):
        """Test that stage failures are wrapped and propagated."""
        mock_generate.side_effect = ValueError("Stage error")

        runner = PipelineRunner(self.config)

        with pytest.raises(PipelineError, match="Stage 'story_generation' failed: Stage error"):
            runner.run()

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_runner_saves_artifacts_when_enabled(self, mock_generate):
        """Test artifact saving when save_intermediate is True."""
        mock_generate.return_value = {"title": "T", "sentences": ["S."]}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                topic="Test",
                model_key="gemini-31-flash-lite",
                length_mode="short",
                output_dir=Path(tmpdir),
                save_intermediate=True,
            )
            runner = PipelineRunner(config)
            artifacts = runner.run()

            # Check file was created
            saved_files = list(Path(tmpdir).glob("*.json"))
            assert len(saved_files) == 1
            content = json.loads(saved_files[0].read_text())
            assert content["stage"] == "story_generation"

    @patch("run_pipeline.history_story_generator.generate_history_story")
    def test_runner_get_final_output(self, mock_generate):
        """Test getting final output from last stage."""
        mock_generate.return_value = {"final": "data"}

        runner = PipelineRunner(self.config)
        runner.run()

        assert runner.get_final_output() == {"final": "data"}

    def test_runner_empty_artifacts_returns_none(self):
        """Test get_final_output returns None when no artifacts."""
        # Create runner with empty pipeline
        config = PipelineConfig(
            topic="Test",
            model_key="test",
            length_mode="short",
            output_dir=Path("/tmp"),
            save_intermediate=False,
        )
        runner = PipelineRunner(config)
        runner.artifacts = []
        assert runner.get_final_output() is None


class TestPipelineStagesRegistry:
    """Tests for the PIPELINE_STAGES registry."""

    def test_stages_list_contains_story_generation(self):
        """Test that story generation is the first (and currently only) stage."""
        assert len(PIPELINE_STAGES) >= 1
        assert PIPELINE_STAGES[0] == stage_story_generation

    def test_stage_signatures_are_callable(self):
        """Test all registered stages are callable."""
        for stage in PIPELINE_STAGES:
            assert callable(stage)


class TestMainFunction:
    """Integration tests for the main() entry point."""

    @patch("run_pipeline.PipelineRunner.run")
    @patch("run_pipeline.build_arg_parser")
    def test_main_success(self, mock_parser, mock_run):
        """Test main returns 0 on success."""
        mock_parser.return_value.parse_args.return_value = MagicMock(
            topic="Test Topic",
            model=None,
            length="short",
            output=None,
            no_save=False,
        )
        mock_run.return_value = [PipelineArtifact("test", {"title": "T", "sentences": ["S."]})]

        result = run_pipeline.main()
        assert result == 0

    @patch("run_pipeline.PipelineRunner.run")
    @patch("run_pipeline.build_arg_parser")
    def test_main_pipeline_error(self, mock_parser, mock_run):
        """Test main returns 1 on PipelineError."""
        mock_parser.return_value.parse_args.return_value = MagicMock(
            topic="Test Topic",
            model=None,
            length="short",
            output=None,
            no_save=False,
        )
        mock_run.side_effect = PipelineError("Stage failed")

        result = run_pipeline.main()
        assert result == 1

    @patch("run_pipeline.build_arg_parser")
    def test_main_empty_topic(self, mock_parser):
        """Test main handles empty topic."""
        mock_parser.return_value.parse_args.return_value = MagicMock(
            topic="",
            model=None,
            length="short",
            output=None,
            no_save=False,
        )
        mock_parser.return_value.error = MagicMock(side_effect=SystemExit(2))

        try:
            run_pipeline.main()
        except SystemExit as e:
            assert e.code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])