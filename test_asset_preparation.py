#!/usr/bin/env python3
"""
Tests for asset_preparation.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, '/root/whiteboard_anime/dynamic_history_video_project')

import asset_preparation
from asset_preparation import (
    prepare_assets,
    stage_asset_preparation,
    create_white_png,
    is_valid_png,
)


class TestCreateWhitePNG:
    """Tests for create_white_png function."""

    def test_creates_valid_png(self):
        """Created file is a valid PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            create_white_png(path)
            assert path.exists()
            assert is_valid_png(path)

    def test_png_signature(self):
        """PNG has correct signature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            create_white_png(path)
            with open(path, 'rb') as f:
                header = f.read(8)
            assert header == b'\x89PNG\r\n\x1a\n'


class TestIsValidPNG:
    """Tests for is_valid_png function."""

    def test_valid_png_returns_true(self):
        """Valid PNG returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            create_white_png(path)
            assert is_valid_png(path) is True

    def test_nonexistent_file_returns_false(self):
        """Non-existent file returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.png"
            assert is_valid_png(path) is False

    def test_invalid_file_returns_false(self):
        """Non-PNG file returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("not a png")
            assert is_valid_png(path) is False


class TestPrepareAssets:
    """Tests for prepare_assets function."""

    def _make_visual_timing(self, count: int) -> dict:
        """Create mock visual_timing with N visuals."""
        return {
            "visuals": [
                {"visual_index": i, "sentence_indices": [i], "image": f"scene_{i+1}.png", "start": 0.0, "end": 1.0, "duration": 1.0}
                for i in range(count)
            ]
        }

    def test_one_visual_creates_scene_1(self):
        """1 visual creates exactly scene 1.png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(1)

            result = prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            assert assets_dir.exists()
            assert (assets_dir / "scene 1.png").exists()
            assert not (assets_dir / "scene 2.png").exists()

            assert result["visual_count"] == 1
            assert result["scene_files"] == ["scene 1.png"]
            assert result["created"] == ["scene 1.png"]
            assert result["skipped"] == []

    def test_three_visuals_creates_three_scenes(self):
        """3 visuals create scene 1.png, scene 2.png, scene 3.png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(3)

            result = prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            for i in range(1, 4):
                assert (assets_dir / f"scene {i}.png").exists()
            assert not (assets_dir / "scene 4.png").exists()

            assert result["visual_count"] == 3
            assert result["scene_files"] == ["scene 1.png", "scene 2.png", "scene 3.png"]

    def test_ten_visuals_creates_ten_scenes(self):
        """10 visuals create 10 placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(10)

            result = prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            for i in range(1, 11):
                assert (assets_dir / f"scene {i}.png").exists()
            assert not (assets_dir / "scene 11.png").exists()

            assert result["visual_count"] == 10

    def test_no_extra_files_created(self):
        """Only exactly N scene files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(2)

            result = prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            files = list(assets_dir.glob("*.png"))
            assert len(files) == 2

    def test_placeholders_are_valid_pngs(self):
        """All created placeholders are valid PNG files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(3)

            prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            for i in range(1, 4):
                path = assets_dir / f"scene {i}.png"
                assert is_valid_png(path), f"{path} is not a valid PNG"

    def test_existing_valid_png_not_overwritten(self):
        """Existing valid PNG files are skipped, not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(2)

            # Pre-create a valid PNG
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            create_white_png(assets_dir / "scene 1.png")
            original_mtime = (assets_dir / "scene 1.png").stat().st_mtime

            result = prepare_assets(visual_timing, output_dir)

            # Should skip scene 1.png, create scene 2.png
            assert "scene 1.png" in result["skipped"]
            assert "scene 2.png" in result["created"]
            # mtime should be unchanged (not overwritten)
            assert (assets_dir / "scene 1.png").stat().st_mtime == original_mtime

    def test_existing_invalid_file_overwritten(self):
        """Existing non-PNG file is overwritten with valid PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(1)

            # Pre-create an invalid file
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            (assets_dir / "scene 1.png").write_text("not a png")

            result = prepare_assets(visual_timing, output_dir)

            # Should overwrite with valid PNG
            assert "scene 1.png" in result["created"]
            assert is_valid_png(assets_dir / "scene 1.png")

    def test_empty_visual_plan_creates_empty_assets_dir(self):
        """Empty visual plan creates assets dir with zero placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"visuals": []}

            result = prepare_assets(visual_timing, output_dir)

            assets_dir = output_dir / "assets"
            assert assets_dir.exists()
            assert len(list(assets_dir.glob("*.png"))) == 0
            assert result["visual_count"] == 0
            assert result["scene_files"] == []

    def test_filenames_derived_from_visual_count(self):
        """Filenames are dynamically derived from visual count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = self._make_visual_timing(5)

            result = prepare_assets(visual_timing, output_dir)

            assert result["scene_files"] == [
                "scene 1.png", "scene 2.png", "scene 3.png", "scene 4.png", "scene 5.png"
            ]

    def test_visual_timing_missing_visuals_key_raises(self):
        """Missing 'visuals' key raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"not_visuals": []}

            with pytest.raises(ValueError, match="missing required key: 'visuals'"):
                prepare_assets(visual_timing, output_dir)

    def test_visual_timing_visuals_not_list_raises(self):
        """'visuals' not a list raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"visuals": "not a list"}

            with pytest.raises(TypeError, match="must be a list"):
                prepare_assets(visual_timing, output_dir)


class TestStageAssetPreparation:
    """Tests for stage_asset_preparation pipeline stage."""

    def setup_method(self):
        self.config = MagicMock()
        self.config.output_dir = Path("/tmp/test")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open")
    @patch("json.load")
    def test_stage_loads_visual_timing_and_prepares(self, mock_json_load, mock_open, mock_exists):
        """Stage loads visual_timing artifact and creates assets."""
        # Mock visual_timing file content
        mock_json_load.return_value = {
            "stage": "visual_timing",
            "data": {
                "visuals": [
                    {"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0.0, "end": 1.0, "duration": 1.0},
                    {"visual_index": 1, "sentence_indices": [1], "image": "scene_2.png", "start": 1.0, "end": 2.0, "duration": 1.0},
                ]
            },
            "meta": {}
        }

        artifact = stage_asset_preparation(self.config, {})

        assert artifact.stage_name == "asset_preparation"
        assert "assets_dir" in artifact.data
        assert artifact.data["visual_count"] == 2
        assert artifact.data["scene_files"] == ["scene 1.png", "scene 2.png"]

    @patch("pathlib.Path.exists", return_value=False)
    def test_stage_missing_visual_timing_raises(self, mock_exists):
        """Stage raises FileNotFoundError if visual_timing.json missing."""
        with pytest.raises(FileNotFoundError, match="Visual timing artifact not found"):
            stage_asset_preparation(self.config, {})

    def test_stage_invalid_visual_timing_raises(self):
        """Stage raises ValueError if visual_timing missing 'visuals' field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            timing_path = Path(tmpdir) / "visual_timing.json"
            timing_path.write_text(json.dumps({"stage": "visual_timing", "data": {}, "meta": {}}))

            config = MagicMock()
            config.output_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="missing 'visuals' field"):
                stage_asset_preparation(config, {})


class TestValidationEdgeCases:
    """Additional edge case tests."""

    def test_creates_assets_dir_if_not_exists(self):
        """Assets directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"visuals": [{"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0.0, "end": 1.0, "duration": 1.0}]}

            prepare_assets(visual_timing, output_dir)

            assert (output_dir / "assets").exists()
            assert (output_dir / "assets").is_dir()

    def test_relative_assets_dir_in_result(self):
        """Result contains relative assets_dir path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"visuals": [{"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0.0, "end": 1.0, "duration": 1.0}]}

            result = prepare_assets(visual_timing, output_dir)

            assert result["assets_dir"] == "assets"

    def test_metadata_counts(self):
        """Created/skipped counts in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            visual_timing = {"visuals": [{"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0.0, "end": 1.0, "duration": 1.0}]}

            # First run - creates
            result1 = prepare_assets(visual_timing, output_dir)
            assert len(result1["created"]) == 1
            assert len(result1["skipped"]) == 0

            # Second run - skips
            result2 = prepare_assets(visual_timing, output_dir)
            assert len(result2["created"]) == 0
            assert len(result2["skipped"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])