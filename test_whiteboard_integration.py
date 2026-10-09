#!/usr/bin/env python3
"""
test_whiteboard_integration.py — Unit tests for Whiteboard Animator Integration

Tests cover:
- 1 visual
- 3 visuals
- arbitrary/dynamic visual count
- correct scene-to-image mapping
- correct duration passed to Whiteboard Animator
- missing image
- invalid timing
- output creation
- no scene concatenation
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from whiteboard_integration import (
    WhiteboardIntegrationError,
    validate_visual_timing,
    validate_assets_dir,
    run_whiteboard_animate,
    process_run_directory,
    detect_image_format,
    get_image_for_whiteboard,
)


def create_valid_png(path: Path) -> None:
    """Create a minimal valid PNG file at the given path using PIL."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='white')
    img.save(path, format='PNG')


def create_valid_jpeg(path: Path) -> None:
    """Create a minimal valid JPEG file at the given path using PIL."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img.save(path, format='JPEG')


def create_corrupt_image(path: Path) -> None:
    """Create a corrupt/invalid image file."""
    path.write_bytes(b"not a valid image at all")


class TestValidateVisualTiming(unittest.TestCase):
    """Tests for validate_visual_timing function."""

    def test_single_visual_valid(self):
        """Test validation passes for a single valid visual."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0, 1],
                    "image": "scene_1.png",
                    "start": 0.0,
                    "end": 10.5,
                    "duration": 10.5,
                }
            ]
        }
        result = validate_visual_timing(visual_timing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["visual_index"], 0)

    def test_three_visuals_valid(self):
        """Test validation passes for three valid visuals."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0, 1],
                    "image": "scene_1.png",
                    "start": 0.0,
                    "end": 10.5,
                    "duration": 10.5,
                },
                {
                    "visual_index": 1,
                    "sentence_indices": [2, 3],
                    "image": "scene_2.png",
                    "start": 10.5,
                    "end": 25.0,
                    "duration": 14.5,
                },
                {
                    "visual_index": 2,
                    "sentence_indices": [4],
                    "image": "scene_3.png",
                    "start": 25.0,
                    "end": 32.0,
                    "duration": 7.0,
                },
            ]
        }
        result = validate_visual_timing(visual_timing)
        self.assertEqual(len(result), 3)

    def test_arbitrary_visual_count(self):
        """Test validation works for arbitrary number of visuals."""
        visuals = []
        for i in range(10):
            visuals.append({
                "visual_index": i,
                "sentence_indices": [i],
                "image": f"scene_{i + 1}.png",
                "start": float(i * 5),
                "end": float((i + 1) * 5),
                "duration": 5.0,
            })
        visual_timing = {"visuals": visuals}
        result = validate_visual_timing(visual_timing)
        self.assertEqual(len(result), 10)

    def test_missing_visuals_key(self):
        """Test error when 'visuals' key is missing."""
        visual_timing = {"data": []}

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("missing required key: 'visuals'", str(ctx.exception))

    def test_visuals_not_a_list(self):
        """Test error when 'visuals' is not a list."""
        visual_timing = {"visuals": "not a list"}

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("'visuals' must be a list", str(ctx.exception))

    def test_empty_visuals_list(self):
        """Test error when 'visuals' list is empty."""
        visual_timing = {"visuals": []}

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("must not be empty", str(ctx.exception))

    def test_visual_not_an_object(self):
        """Test error when a visual entry is not an object."""
        visual_timing = {"visuals": ["not an object"]}

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("visual[0] must be an object", str(ctx.exception))

    def test_missing_required_keys(self):
        """Test error when required keys are missing."""
        visual_timing = {
            "visuals": [
                {"visual_index": 0, "image": "scene_1.png"}  # missing start, end, duration
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("missing required key", str(ctx.exception))

    def test_visual_index_mismatch(self):
        """Test error when visual_index doesn't match position."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 5,  # wrong index
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": 0.0,
                    "end": 5.0,
                    "duration": 5.0,
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("visual_index mismatch", str(ctx.exception))

    def test_invalid_image_format(self):
        """Test error when image filename format is invalid."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "invalid.png",  # wrong format
                    "start": 0.0,
                    "end": 5.0,
                    "duration": 5.0,
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("must be in format 'scene_N.png'", str(ctx.exception))

    def test_non_numeric_timing(self):
        """Test error when timing values are not numbers."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": "0.0",  # string instead of number
                    "end": 5.0,
                    "duration": 5.0,
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("must be a number", str(ctx.exception))

    def test_non_positive_duration(self):
        """Test error when duration is not positive."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": 0.0,
                    "end": 5.0,
                    "duration": 0,  # not positive
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("'duration' must be positive", str(ctx.exception))

    def test_negative_start(self):
        """Test error when start is negative."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": -1.0,
                    "end": 5.0,
                    "duration": 6.0,
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("'start' must be non-negative", str(ctx.exception))

    def test_end_not_greater_than_start(self):
        """Test error when end <= start."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": 5.0,
                    "end": 5.0,
                    "duration": 0.0,
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("'end' (5.0) must be > 'start' (5.0)", str(ctx.exception))

    def test_duration_mismatch(self):
        """Test error when duration != end - start."""
        visual_timing = {
            "visuals": [
                {
                    "visual_index": 0,
                    "sentence_indices": [0],
                    "image": "scene_1.png",
                    "start": 0.0,
                    "end": 10.0,
                    "duration": 5.0,  # should be 10.0
                }
            ]
        }

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_visual_timing(visual_timing)
        self.assertIn("'duration' (5.0) != 'end' - 'start'", str(ctx.exception))


class TestValidateAssetsDir(unittest.TestCase):
    """Tests for validate_assets_dir function."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.assets_dir = Path(self.temp_dir) / "assets"
        self.assets_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_image(self):
        """Test validation passes for single image."""
        create_valid_png(self.assets_dir / "scene 1.png")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}
        ]
        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "scene 1.png")

    def test_three_images(self):
        """Test validation passes for three images."""
        for i in range(1, 4):
            create_valid_png(self.assets_dir / f"scene {i}.png")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
            {"visual_index": 1, "image": "scene_2.png", "start": 5, "end": 15, "duration": 10},
            {"visual_index": 2, "image": "scene_3.png", "start": 15, "end": 22, "duration": 7},
        ]
        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 3)
        self.assertEqual(paths[0].name, "scene 1.png")
        self.assertEqual(paths[1].name, "scene 2.png")
        self.assertEqual(paths[2].name, "scene 3.png")

    def test_arbitrary_image_count(self):
        """Test validation works for arbitrary number of images."""
        visuals = []
        for i in range(1, 8):
            create_valid_png(self.assets_dir / f"scene {i}.png")
            visuals.append({
                "visual_index": i - 1,
                "image": f"scene_{i}.png",
                "start": float((i - 1) * 5),
                "end": float(i * 5),
                "duration": 5.0,
            })
        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 7)

    def test_missing_assets_dir(self):
        """Test error when assets directory doesn't exist."""
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir / "nonexistent", visuals)
        self.assertIn("Assets directory not found", str(ctx.exception))

    def test_missing_image_file(self):
        """Test error when expected image is missing."""
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}
        ]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir, visuals)
        self.assertIn("Missing image file", str(ctx.exception))

    def test_invalid_png_signature(self):
        """Test error when image is not a valid PNG."""
        # Create a non-PNG file
        (self.assets_dir / "scene 1.png").write_bytes(b"not a png")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}
        ]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir, visuals)
        self.assertIn("Invalid PNG file (bad signature)", str(ctx.exception))

    def test_unexpected_files_in_assets(self):
        """Test error when assets directory contains unexpected files."""
        create_valid_png(self.assets_dir / "scene 1.png")
        create_valid_png(self.assets_dir / "scene 2.png")
        # Add unexpected file
        (self.assets_dir / "extra.png").write_bytes(b"extra")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
            {"visual_index": 1, "image": "scene_2.png", "start": 5, "end": 10, "duration": 5},
        ]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir, visuals)
        self.assertIn("Unexpected files in assets directory", str(ctx.exception))
        self.assertIn("extra.png", str(ctx.exception))

    def test_scene_to_image_mapping(self):
        """Test correct scene-to-image mapping (scene_N.png -> scene N.png)."""
        create_valid_png(self.assets_dir / "scene 1.png")
        create_valid_png(self.assets_dir / "scene 2.png")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
            {"visual_index": 1, "image": "scene_2.png", "start": 5, "end": 10, "duration": 5},
        ]
        paths = validate_assets_dir(self.assets_dir, visuals)
        # Verify the mapping: scene_1.png -> scene 1.png, scene_2.png -> scene 2.png
        self.assertEqual(paths[0].name, "scene 1.png")
        self.assertEqual(paths[1].name, "scene 2.png")


class TestRunWhiteboardAnimate(unittest.TestCase):
    """Tests for run_whiteboard_animate function."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_path = Path(self.temp_dir) / "test.png"
        self.output_path = Path(self.temp_dir) / "output.mp4"
        create_valid_png(self.image_path)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("whiteboard_integration.WhiteboardAnimator")
    def test_correct_duration_passed(self, mock_animator_cls):
        """Test that correct duration and 92% draw_duration are passed to WhiteboardAnimator."""
        mock_animator = MagicMock()
        mock_animator_cls.return_value = mock_animator

        def side_effect(*args, **kwargs):
            self.output_path.write_bytes(b"fake mp4")
        mock_animator.render_to_file.side_effect = side_effect

        duration = 15.5
        run_whiteboard_animate(self.image_path, self.output_path, duration)

        mock_animator.render_to_file.assert_called_once()
        args, kwargs = mock_animator.render_to_file.call_args
        # args: (img_array, draw_duration, total_duration, output_path)
        draw_duration = args[1]
        total_duration = args[2]
        output_path_str = args[3]

        self.assertAlmostEqual(draw_duration, duration * 0.92)
        self.assertEqual(total_duration, duration)
        self.assertEqual(output_path_str, str(self.output_path))

    @patch("whiteboard_integration.WhiteboardAnimator")
    def test_render_failure_raises_error(self, mock_animator_cls):
        """Test that render_to_file failure raises WhiteboardIntegrationError."""
        mock_animator = MagicMock()
        mock_animator_cls.return_value = mock_animator
        mock_animator.render_to_file.side_effect = RuntimeError("Rendering engine error")

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            run_whiteboard_animate(self.image_path, self.output_path, 10.0)
        self.assertIn("WhiteboardAnimator render failed", str(ctx.exception))
        self.assertIn("Rendering engine error", str(ctx.exception))

    @patch("whiteboard_integration.WhiteboardAnimator")
    def test_output_not_created_raises_error(self, mock_animator_cls):
        """Test that missing output file raises error."""
        mock_animator = MagicMock()
        mock_animator_cls.return_value = mock_animator

        # Don't create output file in mock
        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            run_whiteboard_animate(self.image_path, self.output_path, 10.0)
        self.assertIn("Output MP4 not created", str(ctx.exception))

    @patch("whiteboard_integration.WhiteboardAnimator")
    def test_empty_output_raises_error(self, mock_animator_cls):
        """Test that empty output file raises error."""
        mock_animator = MagicMock()
        mock_animator_cls.return_value = mock_animator

        # Create empty output file
        def side_effect(*args, **kwargs):
            self.output_path.write_bytes(b"")
        mock_animator.render_to_file.side_effect = side_effect

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            run_whiteboard_animate(self.image_path, self.output_path, 10.0)
        self.assertIn("Output MP4 is empty", str(ctx.exception))

    @patch("whiteboard_integration.WhiteboardAnimator")
    def test_creates_output_directory(self, mock_animator_cls):
        """Test that output directory is created if it doesn't exist."""
        mock_animator = MagicMock()
        mock_animator_cls.return_value = mock_animator

        nested_output = Path(self.temp_dir) / "nested" / "dir" / "output.mp4"

        def side_effect(*args, **kwargs):
            nested_output.write_bytes(b"fake mp4")
        mock_animator.render_to_file.side_effect = side_effect

        run_whiteboard_animate(self.image_path, nested_output, 10.0)

        self.assertTrue(nested_output.parent.exists())



class TestProcessRunDirectory(unittest.TestCase):
    """Tests for process_run_directory function."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "test_run"
        self.run_dir.mkdir()
        self.assets_dir = self.run_dir / "assets"
        self.assets_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_setup(self, num_visuals: int = 3) -> dict:
        """Create a valid test setup with N visuals."""
        visuals = []
        for i in range(num_visuals):
            create_valid_png(self.assets_dir / f"scene {i + 1}.png")
            visuals.append({
                "visual_index": i,
                "sentence_indices": [i],
                "image": f"scene_{i + 1}.png",
                "start": float(i * 10),
                "end": float((i + 1) * 10),
                "duration": 10.0,
            })
        visual_timing = {"visuals": visuals}
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))
        return visual_timing

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_single_visual(self, mock_animate):
        """Test processing a run directory with 1 visual."""
        self.create_test_setup(1)
        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 1)
        self.assertEqual(output_paths[0].name, "scene 1.mp4")
        self.assertTrue(output_paths[0].parent.name == "whiteboard")
        mock_animate.assert_called_once()

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_three_visuals(self, mock_animate):
        """Test processing a run directory with 3 visuals."""
        self.create_test_setup(3)
        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 3)
        self.assertEqual(output_paths[0].name, "scene 1.mp4")
        self.assertEqual(output_paths[1].name, "scene 2.mp4")
        self.assertEqual(output_paths[2].name, "scene 3.mp4")
        self.assertEqual(mock_animate.call_count, 3)

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_arbitrary_visual_count(self, mock_animate):
        """Test processing a run directory with arbitrary visual count."""
        self.create_test_setup(7)
        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 7)
        for i, path in enumerate(output_paths):
            self.assertEqual(path.name, f"scene {i + 1}.mp4")
        self.assertEqual(mock_animate.call_count, 7)

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_correct_scene_to_image_mapping(self, mock_animate):
        """Test correct scene-to-image mapping is preserved."""
        self.create_test_setup(3)

        process_run_directory(self.run_dir)

        # Verify calls were made with correct image paths in order
        calls = mock_animate.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0][0].name, "scene 1.png")
        self.assertEqual(calls[1][0][0].name, "scene 2.png")
        self.assertEqual(calls[2][0][0].name, "scene 3.png")

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_correct_duration_passed_per_scene(self, mock_animate):
        """Test correct duration is passed for each scene."""
        visual_timing = {
            "visuals": [
                {"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0, "end": 5.5, "duration": 5.5},
                {"visual_index": 1, "sentence_indices": [1], "image": "scene_2.png", "start": 5.5, "end": 18.2, "duration": 12.7},
                {"visual_index": 2, "sentence_indices": [2], "image": "scene_3.png", "start": 18.2, "end": 25.0, "duration": 6.8},
            ]
        }
        for i in range(3):
            create_valid_png(self.assets_dir / f"scene {i + 1}.png")
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))

        process_run_directory(self.run_dir)

        calls = mock_animate.call_args_list
        self.assertEqual(calls[0][0][2], 5.5)
        self.assertEqual(calls[1][0][2], 12.7)
        self.assertEqual(calls[2][0][2], 6.8)

    def test_missing_visual_timing_json(self):
        """Test error when visual_timing.json is missing."""
        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            process_run_directory(self.run_dir)
        self.assertIn("visual_timing.json not found", str(ctx.exception))

    def test_invalid_visual_timing_json(self):
        """Test error when visual_timing.json is invalid JSON."""
        (self.run_dir / "visual_timing.json").write_text("{ invalid json")

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            process_run_directory(self.run_dir)
        self.assertIn("Failed to parse visual_timing.json", str(ctx.exception))

    def test_missing_assets_dir(self):
        """Test error when assets directory is missing."""
        self.create_test_setup(1)
        import shutil
        shutil.rmtree(self.assets_dir)

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            process_run_directory(self.run_dir)
        self.assertIn("Assets directory not found", str(ctx.exception))

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_missing_image_file(self, mock_animate):
        """Test error when expected image file is missing."""
        self.create_test_setup(3)
        # Remove one image
        (self.assets_dir / "scene 2.png").unlink()

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            process_run_directory(self.run_dir)
        # Note: We don't validate assets here since the function should have failed earlier

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_invalid_timing_data(self, mock_animate):
        """Test error when timing data is invalid."""
        # Duration doesn't match end - start
        visual_timing = {
            "visuals": [
                {"visual_index": 0, "sentence_indices": [0], "image": "scene_1.png", "start": 0, "end": 10, "duration": 5},
            ]
        }
        create_valid_png(self.assets_dir / "scene 1.png")
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            process_run_directory(self.run_dir)
        self.assertIn("'duration' (5) != 'end' - 'start'", str(ctx.exception))

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_output_mp4s_created(self, mock_animate):
        """Test that output MP4s are created in whiteboard subdirectory."""
        self.create_test_setup(3)

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")

        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        whiteboard_dir = self.run_dir / "whiteboard"
        self.assertTrue(whiteboard_dir.exists())
        self.assertTrue(whiteboard_dir.is_dir())

        for path in output_paths:
            self.assertTrue(path.exists())
            self.assertTrue(path.stat().st_size > 0)
            self.assertEqual(path.parent, whiteboard_dir)

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_no_scene_concatenation(self, mock_animate):
        """Test that scene MP4s are NOT concatenated (separate files created)."""
        self.create_test_setup(3)

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")

        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        # Verify separate files exist, not a single concatenated file
        self.assertEqual(len(output_paths), 3)
        for path in output_paths:
            self.assertTrue(path.exists())
            self.assertTrue(path.name.startswith("scene "))
            self.assertTrue(path.name.endswith(".mp4"))

        # Verify no concatenated file was created
        concat_path = self.run_dir / "whiteboard" / "concatenated.mp4"
        self.assertFalse(concat_path.exists())
        final_path = self.run_dir / "whiteboard" / "final.mp4"
        self.assertFalse(final_path.exists())

    def test_preserves_dynamic_scene_order(self):
        """Test that scene order matches visual_timing.json order."""
        # Create visuals in a specific order
        visual_timing = {
            "visuals": [
                {"visual_index": 0, "sentence_indices": [2], "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
                {"visual_index": 1, "sentence_indices": [0], "image": "scene_2.png", "start": 5, "end": 10, "duration": 5},
                {"visual_index": 2, "sentence_indices": [1], "image": "scene_3.png", "start": 10, "end": 15, "duration": 5},
            ]
        }
        for i in range(3):
            create_valid_png(self.assets_dir / f"scene {i + 1}.png")
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))

        with patch("whiteboard_integration.run_whiteboard_animate") as mock_animate:
            def mock_animate_side_effect(image_path, output_path, duration):
                output_path.write_bytes(b"fake mp4 content")
            mock_animate.side_effect = mock_animate_side_effect

            output_paths = process_run_directory(self.run_dir)

            # Output order must match visual_timing.json order (visual_index order)
            self.assertEqual(output_paths[0].name, "scene 1.mp4")
            self.assertEqual(output_paths[1].name, "scene 2.mp4")
            self.assertEqual(output_paths[2].name, "scene 3.mp4")


class TestCLI(unittest.TestCase):
    """Test the CLI entry point."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "test_run"
        self.run_dir.mkdir()
        self.assets_dir = self.run_dir / "assets"
        self.assets_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_setup(self, num_visuals: int = 2):
        """Create a valid test setup with N visuals."""
        visuals = []
        for i in range(num_visuals):
            create_valid_png(self.assets_dir / f"scene {i + 1}.png")
            visuals.append({
                "visual_index": i,
                "sentence_indices": [i],
                "image": f"scene_{i + 1}.png",
                "start": float(i * 10),
                "end": float((i + 1) * 10),
                "duration": 10.0,
            })
        visual_timing = {"visuals": visuals}
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_cli_success(self, mock_animate):
        """Test CLI runs successfully."""
        self.create_test_setup(2)

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        # Test the main function directly
        import whiteboard_integration
        from io import StringIO
        import sys

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            whiteboard_integration.main([str(self.run_dir)])
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("Successfully generated MP4s", output)


class TestImageFormatDetection(unittest.TestCase):
    """Tests for image format detection."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_png_format(self):
        """Test that a real PNG file is detected as PNG."""
        png_path = Path(self.temp_dir) / "test.png"
        create_valid_png(png_path)

        fmt = detect_image_format(png_path)
        self.assertEqual(fmt, 'PNG')

    def test_detect_jpeg_format(self):
        """Test that a real JPEG file is detected as JPEG."""
        jpeg_path = Path(self.temp_dir) / "test.jpg"
        create_valid_jpeg(jpeg_path)

        fmt = detect_image_format(jpeg_path)
        self.assertEqual(fmt, 'JPEG')

    def test_detect_corrupt_image(self):
        """Test that corrupt image raises error."""
        corrupt_path = Path(self.temp_dir) / "corrupt.png"
        create_corrupt_image(corrupt_path)

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            detect_image_format(corrupt_path)
        self.assertIn("Cannot read image file", str(ctx.exception))

    def test_get_image_for_whiteboard_png_passed_directly(self):
        """Test that PNG file is passed directly (no conversion)."""
        png_path = Path(self.temp_dir) / "test.png"
        create_valid_png(png_path)

        wb_path = get_image_for_whiteboard(png_path)

        self.assertEqual(wb_path, png_path)

    def test_get_image_for_whiteboard_jpeg_passed_directly(self):
        """Test that JPEG file is passed directly (no conversion)."""
        jpeg_path = Path(self.temp_dir) / "test.jpg"
        create_valid_jpeg(jpeg_path)

        wb_path = get_image_for_whiteboard(jpeg_path)

        self.assertEqual(wb_path, jpeg_path)
        # Verify the file is indeed JPEG
        from PIL import Image
        with Image.open(wb_path) as img:
            self.assertEqual(img.format, 'JPEG')

    def test_original_file_not_modified(self):
        """Test that original file is not modified."""
        jpeg_path = Path(self.temp_dir) / "test.jpg"
        create_valid_jpeg(jpeg_path)

        # Get original size
        original_size = jpeg_path.stat().st_size
        original_mtime = jpeg_path.stat().st_mtime

        wb_path = get_image_for_whiteboard(jpeg_path)

        # Original file should be unchanged
        self.assertEqual(jpeg_path.stat().st_size, original_size)
        self.assertEqual(jpeg_path.stat().st_mtime, original_mtime)
        # Original should still be JPEG
        from PIL import Image
        with Image.open(jpeg_path) as img:
            self.assertEqual(img.format, 'JPEG')

    def test_unsupported_format_error(self):
        """Test that unsupported image format produces clear error."""
        # Create a GIF file
        from PIL import Image
        gif_path = Path(self.temp_dir) / "test.gif"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(gif_path, format='GIF')

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            get_image_for_whiteboard(gif_path)
        self.assertIn("Unsupported image format", str(ctx.exception))
        self.assertIn("Only PNG and JPEG are supported", str(ctx.exception))


class TestValidateAssetsDirWithMixedFormats(unittest.TestCase):
    """Tests for validate_assets_dir with mixed PNG/JPEG formats."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.assets_dir = Path(self.temp_dir) / "assets"
        self.assets_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_png_accepted(self):
        """Test that PNG files are accepted."""
        create_valid_png(self.assets_dir / "scene 1.png")
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]

        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "scene 1.png")

    def test_jpeg_accepted(self):
        """Test that JPEG files are accepted."""
        create_valid_jpeg(self.assets_dir / "scene 1.png")  # filename .png but content JPEG
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]

        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].name, "scene 1.png")

    def test_jpeg_extension_accepted(self):
        """Test that .jpg extension files are accepted."""
        create_valid_jpeg(self.assets_dir / "scene 1.png")
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]
        
        
        

        
        
        
            
        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 1)

    def test_jpeg_filename_scene_2_jpg(self):
        """Test JPEG with .jpg filename matching visual_timing expectation."""
        # visual_timing references scene_1.png, scene_2.png -> maps to scene 1.png, scene 2.png
        # But we can have scene 1.png (PNG) and scene 2.jpg (JPEG)
        create_valid_png(self.assets_dir / "scene 1.png")
        create_valid_jpeg(self.assets_dir / "scene 2.jpg")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
            {"visual_index": 1, "image": "scene_2.png", "start": 5, "end": 10, "duration": 5},
        ]

        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 2)

    def test_mixed_extensions_dynamic(self):
        """Test dynamic mixed extensions: scene 1.png, scene 2.jpg, scene 3.jpeg"""
        create_valid_png(self.assets_dir / "scene 1.png")
        create_valid_jpeg(self.assets_dir / "scene 2.jpg")
        create_valid_jpeg(self.assets_dir / "scene 3.jpeg")
        visuals = [
            {"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5},
            {"visual_index": 1, "image": "scene_2.png", "start": 5, "end": 10, "duration": 5},
            {"visual_index": 2, "image": "scene_3.png", "start": 10, "end": 15, "duration": 5},
        ]

        paths = validate_assets_dir(self.assets_dir, visuals)
        self.assertEqual(len(paths), 3)

    def test_corrupt_image_rejected(self):
        """Test that corrupt image content is rejected even with .png extension."""
        create_corrupt_image(self.assets_dir / "scene 1.png")
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir, visuals)
        self.assertIn("Invalid PNG file (bad signature)", str(ctx.exception))

    def test_duplicate_scene_extensions_rejected(self):
        """Test that duplicate scene files (both .png and .jpg) are rejected."""
        create_valid_png(self.assets_dir / "scene 1.png")
        create_valid_jpeg(self.assets_dir / "scene 1.jpg")
        visuals = [{"visual_index": 0, "image": "scene_1.png", "start": 0, "end": 5, "duration": 5}]

        with self.assertRaises(WhiteboardIntegrationError) as ctx:
            validate_assets_dir(self.assets_dir, visuals)
        self.assertIn("Duplicate scene files", str(ctx.exception))
        self.assertIn("scene 1", str(ctx.exception))


class TestProcessRunDirectoryWithMixedFormats(unittest.TestCase):
    """Tests for process_run_directory with mixed image formats."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.temp_dir) / "test_run"
        self.run_dir.mkdir()
        self.assets_dir = self.run_dir / "assets"
        self.assets_dir.mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_setup(self, num_visuals: int = 3, formats: list = None):
        """Create a valid test setup with N visuals and specified formats."""
        if formats is None:
            formats = ['PNG'] * num_visuals

        visuals = []
        for i in range(num_visuals):
            if formats[i] == 'PNG':
                create_valid_png(self.assets_dir / f"scene {i + 1}.png")
            elif formats[i] == 'JPEG':
                create_valid_jpeg(self.assets_dir / f"scene {i + 1}.png")
            visuals.append({
                "visual_index": i,
                "sentence_indices": [i],
                "image": f"scene_{i + 1}.png",
                "start": float(i * 10),
                "end": float((i + 1) * 10),
                "duration": 10.0,
            })
        visual_timing = {"visuals": visuals}
        (self.run_dir / "visual_timing.json").write_text(json.dumps(visual_timing))
        return visual_timing

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_png_passed_directly_to_whiteboard(self, mock_animate):
        """Test that real PNG file is passed directly to whiteboard-animate."""
        self.create_test_setup(1, formats=['PNG'])

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 1)
        # Verify the original PNG path was passed
        call_args = mock_animate.call_args
        passed_image_path = call_args[0][0]
        self.assertEqual(passed_image_path.name, "scene 1.png")
        self.assertEqual(passed_image_path, self.assets_dir / "scene 1.png")

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_jpeg_passed_directly_to_whiteboard(self, mock_animate):
        """Test that real JPEG file is passed directly to whiteboard-animate (no conversion)."""
        self.create_test_setup(1, formats=['JPEG'])

        def mock_animate_side_effect(image_path, output_path, duration):
            # Verify the passed image is the original JPEG file
            self.assertEqual(image_path, self.assets_dir / "scene 1.png")
            # Verify the file is indeed JPEG
            from PIL import Image
            with Image.open(image_path) as img:
                self.assertEqual(img.format, 'JPEG')
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 1)
        # Verify the original JPEG path was passed
        call_args = mock_animate.call_args
        passed_image_path = call_args[0][0]
        self.assertEqual(passed_image_path.name, "scene 1.png")
        self.assertEqual(passed_image_path, self.assets_dir / "scene 1.png")

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_original_file_not_modified_during_process(self, mock_animate):
        """Test that original file (PNG or JPEG) is not modified during processing."""
        self.create_test_setup(1, formats=['JPEG'])

        original_size = (self.assets_dir / "scene 1.png").stat().st_size
        original_mtime = (self.assets_dir / "scene 1.png").stat().st_mtime

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        # Original file unchanged
        self.assertEqual((self.assets_dir / "scene 1.png").stat().st_size, original_size)
        self.assertEqual((self.assets_dir / "scene 1.png").stat().st_mtime, original_mtime)
        # Original is still JPEG
        from PIL import Image
        with Image.open(self.assets_dir / "scene 1.png") as img:
            self.assertEqual(img.format, 'JPEG')

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_no_temp_files_created(self, mock_animate):
        """Test that no temporary files are created during processing."""
        self.create_test_setup(2, formats=['PNG', 'JPEG'])

        # Clean up any pre-existing tmp files before test
        import tempfile
        sys_temp = Path(tempfile.gettempdir())
        pre_existing = set(sys_temp.glob('tmp*.png'))

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        # No new temp files should remain
        post_files = set(sys_temp.glob('tmp*.png'))
        new_files = post_files - pre_existing
        for f in new_files:
            self.fail(f"Temporary file not cleaned up: {f}")

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_mixed_formats_multiple_scenes(self, mock_animate):
        """Test multiple scenes with mixed PNG/JPEG formats (all passed directly)."""
        self.create_test_setup(3, formats=['PNG', 'JPEG', 'PNG'])

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(len(output_paths), 3)
        calls = mock_animate.call_args_list
        # Scene 1: PNG passed directly
        self.assertEqual(calls[0][0][0].name, "scene 1.png")
        # Scene 2: JPEG passed directly (no conversion)
        self.assertEqual(calls[1][0][0].name, "scene 2.png")
        self.assertEqual(calls[1][0][0], self.assets_dir / "scene 2.png")
        # Scene 3: PNG passed directly
        self.assertEqual(calls[2][0][0].name, "scene 3.png")

    @patch("whiteboard_integration.run_whiteboard_animate")
    def test_output_mp4_names_unchanged(self, mock_animate):
        """Test that output MP4 naming is unchanged regardless of input format."""
        self.create_test_setup(2, formats=['JPEG', 'PNG'])

        def mock_animate_side_effect(image_path, output_path, duration):
            output_path.write_bytes(b"fake mp4 content")
        mock_animate.side_effect = mock_animate_side_effect

        output_paths = process_run_directory(self.run_dir)

        self.assertEqual(output_paths[0].name, "scene 1.mp4")
        self.assertEqual(output_paths[1].name, "scene 2.mp4")
        self.assertEqual(output_paths[0].parent.name, "whiteboard")
        self.assertEqual(output_paths[1].parent.name, "whiteboard")


if __name__ == "__main__":
    unittest.main(verbosity=2)