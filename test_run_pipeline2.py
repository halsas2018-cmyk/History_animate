#!/usr/bin/env python3
"""
test_run_pipeline2.py — Unit tests for Pipeline 2 Controller

Tests cover:
- missing run directory
- valid run directory
- successful delegation to whiteboard_integration
- integration failure propagation
- CLI argument handling
- successful CLI exit
- invalid CLI usage
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from whiteboard_integration import WhiteboardIntegrationError
import run_pipeline2


def create_valid_png(path: Path) -> None:
    """Create a minimal valid PNG file at the given path."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='white')
    img.save(path)


class TestRunPipeline2(unittest.TestCase):
    """Tests for run_pipeline2 main function."""

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

    def create_valid_run_dir(self, num_visuals: int = 3):
        """Create a valid run directory structure."""
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

    @patch("run_pipeline2.process_run_directory")
    def test_valid_run_directory_delegates_successfully(self, mock_process):
        """Test that valid run directory delegates to whiteboard_integration."""
        self.create_valid_run_dir(3)

        mock_process.return_value = [
            self.run_dir / "whiteboard" / "scene 1.mp4",
            self.run_dir / "whiteboard" / "scene 2.mp4",
            self.run_dir / "whiteboard" / "scene 3.mp4",
        ]

        result = run_pipeline2.main([str(self.run_dir)])

        self.assertEqual(result, 0)
        mock_process.assert_called_once_with(self.run_dir)

    @patch("run_pipeline2.process_run_directory")
    def test_successful_delegation_prints_output_paths(self, mock_process):
        """Test that successful delegation prints output paths."""
        self.create_valid_run_dir(2)

        output_paths = [
            self.run_dir / "whiteboard" / "scene 1.mp4",
            self.run_dir / "whiteboard" / "scene 2.mp4",
        ]
        mock_process.return_value = output_paths

        # Capture stdout
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            result = run_pipeline2.main([str(self.run_dir)])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(result, 0)
        output = captured.getvalue()
        self.assertIn("Pipeline 2 completed successfully", output)
        self.assertIn("scene 1.mp4", output)
        self.assertIn("scene 2.mp4", output)

    def test_missing_run_directory(self):
        """Test error when run directory doesn't exist."""
        missing_dir = self.run_dir / "nonexistent"

        result = run_pipeline2.main([str(missing_dir)])

        self.assertEqual(result, 1)
        # Check stderr was printed (we can't easily capture it in this test setup)

    def test_run_path_is_not_a_directory(self):
        """Test error when run path is a file, not a directory."""
        # Create a file instead of directory
        file_path = self.run_dir / "not_a_dir.txt"
        file_path.write_text("not a directory")

        result = run_pipeline2.main([str(file_path)])

        self.assertEqual(result, 1)

    @patch("run_pipeline2.process_run_directory")
    def test_integration_failure_propagation(self, mock_process):
        """Test that WhiteboardIntegrationError is propagated."""
        self.create_valid_run_dir(1)

        mock_process.side_effect = WhiteboardIntegrationError("Missing image file: scene 1.png")

        # Capture stderr
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = captured = StringIO()
        try:
            result = run_pipeline2.main([str(self.run_dir)])
        finally:
            sys.stderr = old_stderr

        self.assertEqual(result, 1)
        output = captured.getvalue()
        self.assertIn("ERROR: Whiteboard integration failed", output)
        self.assertIn("Missing image file", output)

    @patch("run_pipeline2.process_run_directory")
    def test_unexpected_error_propagation(self, mock_process):
        """Test that unexpected exceptions are caught and reported."""
        self.create_valid_run_dir(1)

        mock_process.side_effect = RuntimeError("Unexpected failure")

        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = captured = StringIO()
        try:
            result = run_pipeline2.main([str(self.run_dir)])
        finally:
            sys.stderr = old_stderr

        self.assertEqual(result, 1)
        output = captured.getvalue()
        self.assertIn("UNEXPECTED ERROR", output)
        self.assertIn("Unexpected failure", output)

    def test_cli_argument_handling_no_args(self):
        """Test CLI with no arguments shows usage."""
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = captured = StringIO()
        try:
            result = run_pipeline2.main([])
        finally:
            sys.stderr = old_stderr

        self.assertEqual(result, 1)
        output = captured.getvalue()
        self.assertIn("Usage: python run_pipeline2.py <pipeline1_run_directory>", output)

    def test_cli_argument_handling_too_many_args(self):
        """Test CLI with too many arguments shows usage."""
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = captured = StringIO()
        try:
            result = run_pipeline2.main(["arg1", "arg2"])
        finally:
            sys.stderr = old_stderr

        self.assertEqual(result, 1)
        output = captured.getvalue()
        self.assertIn("Usage: python run_pipeline2.py <pipeline1_run_directory>", output)


class TestRunPipeline2CLI(unittest.TestCase):
    """Test the CLI entry point via subprocess."""

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

    def create_valid_run_dir(self, num_visuals: int = 2):
        """Create a valid run directory structure."""
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

    @patch("run_pipeline2.process_run_directory")
    def test_cli_success(self, mock_process):
        """Test CLI runs successfully with mocked integration."""
        self.create_valid_run_dir(2)

        def mock_process_side_effect(run_dir):
            # Create the output files
            whiteboard_dir = run_dir / "whiteboard"
            whiteboard_dir.mkdir()
            (whiteboard_dir / "scene 1.mp4").write_bytes(b"fake mp4")
            (whiteboard_dir / "scene 2.mp4").write_bytes(b"fake mp4")
            return [
                whiteboard_dir / "scene 1.mp4",
                whiteboard_dir / "scene 2.mp4",
            ]
        mock_process.side_effect = mock_process_side_effect

        import subprocess
        result = subprocess.run(
            [sys.executable, "run_pipeline2.py", str(self.run_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Pipeline 2 completed successfully", result.stdout)
        self.assertIn("scene 1.mp4", result.stdout)
        self.assertIn("scene 2.mp4", result.stdout)

    def test_cli_invalid_usage(self):
        """Test CLI with invalid usage returns error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_pipeline2.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_cli_missing_directory(self):
        """Test CLI with missing directory returns error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_pipeline2.py", "/nonexistent/path"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERROR: Run directory not found", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)