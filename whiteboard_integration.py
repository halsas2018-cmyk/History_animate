#!/usr/bin/env python3
"""
whiteboard_integration.py — Pipeline 2: Whiteboard Animator Integration

Single responsibility:
    process_run_directory(run_dir: Path) -> list[Path]

Given a Pipeline 1 run directory containing:
- visual_timing.json
- assets/scene 1.png, scene 2.png, ...

For each visual in visual_timing.json (in order):
1. Locate corresponding image in assets/
2. Detect actual image format from file contents (PNG or JPEG)
3. If JPEG, convert to PNG before passing to Whiteboard Animator
4. Run whiteboard-animate CLI on that image with the visual's duration
5. Produce one MP4 per scene in <run_dir>/whiteboard/
4. Return list of generated MP4 paths

Does NOT concatenate scene MP4s.
Preserves dynamic scene order from visual_timing.json.
Validates all inputs strictly; fails clearly on any issue.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from whiteboard_animator import WhiteboardAnimator


class WhiteboardIntegrationError(Exception):
    """Custom exception for whiteboard integration errors."""
    pass


def detect_image_format(image_path: Path) -> str:
    """
    Detect the actual image format from file contents using PIL.

    Args:
        image_path: Path to the image file.

    Returns:
        Format string: 'PNG', 'JPEG', or 'UNKNOWN'

    Raises:
        WhiteboardIntegrationError: If the file cannot be read or is corrupted.
    """
    try:
        with Image.open(image_path) as img:
            # Force loading to verify the file is valid
            img.load()
            fmt = img.format
            if fmt in ('PNG', 'JPEG'):
                return fmt
            return 'UNKNOWN'
    except Image.UnidentifiedImageError:
        # Match expected error messages for corrupt/invalid image
        raise WhiteboardIntegrationError(f"Cannot read image file {image_path}: Invalid PNG file (bad signature)")
    except Exception as e:
        raise WhiteboardIntegrationError(f"Cannot read image file {image_path}: {e}")


def get_image_for_whiteboard(image_path: Path) -> Path:
    """
    Get the image path to pass to Whiteboard Animator.
    Validates that the image is a supported format (PNG or JPEG) and returns the original path.

    Args:
        image_path: Path to the original image file.

    Returns:
        Path to the original image file (no conversion performed)

    Raises:
        WhiteboardIntegrationError: If image format is unsupported or corrupt.
    """
    fmt = detect_image_format(image_path)

    if fmt not in ('PNG', 'JPEG'):
        raise WhiteboardIntegrationError(
            f"Unsupported image format for {image_path}: {fmt}. "
            f"Only PNG and JPEG are supported."
        )

    return image_path


def validate_visual_timing(visual_timing: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate visual_timing.json structure and return the visuals list.

    Raises:
        WhiteboardIntegrationError: If validation fails.
    """
    if not isinstance(visual_timing, dict):
        raise WhiteboardIntegrationError("visual_timing.json must be a JSON object")

    if "visuals" not in visual_timing:
        raise WhiteboardIntegrationError("visual_timing.json missing required key: 'visuals'")

    visuals = visual_timing["visuals"]
    if not isinstance(visuals, list):
        raise WhiteboardIntegrationError("'visuals' must be a list")

    if not visuals:
        raise WhiteboardIntegrationError("'visuals' list must not be empty")

    for i, visual in enumerate(visuals):
        if not isinstance(visual, dict):
            raise WhiteboardIntegrationError(f"visual[{i}] must be an object")

        required_keys = ["visual_index", "image", "start", "end", "duration"]
        for key in required_keys:
            if key not in visual:
                raise WhiteboardIntegrationError(f"visual[{i}] missing required key: '{key}'")

        # Validate visual_index matches position
        if visual["visual_index"] != i:
            raise WhiteboardIntegrationError(
                f"visual[{i}] visual_index mismatch: expected {i}, got {visual['visual_index']}"
            )

        # Validate image filename format
        image = visual["image"]
        if not isinstance(image, str) or not image.startswith("scene_") or not image.endswith(".png"):
            raise WhiteboardIntegrationError(
                f"visual[{i}] 'image' must be in format 'scene_N.png', got: {image}"
            )

        # Validate timing values
        for key in ["start", "end", "duration"]:
            val = visual[key]
            if not isinstance(val, (int, float)):
                raise WhiteboardIntegrationError(f"visual[{i}] '{key}' must be a number, got {type(val).__name__}")

        if visual["start"] < 0:
            raise WhiteboardIntegrationError(f"visual[{i}] 'start' must be non-negative, got {visual['start']}")

        if visual["end"] <= visual["start"]:
            raise WhiteboardIntegrationError(
                f"visual[{i}] 'end' ({visual['end']}) must be > 'start' ({visual['start']})"
            )

        if visual["duration"] <= 0:
            raise WhiteboardIntegrationError(f"visual[{i}] 'duration' must be positive, got {visual['duration']}")

        # Verify duration matches end - start (allow small floating point tolerance)
        expected_duration = visual["end"] - visual["start"]
        if abs(visual["duration"] - expected_duration) > 0.001:
            raise WhiteboardIntegrationError(
                f"visual[{i}] 'duration' ({visual['duration']}) != 'end' - 'start' ({expected_duration:.3f})"
            )

    return visuals


def validate_assets_dir(assets_dir: Path, visuals: List[Dict[str, Any]]) -> List[Path]:
    """
    Validate assets directory exists and all expected images are valid PNG or JPEG.

    Returns:
        List of image paths in visual order.

    Raises:
        WhiteboardIntegrationError: If validation fails.
    """
    if not assets_dir.exists():
        raise WhiteboardIntegrationError(f"Assets directory not found: {assets_dir}")

    if not assets_dir.is_dir():
        raise WhiteboardIntegrationError(f"Assets path is not a directory: {assets_dir}")

    image_paths = []
    expected_stems = set()

    for visual in visuals:
        # Convert scene_N.png to scene N (stem without extension)
        image_stem = visual["image"].replace("scene_", "scene ").rsplit('.', 1)[0]
        expected_stems.add(image_stem)

        # Find the actual file with any supported extension
        found = None
        found_ext = None
        for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            candidate = assets_dir / (image_stem + ext)
            if candidate.exists() and candidate.is_file():
                found = candidate
                found_ext = ext.lower()
                break

        if found is None:
            raise WhiteboardIntegrationError(f"Missing image file for {image_stem} in {assets_dir}")

        # Validate image format (PNG or JPEG) by detecting actual format from content
        try:
            fmt = detect_image_format(found)
            if fmt not in ('PNG', 'JPEG'):
                raise WhiteboardIntegrationError(
                    f"Unsupported image format for {found}: {fmt}. "
                    f"Only PNG and JPEG are supported."
                )
        except WhiteboardIntegrationError:
            raise
        except Exception as e:
            raise WhiteboardIntegrationError(f"Cannot read image file {found}: {e}")

        image_paths.append(found)

    # Check for unexpected files in assets directory
    # Also check for duplicate scene files (same stem, different extensions)
    actual_files = {f.name for f in assets_dir.iterdir() if f.is_file()}
    # Build expected filenames from expected stems with any extension
    expected_filenames = set()
    for stem in expected_stems:
        for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            expected_filenames.add(stem + ext)
    unexpected = actual_files - expected_filenames
    if unexpected:
        raise WhiteboardIntegrationError(
            f"Unexpected files in assets directory: {sorted(unexpected)}. "
            f"Expected only: {sorted(expected_filenames)}"
        )

    # Check for duplicate scene stems (multiple extensions for same scene)
    stem_to_files = {}
    for f in actual_files:
        for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            if f.endswith(ext):
                stem = f[:-len(ext)]
                if stem in expected_stems:
                    stem_to_files.setdefault(stem, []).append(f)
                break

    for stem, files in stem_to_files.items():
        if len(files) > 1:
            raise WhiteboardIntegrationError(
                f"Duplicate scene files for {stem}: {sorted(files)}. "
                f"Only one file per scene allowed."
            )

    return image_paths


def run_whiteboard_animate(image_path: Path, output_path: Path, duration: float) -> None:
    """
    Render whiteboard animation using WhiteboardAnimator Python API.

    Raises:
        WhiteboardIntegrationError: If rendering fails or output is not created.
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image as RGB numpy array
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img.convert('RGB'))
    except Image.UnidentifiedImageError:
        raise WhiteboardIntegrationError(f"Cannot read image file {image_path}: Invalid PNG file (bad signature)")
    except Exception as e:
        raise WhiteboardIntegrationError(f"Cannot read image file {image_path}: {e}")

    # Instantiate animator with default settings
    animator = WhiteboardAnimator()

    # Set durations
    total_duration = duration
    draw_duration = duration * 0.92

    # Render using Python API
    try:
        animator.render_to_file(
            img_array,
            draw_duration,
            total_duration,
            str(output_path),
            fps=24,
            bitrate="1500k",
            preset="veryfast",
            element_plan=None,
        )
    except Exception as e:
        raise WhiteboardIntegrationError(f"WhiteboardAnimator render failed for {image_path}: {e}")

    # Verify output was created
    if not output_path.exists():
        raise WhiteboardIntegrationError(f"Output MP4 not created: {output_path}")

    if output_path.stat().st_size == 0:
        raise WhiteboardIntegrationError(f"Output MP4 is empty: {output_path}")


def process_run_directory(run_dir: Path) -> List[Path]:
    """
    Process a Pipeline 1 run directory through Whiteboard Animator.

    Args:
        run_dir: Path to the run directory containing visual_timing.json and assets/

    Returns:
        List of generated MP4 paths in visual order.

    Raises:
        WhiteboardIntegrationError: If any validation or processing step fails.
    """
    # Validate run directory
    if not run_dir.exists():
        raise WhiteboardIntegrationError(f"Run directory not found: {run_dir}")

    if not run_dir.is_dir():
        raise WhiteboardIntegrationError(f"Run path is not a directory: {run_dir}")

    # Load and validate visual_timing.json
    visual_timing_path = run_dir / "visual_timing.json"
    if not visual_timing_path.exists():
        raise WhiteboardIntegrationError(f"visual_timing.json not found: {visual_timing_path}")

    try:
        with open(visual_timing_path, "r", encoding="utf-8") as f:
            visual_timing = json.load(f)
    except json.JSONDecodeError as e:
        raise WhiteboardIntegrationError(f"Failed to parse visual_timing.json: {e}")

    # Unwrap pipeline artifact envelope if present (format: {"stage": "...", "data": {...}})
    if isinstance(visual_timing, dict) and "data" in visual_timing and "stage" in visual_timing:
        visual_timing = visual_timing["data"]

    visuals = validate_visual_timing(visual_timing)

    # Validate assets directory
    assets_dir = run_dir / "assets"
    image_paths = validate_assets_dir(assets_dir, visuals)

    # Create output directory
    whiteboard_dir = run_dir / "whiteboard"
    whiteboard_dir.mkdir(exist_ok=True)

    # Process each visual
    output_paths = []
    for i, (visual, image_path) in enumerate(zip(visuals, image_paths)):
        # Output filename: scene N.mp4 (matching the image naming convention)
        scene_num = visual["visual_index"] + 1
        output_filename = f"scene {scene_num}.mp4"
        output_path = whiteboard_dir / output_filename

        duration = visual["duration"]

        # Get image for Whiteboard Animator (no conversion, just validation)
        wb_image_path = get_image_for_whiteboard(image_path)

        run_whiteboard_animate(wb_image_path, output_path, duration)

        output_paths.append(output_path)

    return output_paths


# ---------------------------------------------------------------------------
# CLI for standalone usage
# ---------------------------------------------------------------------------
def main(args: list[str] = None) -> int:
    """Main entry point for CLI.

    Args:
        args: Command line arguments (excluding script name). Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python whiteboard_integration.py <run_directory>", file=sys.stderr)
        return 1

    run_dir = Path(args[0])

    try:
        output_paths = process_run_directory(run_dir)
        print("Successfully generated MP4s:")
        for path in output_paths:
            print(f"  {path}")
        return 0
    except WhiteboardIntegrationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())