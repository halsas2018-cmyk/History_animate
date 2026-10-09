#!/usr/bin/env python3
"""
run_pipeline2.py — Pipeline 2 Controller

Single responsibility:
    Orchestrate Whiteboard Animator integration and Remotion asset preparation for a Pipeline 1 run directory.

Usage:
    python run_pipeline2.py <pipeline1_run_directory>

The controller:
1. Validates the supplied run directory exists
2. Delegates to whiteboard_integration.process_run_directory()
3. Copies generated scene MP4s, narration.mp3, and visual_timing.json to my-video/public/
4. Propagates failures clearly
5. Exits successfully when Whiteboard Animator processing and asset preparation succeed
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Import the integration module
from whiteboard_integration import process_run_directory, WhiteboardIntegrationError


def files_are_identical(source: Path, dest: Path) -> bool:
    """Check if two files have identical size and SHA-256 hash."""
    if not source.exists() or not dest.exists():
        return False
    if source.stat().st_size != dest.stat().st_size:
        return False
    return hashlib.sha256(source.read_bytes()).hexdigest() == hashlib.sha256(dest.read_bytes()).hexdigest()


def copy_assets_to_public(run_dir: Path, output_paths: List[Path], public_dir: Optional[Path] = None) -> List[Path]:
    """
    Copy generated scene MP4s, narration.mp3, and visual_timing.json to Remotion public directory.

    Args:
        run_dir: Pipeline run directory containing visual_timing.json, narration.mp3, etc.
        output_paths: List of generated scene MP4 paths from process_run_directory.
        public_dir: Target Remotion public directory. Defaults to <project_dir>/my-video/public.

    Returns:
        List of paths to files prepared in the public directory.
    """
    if public_dir is None:
        public_dir = Path(__file__).resolve().parent / "my-video" / "public"

    public_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []

    # 1. Copy generated scene MP4s
    for scene_path in output_paths:
        dest_path = public_dir / scene_path.name
        if not files_are_identical(scene_path, dest_path):
            shutil.copy2(scene_path, dest_path)
            print(f"  Copied {scene_path.name} -> {dest_path}")
        else:
            print(f"  Skipped identical {scene_path.name}")
        copied_files.append(dest_path)

    # 2. Copy narration.mp3
    narration_path = run_dir / "narration.mp3"
    if narration_path.exists():
        dest_narration = public_dir / "narration.mp3"
        if not files_are_identical(narration_path, dest_narration):
            shutil.copy2(narration_path, dest_narration)
            print(f"  Copied narration.mp3 -> {dest_narration}")
        else:
            print(f"  Skipped identical narration.mp3")
        copied_files.append(dest_narration)

    # 3. Copy visual_timing.json
    timing_path = run_dir / "visual_timing.json"
    if timing_path.exists():
        dest_timing = public_dir / "visual_timing.json"
        if not files_are_identical(timing_path, dest_timing):
            shutil.copy2(timing_path, dest_timing)
            print(f"  Copied visual_timing.json -> {dest_timing}")
        else:
            print(f"  Skipped identical visual_timing.json")
        copied_files.append(dest_timing)

    # 4. Copy whisperx_timing.json (word-level timestamps for captions)
    whisperx_path = run_dir / "whisperx_timing.json"
    if whisperx_path.exists():
        dest_whisperx = public_dir / "whisperx_timing.json"
        if not files_are_identical(whisperx_path, dest_whisperx):
            shutil.copy2(whisperx_path, dest_whisperx)
            print(f"  Copied whisperx_timing.json -> {dest_whisperx}")
        else:
            print(f"  Skipped identical whisperx_timing.json")
        copied_files.append(dest_whisperx)

    return copied_files


def main(args: list[str] = None, public_dir: Optional[Path] = None) -> int:
    """
    Main entry point for Pipeline 2 controller.

    Args:
        args: Command line arguments (excluding script name). Defaults to sys.argv[1:].
        public_dir: Optional target public directory for testing/override.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if args is None:
        args = sys.argv[1:]

    # Validate CLI arguments
    if len(args) != 1:
        print("Usage: python run_pipeline2.py <pipeline1_run_directory>", file=sys.stderr)
        return 1

    run_dir_arg = args[0]
    run_dir = Path(run_dir_arg)

    # Validate run directory exists
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    if not run_dir.is_dir():
        print(f"ERROR: Run path is not a directory: {run_dir}", file=sys.stderr)
        return 1

    # Delegate to whiteboard integration and copy assets
    try:
        output_paths = process_run_directory(run_dir)
        print(f"Pipeline 2 animation completed successfully. Generated {len(output_paths)} scene MP4(s):")
        for path in output_paths:
            print(f"  {path}")

        print("\nCopying assets to Remotion public directory...")
        copied = copy_assets_to_public(run_dir, output_paths, public_dir=public_dir)
        print(f"Asset preparation complete: {len(copied)} file(s) ready in public directory.")
        return 0
    except WhiteboardIntegrationError as e:
        print(f"ERROR: Whiteboard integration failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())