#!/usr/bin/env python3
"""
run_pipeline2.py — Pipeline 2 Controller

Single responsibility:
    Orchestrate Whiteboard Animator integration for a Pipeline 1 run directory.

Usage:
    python run_pipeline2.py <pipeline1_run_directory>

The controller:
1. Validates the supplied run directory exists
2. Delegates to whiteboard_integration.process_run_directory()
3. Propagates failures clearly
4. Exits successfully when Whiteboard Animator processing succeeds

Does NOT:
- Modify whiteboard_integration.py
- Implement Remotion, transitions, narration, audio, captions, or final video composition
- Assume a fixed number of scenes
"""

import sys
from pathlib import Path

# Import the integration module
from whiteboard_integration import process_run_directory, WhiteboardIntegrationError


def main(args: list[str] = None) -> int:
    """
    Main entry point for Pipeline 2 controller.

    Args:
        args: Command line arguments (excluding script name). Defaults to sys.argv[1:].

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

    # Delegate to whiteboard integration
    try:
        output_paths = process_run_directory(run_dir)
        print(f"Pipeline 2 completed successfully. Generated {len(output_paths)} scene MP4(s):")
        for path in output_paths:
            print(f"  {path}")
        return 0
    except WhiteboardIntegrationError as e:
        print(f"ERROR: Whiteboard integration failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())