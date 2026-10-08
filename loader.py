#!/usr/bin/env python3
"""
Loader for dynamic history video project.
Discovers and loads project configuration, validates data,
and generates a normalized manifest.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        raise ValueError(f"Could not read {filepath}: {e}")


def validate_project_config(project_data: Dict[str, Any]) -> None:
    """Validate project configuration."""
    required_top = ["project_id", "title", "description", "output"]
    for field in required_top:
        if field not in project_data:
            raise ValueError(f"Missing required project field: '{field}'")

    output = project_data.get("output", {})
    required_output = ["directory", "filename_pattern", "format", "resolution", "fps"]
    for field in required_output:
        if field not in output:
            raise ValueError(f"Missing required output field: '{field}'")

    resolution = output.get("resolution", {})
    if "width" not in resolution or "height" not in resolution:
        raise ValueError("Output resolution must include 'width' and 'height'")
    if not isinstance(resolution["width"], int) or resolution["width"] <= 0:
        raise ValueError("Output resolution width must be a positive integer")
    if not isinstance(resolution["height"], int) or resolution["height"] <= 0:
        raise ValueError("Output resolution height must be a positive integer")
    if not isinstance(output["fps"], int) or output["fps"] <= 0:
        raise ValueError("Output fps must be a positive integer")


def validate_scene_config(scene_data: Dict[str, Any], scene_dir: Path) -> None:
    """Validate scene configuration."""
    required_scene = ["scene_id", "title", "description", "image", "order"]
    for field in required_scene:
        if field not in scene_data:
            raise ValueError(f"Scene missing required field: '{field}'")

    image = scene_data.get("image", {})
    if "filename" not in image:
        raise ValueError("Scene image missing 'filename'")

    # Check that image file exists
    image_path = scene_dir / image["filename"]
    if not image_path.is_file():
        raise ValueError(f"Image file not found: {image_path}")

    # Ensure order is integer
    if not isinstance(scene_data["order"], int):
        raise ValueError("Scene 'order' must be an integer")


def discover_scenes(project_root: Path) -> List[Path]:
    """Discover scene directories under scenes/."""
    scenes_dir = project_root / "scenes"
    if not scenes_dir.is_dir():
        raise ValueError(f"Scenes directory not found: {scenes_dir}")

    scene_dirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {scenes_dir}")

    return scene_dirs


def load_project(project_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and validate project and all scenes."""
    project_file = project_root / "project.json"
    if not project_file.is_file():
        raise ValueError(f"Project configuration not found: {project_file}")

    project_data = load_json_file(project_file)
    validate_project_config(project_data)

    scene_dirs = discover_scenes(project_root)
    scenes_data = []

    for scene_dir in scene_dirs:
        scene_file = scene_dir / "scene.json"
        if not scene_file.is_file():
            raise ValueError(f"Missing scene.json in {scene_dir}")

        scene_data = load_json_file(scene_file)
        validate_scene_config(scene_data, scene_dir)

        # Build normalized scene entry for manifest
        image_rel = scene_data["image"]["filename"]
        scene_entry = {
            "scene_id": scene_data["scene_id"],
            "title": scene_data["title"],
            "description": scene_data["description"],
            "image_path": str((scene_dir / image_rel).resolve()),
            "order": scene_data["order"],
            # Placeholder for future fields (can be added later without breaking loader)
            "duration": scene_data.get("duration"),
            "narration": scene_data.get("narration"),
            "captions": scene_data.get("captions"),
            "transition": scene_data.get("transition"),
            "whiteboard_animator": scene_data.get("whiteboard_animator"),
            "remotion": scene_data.get("remotion"),
        }
        scenes_data.append(scene_entry)

    # Sort scenes by order
    scenes_data.sort(key=lambda s: s["order"])

    # Ensure orders are unique and sequential starting from 1? Not required, but we can check for duplicates.
    orders = [s["order"] for s in scenes_data]
    if len(set(orders)) != len(orders):
        raise ValueError("Scene orders must be unique")

    return project_data, scenes_data


def generate_manifest(project_root: Path) -> Dict[str, Any]:
    """Generate normalized manifest."""
    project_data, scenes_data = load_project(project_root)

    manifest = {
        "project": project_data,
        "scenes": scenes_data
    }

    return manifest


def main() -> None:
    """Main entry point."""
    # Assume we are run from the project root
    project_root = Path.cwd()

    try:
        manifest = generate_manifest(project_root)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    manifest_file = project_root / "manifest.json"
    try:
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest generated successfully: {manifest_file}")
    except Exception as e:
        print(f"Error writing manifest: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()