#!/usr/bin/env python3
"""
asset_preparation.py — VISUAL_TIMING → asset directory with placeholder PNGs

Single responsibility:
    prepare_assets(visual_timing: dict, output_dir: Path) -> dict

Creates an assets/ directory with valid PNG placeholder files for each visual.

Rules:
- Number of placeholders determined dynamically from visual_timing.json
- Filename convention: "scene 1.png", "scene 2.png", ...
- Each placeholder is a valid PNG (1x1 white pixel)
- Existing valid PNGs are NOT overwritten
- No network/API calls, completely local and deterministic
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import struct
import zlib


def create_white_png(path: Path, width: int = 1, height: int = 1) -> None:
    """
    Create a minimal valid white PNG file at the given path.

    Uses a 1x1 white pixel PNG (67 bytes) - the smallest valid PNG.
    """
    # 1x1 white pixel PNG (pre-computed)
    # PNG signature + IHDR + IDAT + IEND
    png_data = (
        b'\x89PNG\r\n\x1a\n'  # PNG signature
        b'\x00\x00\x00\rIHDR'  # IHDR chunk length + type
        b'\x00\x00\x00\x01'    # width = 1
        b'\x00\x00\x00\x01'    # height = 1
        b'\x08\x02\x00\x00\x00'  # bit depth=8, color type=2 (RGB), compression=0, filter=0, interlace=0
        b'\x90wS\xde'          # CRC32 of IHDR
        b'\x00\x00\x00\x0cIDAT'  # IDAT chunk length + type
        b'\x08\xd7c\xf8\x0f\x00\x01\x01\x01\x00\x00'  # Compressed data (1x1 white pixel)
        b'\x00\x1c\x00\xa4'    # CRC32 of IDAT
        b'\x00\x00\x00\x00IEND'  # IEND chunk
        b'\xaeB`\x82'          # CRC32 of IEND
    )
    path.write_bytes(png_data)


def is_valid_png(path: Path) -> bool:
    """Check if a file is a valid PNG by reading its signature."""
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
        return header == b'\x89PNG\r\n\x1a\n'
    except Exception:
        return False


def prepare_assets(visual_timing: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Create assets directory with placeholder PNGs for each visual.

    Parameters
    ----------
    visual_timing : dict
        Output from visual_timing stage: {"visuals": [{"visual_index": int, "image": str, ...}, ...]}
    output_dir : Path
        Pipeline run directory (e.g., output/run_20261007_092425/)

    Returns
    -------
    dict
        {
            "assets_dir": str,           # relative path to assets directory
            "visual_count": int,         # number of visuals
            "scene_files": list[str]     # ordered list of scene filenames
        }
    """
    # ---- Input validation ----------------------------------------------------
    if not isinstance(visual_timing, dict):
        raise TypeError(f"visual_timing must be a dict, got {type(visual_timing).__name__}")
    if "visuals" not in visual_timing:
        raise ValueError("visual_timing missing required key: 'visuals'")

    visuals = visual_timing["visuals"]
    if not isinstance(visuals, list):
        raise TypeError(f"visual_timing['visuals'] must be a list, got {type(visuals).__name__}")

    # ---- Create assets directory ---------------------------------------------
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # ---- Determine expected scene files from visual_timing -------------------
    visual_count = len(visuals)
    scene_files = [f"scene {i + 1}.png" for i in range(visual_count)]

    # ---- Create placeholders (only if missing or invalid) --------------------
    created = []
    skipped = []
    for scene_file in scene_files:
        scene_path = assets_dir / scene_file
        if scene_path.exists():
            if is_valid_png(scene_path):
                skipped.append(scene_file)
                continue
            else:
                # File exists but is not a valid PNG - overwrite with placeholder
                create_white_png(scene_path)
                created.append(scene_file)
        else:
            create_white_png(scene_path)
            created.append(scene_file)

    # ---- Build artifact -------------------------------------------------------
    return {
        "assets_dir": str(assets_dir.relative_to(output_dir)),
        "visual_count": visual_count,
        "scene_files": scene_files,
        "created": created,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Pipeline stage function
# ---------------------------------------------------------------------------
def stage_asset_preparation(config: Any, prev: dict) -> Any:
    """
    Pipeline stage: Asset Preparation.

    Loads visual_timing artifact from disk.
    Creates assets/ directory with placeholder PNGs.
    Returns a PipelineArtifact with asset preparation info.
    """
    try:
        from run_pipeline import PipelineArtifact
    except ImportError:
        class PipelineArtifact:
            def __init__(self, stage_name: str, data: dict, metadata: dict = None):
                self.stage_name = stage_name
                self.data = data
                self.metadata = metadata or {}
            def to_json(self) -> str:
                import json
                return json.dumps(
                    {"stage": self.stage_name, "data": self.data, "meta": self.metadata},
                    indent=2,
                    ensure_ascii=False,
                )

    # Load visual_timing artifact from disk
    timing_path = config.output_dir / "visual_timing.json"
    if not timing_path.exists():
        raise FileNotFoundError(
            f"Visual timing artifact not found: {timing_path}. "
            "Ensure the visual_timing stage has run and saved intermediate artifacts."
        )
    try:
        with open(timing_path, "r", encoding="utf-8") as f:
            timing_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse visual_timing artifact: {e}") from e

    # Unwrap pipeline artifact envelope if present
    if isinstance(timing_data, dict) and "data" in timing_data and "stage" in timing_data:
        timing_data = timing_data["data"]

    # Extract visual_timing
    if not isinstance(timing_data, dict) or "visuals" not in timing_data:
        raise ValueError("Visual timing artifact missing 'visuals' field")

    # Prepare assets
    assets_info = prepare_assets(timing_data, config.output_dir)

    # Build artifact
    artifact = PipelineArtifact(
        stage_name="asset_preparation",
        data=assets_info,
        metadata={
            "visual_count": assets_info["visual_count"],
            "created_count": len(assets_info["created"]),
            "skipped_count": len(assets_info["skipped"]),
        },
    )
    return artifact


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python asset_preparation.py <visual_timing_json_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    timing_path = sys.argv[1]

    with open(timing_path, "r", encoding="utf-8") as f:
        timing_data = json.load(f)

    # Unwrap envelope if present
    if isinstance(timing_data, dict) and "data" in timing_data and "stage" in timing_data:
        timing_data = timing_data["data"]

    # Dummy config with output_dir
    class DummyConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    config = DummyConfig(Path(timing_path).parent)

    try:
        from run_pipeline import PipelineArtifact
    except ImportError:
        class PipelineArtifact:
            def __init__(self, stage_name: str, data: dict, metadata: dict = None):
                self.stage_name = stage_name
                self.data = data
                self.metadata = metadata or {}
            def to_json(self) -> str:
                import json
                return json.dumps(
                    {"stage": self.stage_name, "data": self.data, "meta": self.metadata},
                    indent=2,
                    ensure_ascii=False,
                )

    artifact = stage_asset_preparation(config, timing_data)
    print(json.dumps(artifact.data, indent=2, ensure_ascii=False))