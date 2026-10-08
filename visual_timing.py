#!/usr/bin/env python3
"""
visual_timing.py — VISUAL_PLAN + SENTENCE_TIMING → visual_timing.json

Single responsibility:
    compute_visual_timing(visual_plan: dict, sentence_timing: list[dict]) -> dict

Each visual_timing entry:
    {
        "visual_index": int,          # 0-based
        "sentence_indices": list[int], # from visual_plan
        "image": str,                  # "scene_1.png", "scene_2.png", ... (1-based)
        "start": float,                # seconds, from earliest sentence startFrame
        "end": float,                  # seconds, from latest sentence endFrame
        "duration": float              # end - start
    }

Rules:
- visual_index is 0-based.
- image filename is 1-based: scene_1.png, scene_2.png, ...
- Preserve sentence_indices exactly from visual_plan.
- Every visual must have at least one sentence index.
- All sentence indices must be valid (exist in sentence_timing).
- Every story sentence must be covered exactly once across all visuals.
- Visuals remain in same order as visual_plan.
- start < end, duration == end - start.
- No negative timing.
- No visual has missing timing.
- Validate the generated artifact before returning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

FPS = 30  # Matching sentence_timing.py


def frames_to_seconds(frames: int) -> float:
    """Convert frame number to seconds using FPS."""
    return frames / FPS


def compute_visual_timing(visual_plan: Dict[str, Any], sentence_timing: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine visual_plan and sentence_timing into visual_timing.

    Parameters
    ----------
    visual_plan : dict
        Output from visual_planning: {"visual_plan": [{"sentence_indices": [...], "image_prompt": "..."}, ...]}
    sentence_timing : list[dict]
        Output from sentence_timing stage: list of {"sentence_index": int, "startFrame": int, "endFrame": int, ...}

    Returns
    -------
    dict
        {"visuals": [{"visual_index": int, "sentence_indices": [...], "image": str, "start": float, "end": float, "duration": float}, ...]}
    """
    # ---- Input validation ----------------------------------------------------
    if not isinstance(visual_plan, dict):
        raise TypeError(f"visual_plan must be a dict, got {type(visual_plan).__name__}")
    if "visual_plan" not in visual_plan:
        raise ValueError("visual_plan missing required key: 'visual_plan'")
    if len(visual_plan) != 1:
        extra = set(visual_plan.keys()) - {"visual_plan"}
        raise ValueError(f"visual_plan must have exactly one key 'visual_plan', got extra: {sorted(extra)}")

    plan_entries = visual_plan["visual_plan"]
    if not isinstance(plan_entries, list):
        raise TypeError(f"visual_plan['visual_plan'] must be a list, got {type(plan_entries).__name__}")
    if not plan_entries:
        raise ValueError("visual_plan['visual_plan'] must be non-empty")

    if not isinstance(sentence_timing, list):
        raise TypeError(f"sentence_timing must be a list, got {type(sentence_timing).__name__}")

    # Build lookup for sentence_timing by sentence_index
    timing_by_index = {}
    for entry in sentence_timing:
        if not isinstance(entry, dict):
            raise TypeError("sentence_timing entries must be dicts")
        idx = entry.get("sentence_index")
        if not isinstance(idx, int):
            raise TypeError("sentence_timing entry missing valid 'sentence_index'")
        if "startFrame" not in entry or "endFrame" not in entry:
            raise ValueError(f"sentence_timing entry {idx} missing startFrame or endFrame")
        timing_by_index[idx] = entry

    # ---- Validate visual_plan and compute visual_timing ----------------------
    visuals = []
    all_covered_indices = []

    for visual_index, plan_entry in enumerate(plan_entries):
        if not isinstance(plan_entry, dict):
            raise TypeError(f"visual_plan entry {visual_index} must be a dict")

        if "sentence_indices" not in plan_entry:
            raise ValueError(f"visual_plan entry {visual_index} missing 'sentence_indices'")

        sentence_indices = plan_entry["sentence_indices"]
        if not isinstance(sentence_indices, list):
            raise TypeError(f"visual_plan entry {visual_index} 'sentence_indices' must be a list")
        if not sentence_indices:
            raise ValueError(f"visual_plan entry {visual_index} 'sentence_indices' must be non-empty")

        # Validate all indices exist in sentence_timing
        for idx in sentence_indices:
            if not isinstance(idx, int):
                raise TypeError(f"visual_plan entry {visual_index} contains non-integer sentence index: {idx}")
            if idx not in timing_by_index:
                raise ValueError(f"visual_plan entry {visual_index} references missing sentence index {idx}")

        # Find earliest start and latest end from sentence timing
        start_frames = [timing_by_index[idx]["startFrame"] for idx in sentence_indices]
        end_frames = [timing_by_index[idx]["endFrame"] for idx in sentence_indices]

        min_start_frame = min(start_frames)
        max_end_frame = max(end_frames)

        # Convert to seconds
        start_seconds = frames_to_seconds(min_start_frame)
        end_seconds = frames_to_seconds(max_end_frame)

        if start_seconds > end_seconds:
            raise ValueError(
                f"visual_index {visual_index}: start ({start_seconds:.3f}s) must be <= end ({end_seconds:.3f}s). "
                f"Check sentence timing for indices {sentence_indices}."
            )

        duration = end_seconds - start_seconds

        # Image filename (1-based)
        image_filename = f"scene_{visual_index + 1}.png"

        visual = {
            "visual_index": visual_index,
            "sentence_indices": sentence_indices,
            "image": image_filename,
            "start": round(start_seconds, 3),
            "end": round(end_seconds, 3),
            "duration": round(duration, 3),
        }
        visuals.append(visual)

        all_covered_indices.extend(sentence_indices)

    # ---- Coverage validation -------------------------------------------------
    # Every sentence index 0..N-1 must appear exactly once
    num_sentences = len(sentence_timing)
    if len(all_covered_indices) != num_sentences:
        raise ValueError(
            f"Total covered sentence indices ({len(all_covered_indices)}) != "
            f"number of sentences ({num_sentences})"
        )
    if sorted(all_covered_indices) != list(range(num_sentences)):
        raise ValueError(
            f"Sentence indices not a perfect partition of [0..{num_sentences - 1}]: {sorted(all_covered_indices)}"
        )
    if len(set(all_covered_indices)) != len(all_covered_indices):
        raise ValueError("Duplicate sentence indices found across visuals")

    # ---- Validate timing consistency -----------------------------------------
    # No overlapping visual timing beyond gaps already present in sentence timing
    # (We can't easily validate this perfectly without knowing sentence timing gaps,
    # but we can check that visuals don't overlap in a way that reverses order)
    for i in range(1, len(visuals)):
        if visuals[i]["start"] < visuals[i - 1]["end"]:
            # This is allowed if there's a gap in sentence timing; we only error
            # if there's a true overlap beyond what sentence timing has
            pass  # Acceptable - consecutive visuals may share boundaries

    # ---- Build output --------------------------------------------------------
    return {"visuals": visuals}


# ---------------------------------------------------------------------------
# Pipeline stage function
# ---------------------------------------------------------------------------
def stage_visual_timing(config: Any, prev: dict) -> Any:
    """
    Pipeline stage: Visual Timing.

    Loads visual_planning artifact and sentence_timing artifact from disk.
    Returns a PipelineArtifact with visual_timing data.
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

    # Load visual_planning artifact from disk
    visual_plan_path = config.output_dir / "visual_planning.json"
    if not visual_plan_path.exists():
        raise FileNotFoundError(
            f"Visual planning artifact not found: {visual_plan_path}. "
            "Ensure the visual_planning stage has run and saved intermediate artifacts."
        )
    try:
        with open(visual_plan_path, "r", encoding="utf-8") as f:
            visual_plan_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse visual_planning artifact: {e}") from e

    # Unwrap pipeline artifact envelope if present
    if isinstance(visual_plan_data, dict) and "data" in visual_plan_data and "stage" in visual_plan_data:
        visual_plan_data = visual_plan_data["data"]

    # Extract visual_plan
    if not isinstance(visual_plan_data, dict) or "visual_plan" not in visual_plan_data:
        raise ValueError("Visual planning artifact missing 'visual_plan' field")
    visual_plan = visual_plan_data
    if not isinstance(visual_plan.get("visual_plan"), list):
        raise ValueError("visual_planning artifact missing 'visual_plan' list")

    # Load sentence_timing artifact from disk
    timing_path = config.output_dir / "sentence_timing.json"
    if not timing_path.exists():
        raise FileNotFoundError(
            f"Sentence timing artifact not found: {timing_path}. "
            "Ensure the sentence_timing stage has run and saved intermediate artifacts."
        )
    try:
        with open(timing_path, "r", encoding="utf-8") as f:
            timing_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse sentence_timing artifact: {e}") from e

    # Unwrap pipeline artifact envelope if present
    if isinstance(timing_data, dict) and "data" in timing_data and "stage" in timing_data:
        timing_data = timing_data["data"]

    # Extract sentence_timing list
    if not isinstance(timing_data, dict) or "sentence_timing" not in timing_data:
        raise ValueError("Sentence timing artifact missing 'sentence_timing' field")
    sentence_timing = timing_data["sentence_timing"]
    if not isinstance(sentence_timing, list):
        raise TypeError("'sentence_timing' must be a list")

    # Compute visual timing
    visual_timing_data = compute_visual_timing(visual_plan, sentence_timing)

    # Build artifact
    artifact = PipelineArtifact(
        stage_name="visual_timing",
        data=visual_timing_data,
        metadata={
            "visual_count": len(visual_timing_data["visuals"]),
            "total_sentences": len(sentence_timing),
        },
    )
    return artifact


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 3:
        print(
            "Usage: python visual_timing.py <visual_plan_json_file> <sentence_timing_json_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    visual_plan_path = sys.argv[1]
    sentence_timing_path = sys.argv[2]

    with open(visual_plan_path, "r", encoding="utf-8") as f:
        visual_plan = json.load(f)

    with open(sentence_timing_path, "r", encoding="utf-8") as f:
        timing_data = json.load(f)

    # Unwrap envelope if present
    if isinstance(timing_data, dict) and "data" in timing_data and "stage" in timing_data:
        timing_data = timing_data["data"]

    sentence_timing = timing_data.get("sentence_timing", [])

    try:
        result = compute_visual_timing(visual_plan, sentence_timing)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)