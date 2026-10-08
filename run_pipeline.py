#!/usr/bin/env python3
"""
run_pipeline.py — Central dynamic pipeline controller for history-video generation.

Architecture:
- Thin orchestration layer that chains discrete, reusable pipeline stages.
- Each stage is a pure function with a single responsibility.
- Stages communicate via explicit JSON-serializable artifacts (dicts).
- New stages are appended to the pipeline list; no rewriting of this controller needed.

Environment:
- Uses the virtual environment at /root/kinetic_typo_vid/venv/
- Activate with: source /root/kinetic_typo_vid/venv/bin/activate
- Or run directly with: /root/kinetic_typo_vid/venv/bin/python3 run_pipeline.py
"""

from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import history_story_generator
import visual_planning
import visual_timing
import asset_preparation
import voice_generation
import whisperx_timing
import sentence_timing
from llm_client import load_env_file


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a pipeline run."""
    topic: str
    model_key: str
    length_mode: str
    output_dir: Path
    save_intermediate: bool = True

    @staticmethod
    def from_args(args: argparse.Namespace) -> "PipelineConfig":
        return PipelineConfig(
            topic=args.topic,
            model_key=args.model or "gemini-31-flash-lite",
            length_mode=args.length or "short",
            output_dir=Path(args.output).resolve() if args.output else Path.cwd() / "output" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_intermediate=not args.no_save,
        )


@dataclass
class PipelineArtifact:
    """Container for stage outputs — the currency of the pipeline."""
    stage_name: str
    data: dict
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"stage": self.stage_name, "data": self.data, "meta": self.metadata}, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Stage Protocol
# ═══════════════════════════════════════════════════════════════════════════

StageFn = Callable[[PipelineConfig, dict], PipelineArtifact]
"""Signature: (config, previous_stage_output) -> PipelineArtifact"""


def stage_story_generation(config: PipelineConfig, _prev: dict) -> PipelineArtifact:
    """Stage 1: Generate the historical story from the topic."""
    story = history_story_generator.generate_history_story(
        topic=config.topic,
        model_key=config.model_key,
        length_mode=config.length_mode,
    )
    return PipelineArtifact(
        stage_name="story_generation",
        data=story,
        metadata={"model_key": config.model_key, "length_mode": config.length_mode},
    )


def stage_voice_generation(config: PipelineConfig, prev: dict) -> PipelineArtifact:
    """Stage 2: Generate voice narration from the story."""
    # prev is the story artifact from story_generation
    result = voice_generation.generate_voice(
        story=prev,
        output_dir=config.output_dir,
    )
    return PipelineArtifact(
        stage_name="voice_generation",
        data=result,
        metadata={"voice": result["voice"], "rate": result["rate"]},
    )


def stage_visual_planning(config: PipelineConfig, prev: dict) -> PipelineArtifact:
    """Stage 3: Plan visuals from the story."""
    # prev is the story artifact (passed by PipelineRunner)
    plan = visual_planning.plan_visuals(
        story=prev,
        model_key=config.model_key,
    )
    return PipelineArtifact(
        stage_name="visual_planning",
        data=plan,
        metadata={"model_key": config.model_key},
    )


# Pipeline stages in execution order. Append future stages here.
PIPELINE_STAGES: list[StageFn] = [
    stage_story_generation,
    stage_voice_generation,
    whisperx_timing.stage_whisperx_timing,
    sentence_timing.stage_sentence_timing,
    stage_visual_planning,
    visual_timing.stage_visual_timing,
    asset_preparation.stage_asset_preparation,
    # Future: stage_image_generation, stage_whiteboard, stage_remotion, ...
]


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """Executes the pipeline stages sequentially, passing artifacts forward."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.artifacts: list[PipelineArtifact] = []

    def run(self) -> list[PipelineArtifact]:
        """Execute all stages. Returns list of artifacts from each stage."""
        prev_output: dict = {}
        story_artifact: dict = {}

        for i, stage_fn in enumerate(PIPELINE_STAGES):
            stage_name = stage_fn.__name__.replace("stage_", "")
            print(f"[{i + 1}/{len(PIPELINE_STAGES)}] Running stage: {stage_name}...")

            try:
                # Stage 1 (story_generation) receives empty prev_output.
                # All subsequent stages receive the story artifact as input.
                if i == 0:
                    artifact = stage_fn(self.config, prev_output)
                    story_artifact = artifact.data
                else:
                    artifact = stage_fn(self.config, story_artifact)

                self.artifacts.append(artifact)

                # Keep prev_output for potential future chaining needs
                prev_output = artifact.data

                if self.config.save_intermediate:
                    self._save_artifact(artifact)

                print(f"  ✓ {stage_name} completed")

            except Exception as e:
                # Failure propagation: wrap and re-raise with stage context
                raise PipelineError(f"Stage '{stage_name}' failed: {e}") from e

        return self.artifacts

    def _save_artifact(self, artifact: PipelineArtifact) -> None:
        """Persist artifact to the output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{artifact.stage_name}.json"
        path = self.config.output_dir / filename
        path.write_text(artifact.to_json(), encoding="utf-8")
        print(f"  → Saved artifact: {path}")

    def get_final_output(self) -> Optional[dict]:
        """Return the data from the last successful stage."""
        return self.artifacts[-1].data if self.artifacts else None


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the dynamic history-video pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py "The Rise of the Mali Empire"
  python run_pipeline.py "The Fall of Rome" --model groq-gpt-oss-20b --length long
  python run_pipeline.py "The Silk Road" --output ./my_runs --no-save
        """,
    )
    p.add_argument("topic", help="Historical topic to generate a story about")
    p.add_argument("--model", help="Model key from llm_client.MODEL_REGISTRY (default: gemini-31-flash-lite)")
    p.add_argument("--length", choices=("short", "long"), default="short", help="Story length mode")
    p.add_argument("--output", help="Output directory for artifacts")
    p.add_argument("--no-save", action="store_true", help="Do not save intermediate artifacts to disk")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load .env file explicitly (must happen before any API key access)
    load_env_file()

    # Validate topic early
    if not args.topic or not args.topic.strip():
        parser.error("topic is required and must be non-empty")

    config = PipelineConfig.from_args(args)

    try:
        runner = PipelineRunner(config)
        artifacts = runner.run()

        final_output = runner.get_final_output()
        if final_output:
            print("\n=== Pipeline completed successfully ===")
            print(f"Artifacts generated: {len(artifacts)}")
            print(f"Final output keys: {list(final_output.keys())}")
            if config.save_intermediate:
                print(f"Output directory: {config.output_dir}")
        else:
            print("Pipeline completed but no output produced.")

        return 0

    except PipelineError as e:
        print(f"\n✗ Pipeline failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())