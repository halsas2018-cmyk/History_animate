import {
  CalculateMetadataFunction,
  Composition,
  OffthreadVideo,
  Audio,
  staticFile,
  Sequence,
} from "remotion";
import { TransitionSeries } from "@remotion/transitions";
import { getSceneTransition, getSceneTransitionTiming } from "./SceneTransition";
import React from "react";

export interface VisualEntry {
  visual_index: number;
  sentence_indices: number[];
  image: string;
  start: number;
  end: number;
  duration: number;
}

export interface VisualTimingData {
  stage: string;
  data: {
    visuals: VisualEntry[];
  };
  meta: {
    visual_count: number;
    total_sentences: number;
  };
}

export type CompositionProps = {
  visualTimingData?: VisualTimingData;
  audioSrc?: string;
  transitionAudioSrc?: string;
  transitionDurationInFrames?: number;
};

// Default fallback timing data if visual_timing.json is unavailable
const DEFAULT_VISUAL_TIMING_DATA: VisualTimingData = {
  stage: "visual_timing",
  data: {
    visuals: [
      {
        visual_index: 0,
        sentence_indices: [0, 1],
        image: "scene_1.png",
        start: 0.2,
        end: 13.567,
        duration: 13.367,
      },
      {
        visual_index: 1,
        sentence_indices: [2, 3],
        image: "scene_2.png",
        start: 13.933,
        end: 28.3,
        duration: 14.367,
      },
      {
        visual_index: 2,
        sentence_indices: [4, 5],
        image: "scene_3.png",
        start: 28.767,
        end: 42.767,
        duration: 14.0,
      },
      {
        visual_index: 3,
        sentence_indices: [6, 7],
        image: "scene_4.png",
        start: 43.2,
        end: 53.6,
        duration: 10.4,
      },
      {
        visual_index: 4,
        sentence_indices: [8],
        image: "scene_5.png",
        start: 53.967,
        end: 59.267,
        duration: 5.3,
      },
    ],
  },
  meta: {
    visual_count: 5,
    total_sentences: 9,
  },
};

const FPS = 30;
const DEFAULT_TRANSITION_DURATION_IN_FRAMES = 12; // 0.4s @ 30fps

const calculateMetadata: CalculateMetadataFunction<CompositionProps> = async ({
  props,
}) => {
  let timingData = props.visualTimingData;

  if (!timingData) {
    try {
      const response = await fetch(staticFile("visual_timing.json"));
      if (response.ok) {
        timingData = (await response.json()) as VisualTimingData;
      }
    } catch {
      // Fallback if fetch fails
      timingData = DEFAULT_VISUAL_TIMING_DATA;
    }
  }

  if (!timingData || !timingData.data?.visuals?.length) {
    timingData = DEFAULT_VISUAL_TIMING_DATA;
  }

  const visuals = timingData.data.visuals;
  const lastVisual = visuals[visuals.length - 1];
  const totalDurationSeconds = lastVisual ? lastVisual.end : 59.267;
  const durationInFrames = Math.max(1, Math.round(totalDurationSeconds * FPS));

  return {
    durationInFrames,
    fps: FPS,
    width: 1080,
    height: 1920,
    props: {
      ...props,
      visualTimingData: timingData,
    },
  };
};

export const MyComposition = () => {
  return (
    <Composition
      id="MyComp"
      component={MyComponent}
      durationInFrames={1778}
      fps={FPS}
      width={1080}
      height={1920}
      schema={undefined}
      calculateMetadata={calculateMetadata}
      defaultProps={{
        audioSrc: "narration.mp3",
        transitionAudioSrc: "page-flip.mp3",
        transitionDurationInFrames: DEFAULT_TRANSITION_DURATION_IN_FRAMES,
      }}
    />
  );
};

export const MyComponent: React.FC<CompositionProps> = ({
  visualTimingData,
  audioSrc = "narration.mp3",
  transitionAudioSrc = "page-flip.mp3",
  transitionDurationInFrames = DEFAULT_TRANSITION_DURATION_IN_FRAMES,
}) => {
  const data = visualTimingData || DEFAULT_VISUAL_TIMING_DATA;
  const visuals = data.data.visuals;

  return (
    <>
      {/* Background narration voiceover */}
      {audioSrc && <Audio src={staticFile(audioSrc)} />}

      {/* Dynamic scene transitions */}
      <TransitionSeries>
        {visuals.map((visual, index) => {
          const sceneNumber = visual.visual_index !== undefined ? visual.visual_index + 1 : index + 1;
          const isLastScene = index === visuals.length - 1;

          // Convert visual image name (e.g. "scene_1.png" -> "scene 1.mp4") or use index
          const videoFilename = visual.image
            ? visual.image.replace(/\.[^.]+$/, "").replace(/_/g, " ") + ".mp4"
            : `scene ${sceneNumber}.mp4`;

          // Calculate timing bounds
          const currentStartFrame = index === 0 ? 0 : Math.round(visual.start * FPS);
          const nextStartFrame = !isLastScene
            ? Math.round(visuals[index + 1].start * FPS)
            : Math.round(visual.end * FPS);

          const nominalDurationInFrames = Math.max(1, nextStartFrame - currentStartFrame);

          // Extend non-last scenes by transition length so next scene begins at exact authoritative time
          const sequenceDurationInFrames = isLastScene
            ? nominalDurationInFrames
            : nominalDurationInFrames + transitionDurationInFrames;

          return (
            <React.Fragment key={`scene-fragment-${sceneNumber}`}>
              <TransitionSeries.Sequence
                durationInFrames={sequenceDurationInFrames}
                key={`scene-seq-${sceneNumber}`}
              >
                <OffthreadVideo
                  src={staticFile(videoFilename)}
                  muted
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                  }}
                />
              </TransitionSeries.Sequence>

              {!isLastScene && (
                <TransitionSeries.Transition
                  key={`transition-${sceneNumber}`}
                  presentation={getSceneTransition("from-right")}
                  timing={getSceneTransitionTiming(transitionDurationInFrames)}
                />
              )}
            </React.Fragment>
          );
        })}
      </TransitionSeries>

      {/* Audio transitions dynamically synchronized at each scene boundary */}
      {transitionAudioSrc &&
        visuals.map((visual, index) => {
          if (index === visuals.length - 1) return null;

          const nextVisual = visuals[index + 1];
          const transitionStartFrame = Math.round(nextVisual.start * FPS);

          return (
            <Sequence
              key={`transition-sfx-${index}`}
              from={transitionStartFrame}
              durationInFrames={transitionDurationInFrames}
            >
              <Audio
                src={staticFile(transitionAudioSrc)}
                volume={0.35}
              />
            </Sequence>
          );
        })}
    </>
  );
};