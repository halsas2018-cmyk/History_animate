import { CalculateMetadataFunction, Composition, OffthreadVideo, Audio, staticFile, Sequence } from "remotion";

type Props = {};

interface VisualTimingData {
  stage: string;
  data: {
    visuals: Array<{
      visual_index: number;
      sentence_indices: number[];
      image: string;
      start: number;
      end: number;
      duration: number;
    }>;
  };
  meta: {
    visual_count: number;
    total_sentences: number;
  };
}

// Since we can't read files directly in the browser environment,
// we'll use a simpler approach and pass the timing data through props
// or use a fixed duration for now
const calculateMetadata: CalculateMetadataFunction<Props> = () => {
  // Fixed duration based on the visual_timing.json we saw earlier
  // Last scene ends at 59.267 seconds
  const totalDuration = 59.267; // seconds

  return {
    durationInFrames: Math.round(totalDuration * 30), // 30 fps
    fps: 30,
    width: 1080,
    height: 1920
  };
};

export const MyComposition = () => {
  return (
    <Composition
      id="MyComp"
      component={MyComponent}
      durationInFrames={1778} // 59.267 seconds * 30 fps = 1778 frames (rounded)
      fps={30}
      width={1080}
      height={1920}
      calculateMetadata={calculateMetadata}
    />
  );
};

export const MyComponent: React.FC<Props> = () => {
  // Hardcoded timing data from visual_timing.json for now
  // In a more advanced implementation, this could be fetched or passed differently
  const visualTimingData = {
    "stage": "visual_timing",
    "data": {
      "visuals": [
        {
          "visual_index": 0,
          "sentence_indices": [0, 1],
          "image": "scene_1.png",
          "start": 0.2,
          "end": 13.567,
          "duration": 13.367
        },
        {
          "visual_index": 1,
          "sentence_indices": [2, 3],
          "image": "scene_2.png",
          "start": 13.933,
          "end": 28.3,
          "duration": 14.367
        },
        {
          "visual_index": 2,
          "sentence_indices": [4, 5],
          "image": "scene_3.png",
          "start": 28.767,
          "end": 42.767,
          "duration": 14.0
        },
        {
          "visual_index": 3,
          "sentence_indices": [6, 7],
          "image": "scene_4.png",
          "start": 43.2,
          "end": 53.6,
          "duration": 10.4
        },
        {
          "visual_index": 4,
          "sentence_indices": [8],
          "image": "scene_5.png",
          "start": 53.967,
          "end": 59.267,
          "duration": 5.3
        }
      ]
    },
    "meta": {
      "visual_count": 5,
      "total_sentences": 9
    }
  };

  return (
    <>
      {/* Narration audio - plays throughout the entire composition */}
      <Audio src={staticFile("narration.mp3")} />

      {/* Scene videos with 2-frame overlap to prevent black/decode gaps */}
      {getSceneComponents(visualTimingData)}
    </>
  );
};

// Helper function to generate scene components based on timing data
// Scenes start at their authoritative times to maintain audio-video sync
// Remotion handles internal buffering to minimize decode gaps
function getSceneComponents(visualTimingData: VisualTimingData) {
  return visualTimingData.data.visuals.map((visual, index) => {
    const sceneNumber = index + 1;
    const startTime = visual.start;
    const endTime = visual.end;

    // Convert seconds to frames (30 fps) - NO FRAME SHIFTING to preserve sync
    const startFrame = Math.round(startTime * 30);
    const endFrame = Math.round(endTime * 30);
    const durationInFrames = endFrame - startFrame;

    return (
      <Sequence from={startFrame} durationInFrames={durationInFrames} key={`scene-${sceneNumber}`}>
        <OffthreadVideo
          src={staticFile(`scene ${sceneNumber}.mp4`)}
          muted
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
          // Note: We don't set 'to' prop to allow the video to play its full duration
          // Decode/preload handling is managed internally by Remotion
        />
      </Sequence>
    );
  });
}