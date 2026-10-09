import { wipe } from "@remotion/transitions/wipe";
import { linearTiming } from "@remotion/transitions";

/**
 * Returns a wipe transition presentation configuration for use with TransitionSeries.Transition.
 * Gives a page-opening-style wipe between history scenes.
 */
export const getSceneTransition = (direction: string = "from-right") => {
  return wipe({
    direction,
  });
};

/**
 * Returns linear timing for the transition duration.
 * Default is 12 frames (0.4s @ 30fps) matching the natural pause between scenes.
 */
export const getSceneTransitionTiming = (durationInFrames: number = 12) => {
  return linearTiming({
    durationInFrames,
  });
};