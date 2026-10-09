import { bookFlip, BookFlipDirection } from "@remotion/transitions/book-flip";
import { linearTiming } from "@remotion/transitions";

/**
 * Returns a book-flip transition presentation configuration for use with TransitionSeries.Transition.
 * Gives a page-flipping / book-turn feel between history scenes.
 */
export const getSceneTransition = (direction: BookFlipDirection = "from-right") => {
  return bookFlip({
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