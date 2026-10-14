import React, { useState, useEffect, useCallback, useMemo } from "react";
import { staticFile, useDelayRender } from "remotion";
import type { Caption } from "@remotion/captions";
import {
  CaptionOverlay,
  CaptionDisplayMode,
  EmojiCandidate,
  DEFAULT_HIGHLIGHT_COLOR,
} from "./CaptionOverlay";

// ---------------------------------------------------------------------------
// Types & Schema Normalization
// ---------------------------------------------------------------------------

export interface WhisperXWord {
  index?: number;
  word: string;
  start: number;
  end: number;
  score?: number;
}

export interface WhisperXSegment {
  start?: number;
  end?: number;
  text?: string;
  words?: WhisperXWord[];
}

export interface WhisperXTimingPayload {
  stage?: string;
  data?: {
    audio_path?: string;
    duration?: number;
    words?: WhisperXWord[];
    segments?: WhisperXSegment[];
  };
  words?: WhisperXWord[];
  meta?: Record<string, unknown>;
}

export interface YoutubeClippingWord {
  word: string;
  clip_start: number;
  clip_end: number;
  global_start?: number;
  global_end?: number;
  score?: number;
}

export interface YoutubeClippingGroup {
  group_id?: string;
  start_time?: number;
  end_time?: number;
  words?: YoutubeClippingWord[];
}

export interface YoutubeClippingCandidate {
  candidate_id?: number;
  caption_groups?: YoutubeClippingGroup[];
}

export interface YoutubeClippingManifest {
  candidates?: YoutubeClippingCandidate[];
  caption_groups?: YoutubeClippingGroup[];
}

export type AnyTimingPayload =
  | WhisperXTimingPayload
  | YoutubeClippingManifest
  | Caption[]
  | Record<string, unknown>;

export interface CaptionLayerProps {
  timingSrc?: string;
  timingData?: AnyTimingPayload;
  emojiCandidate?: EmojiCandidate | null;
  displayMode?: CaptionDisplayMode;
  highlightColor?: string;
  fontSize?: number;
  combineTokensWithinMilliseconds?: number;
  position?: "center" | "bottom";
  bottomOffsetPercent?: number;
}

// ---------------------------------------------------------------------------
// Normalizer helper: handles WhisperX, youtube_clipping, and raw Caption arrays
// ---------------------------------------------------------------------------

export function normalizeTimingToCaptions(
  raw: AnyTimingPayload | null | undefined
): Caption[] {
  if (!raw) return [];

  // Case 1: Already an array of standard Caption objects
  if (Array.isArray(raw)) {
    if (raw.length > 0 && "startMs" in raw[0] && "text" in raw[0]) {
      return raw as Caption[];
    }
    // Array of { word, start, end }
    return (raw as unknown as WhisperXWord[])
      .filter((w) => w && typeof w.word === "string")
      .map((w) => ({
        text: ` ${w.word.trim()}`,
        startMs: Math.round(Number(w.start ?? 0) * 1000),
        endMs: Math.round(Number(w.end ?? w.start ?? 0) * 1000),
        timestampMs: Math.round(Number(w.start ?? 0) * 1000),
        confidence: w.score ?? null,
      }));
  }

  // Case 2: WhisperX timing schema ({ stage: "whisperx_timing", data: { words: [...] } })
  const whisperPayload = raw as WhisperXTimingPayload;
  const wordsList: WhisperXWord[] =
    whisperPayload.data?.words ??
    whisperPayload.words ??
    whisperPayload.data?.segments?.flatMap((s: WhisperXSegment) => s.words ?? []) ??
    [];

  if (Array.isArray(wordsList) && wordsList.length > 0) {
    return wordsList
      .filter((w) => w && typeof w.word === "string")
      .map((w) => ({
        text: ` ${w.word.trim()}`,
        startMs: Math.round(Number(w.start) * 1000),
        endMs: Math.round(Number(w.end) * 1000),
        timestampMs: Math.round(Number(w.start) * 1000),
        confidence: w.score ?? null,
      }));
  }

  // Case 3: youtube_clipping caption manifest schema
  const clippingPayload = raw as YoutubeClippingManifest;
  const groups: YoutubeClippingGroup[] =
    clippingPayload.caption_groups ??
    clippingPayload.candidates?.[0]?.caption_groups ??
    [];

  if (Array.isArray(groups) && groups.length > 0) {
    const result: Caption[] = [];
    for (const group of groups) {
      for (const w of group.words ?? []) {
        if (!w || typeof w.word !== "string") continue;
        result.push({
          text: ` ${w.word.trim()}`,
          startMs: Math.round(Number(w.clip_start) * 1000),
          endMs: Math.round(Number(w.clip_end) * 1000),
          timestampMs: Math.round(Number(w.clip_start) * 1000),
          confidence: w.score ?? null,
        });
      }
    }
    return result;
  }

  return [];
}

// ---------------------------------------------------------------------------
// CaptionLayer — dynamic loader using Remotion best practices
// ---------------------------------------------------------------------------

export const CaptionLayer: React.FC<CaptionLayerProps> = ({
  timingSrc = "timing.json",
  timingData,
  emojiCandidate = null,
  displayMode = "phrase",
  highlightColor = DEFAULT_HIGHLIGHT_COLOR,
  fontSize,
  combineTokensWithinMilliseconds,
  position = "bottom",
  bottomOffsetPercent = 14,
}) => {
  const [loadedPayload, setLoadedPayload] = useState<AnyTimingPayload | null>(
    timingData ?? null
  );

  const { delayRender, continueRender, cancelRender } = useDelayRender();
  const [handle] = useState(() =>
    timingData ? null : delayRender("Loading dynamic captions timing")
  );

  const fetchTiming = useCallback(async () => {
    if (timingData) return;

    try {
      // First attempt: try timingSrc (defaults to timing.json)
      let res = await fetch(staticFile(timingSrc));

      // Fallback: if 404 and timingSrc was timing.json, try whisperx_timing.json
      if (!res.ok && timingSrc === "timing.json") {
        res = await fetch(staticFile("whisperx_timing.json"));
      }

      if (!res.ok) {
        throw new Error(
          `Failed to load captions from ${timingSrc}: HTTP status ${res.status}`
        );
      }

      const json = await res.json();
      setLoadedPayload(json);

      if (handle !== null) {
        continueRender(handle);
      }
    } catch (err) {
      console.error("[CaptionLayer] Failed to load captions:", err);
      if (handle !== null) {
        cancelRender(err);
      }
    }
  }, [timingSrc, timingData, handle, continueRender, cancelRender]);

  useEffect(() => {
    fetchTiming();
  }, [fetchTiming]);

  // Normalize whatever timing payload was provided / loaded
  const captions = useMemo(() => {
    return normalizeTimingToCaptions(loadedPayload);
  }, [loadedPayload]);

  if (!captions || captions.length === 0) {
    return null;
  }

  return (
    <CaptionOverlay
      captions={captions}
      emojiCandidate={emojiCandidate}
      displayMode={displayMode}
      highlightColor={highlightColor}
      fontSize={fontSize}
      combineTokensWithinMilliseconds={combineTokensWithinMilliseconds}
      position={position}
      bottomOffsetPercent={bottomOffsetPercent}
    />
  );
};
