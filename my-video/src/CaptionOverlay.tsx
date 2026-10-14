import React, { useMemo } from "react";
import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig } from "remotion";
import { createTikTokStyleCaptions } from "@remotion/captions";
import type { Caption, TikTokPage } from "@remotion/captions";

// ---------------------------------------------------------------------------
// Constants & Defaults
// ---------------------------------------------------------------------------

export const DEFAULT_HIGHLIGHT_COLOR = "#39E508"; // Neon green highlight
export const DEFAULT_SWITCH_CAPTIONS_EVERY_MS = 1200; // ms to group onto one page in phrase mode

export type CaptionDisplayMode = "word" | "phrase";

export interface EmojiSelection {
  clip_start: number;
  clip_end: number;
  emoji: string;
}

export interface EmojiCandidate {
  candidate_id?: number;
  emoji_selections?: EmojiSelection[];
}

export interface CaptionOverlayProps {
  captions: Caption[];
  emojiCandidate?: EmojiCandidate | null;
  displayMode?: CaptionDisplayMode;
  highlightColor?: string;
  fontSize?: number;
  combineTokensWithinMilliseconds?: number;
  position?: "center" | "bottom";
  bottomOffsetPercent?: number;
}

// ---------------------------------------------------------------------------
// CaptionPage — renders a page (word or phrase) with dynamic word highlight
// ---------------------------------------------------------------------------

const CaptionPage: React.FC<{
  page: TikTokPage;
  emojiCandidate?: EmojiCandidate | null;
  highlightColor: string;
  fontSize: number;
  position: "center" | "bottom";
  bottomOffsetPercent: number;
}> = ({
  page,
  emojiCandidate,
  highlightColor,
  fontSize,
  position,
  bottomOffsetPercent,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // frame is relative to Sequence start; add page.startMs for absolute video time
  const currentTimeMs = (frame / fps) * 1000;
  const absoluteTimeMs = page.startMs + currentTimeMs;

  // Resolve matching emoji at current absolute clip time (from youtube_clipping pattern)
  let emoji = "";
  if (emojiCandidate) {
    for (const selection of emojiCandidate.emoji_selections ?? []) {
      const startMs = selection.clip_start * 1000;
      const endMs = selection.clip_end * 1000;
      if (absoluteTimeMs >= startMs && absoluteTimeMs < endMs) {
        emoji = selection.emoji;
        break;
      }
    }
  }

  return (
    <AbsoluteFill
      style={{
        justifyContent: position === "bottom" ? "flex-end" : "center",
        alignItems: "center",
        paddingBottom: position === "bottom" ? `${bottomOffsetPercent}%` : 0,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          display: "inline-flex",
          flexWrap: "wrap",
          justifyContent: "center",
          alignItems: "center",
          maxWidth: "88%",
          textAlign: "center",
          fontSize,
          fontWeight: 900,
          fontFamily:
            "system-ui, -apple-system, BlinkMacSystemFont, 'Impact', 'Segoe UI', Roboto, sans-serif",
          textTransform: "uppercase",
          letterSpacing: "1px",
          lineHeight: 1.25,
          padding: "10px 24px",
          borderRadius: "16px",
          backgroundColor: "rgba(0, 0, 0, 0.45)",
          backdropFilter: "blur(2px)",
        }}
      >
        {page.tokens.map((token, tokenIndex) => {
          const isActive =
            token.fromMs <= absoluteTimeMs && token.toMs > absoluteTimeMs;

          return (
            <span
              key={`${token.fromMs}-${tokenIndex}`}
              style={{
                color: isActive ? highlightColor : "#FFFFFF",
                transform: isActive ? "scale(1.1)" : "scale(1.0)",
                display: "inline-block",
                margin: "0 6px",
                textShadow: isActive
                  ? `0 0 16px ${highlightColor}, 0 2px 4px rgba(0,0,0,0.9), 2px 2px 2px black`
                  : "0 2px 4px rgba(0,0,0,0.9), 2px 2px 2px black",
                WebkitTextStroke: isActive ? "2px black" : "1.5px black",
              }}
            >
              {token.text.trim()}
            </span>
          );
        })}
        {emoji && (
          <span
            style={{
              marginLeft: 12,
              fontSize: Math.round(fontSize * 1.1),
              display: "inline-block",
            }}
          >
            {emoji}
          </span>
        )}
      </div>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------
// CaptionOverlay — renders timed sequences based on Remotion best practices
// ---------------------------------------------------------------------------

export const CaptionOverlay: React.FC<CaptionOverlayProps> = ({
  captions,
  emojiCandidate = null,
  displayMode = "phrase",
  highlightColor = DEFAULT_HIGHLIGHT_COLOR,
  fontSize: customFontSize,
  combineTokensWithinMilliseconds,
  position = "bottom",
  bottomOffsetPercent = 14,
}) => {
  const { fps, width, height } = useVideoConfig();

  // Dynamic responsive font size if not explicitly specified
  const isPortrait = height > width;
  const responsiveFontSize = customFontSize ?? (isPortrait ? 68 : 52);

  // Determine windowing duration
  // If displayMode === "word", set pageBreakAfter on every token and combine=0 for 1 word per page
  // If displayMode === "phrase", use combineTokensWithinMilliseconds (default 1200ms)
  const isWordByWord = displayMode === "word";
  const effectiveWindowMs = isWordByWord
    ? 0
    : (combineTokensWithinMilliseconds ?? DEFAULT_SWITCH_CAPTIONS_EVERY_MS);

  const preparedCaptions: Caption[] = useMemo(() => {
    if (!captions || captions.length === 0) return [];
    if (!isWordByWord) return captions;

    // Force page break on every word for single word-by-word display
    return captions.map((c) => ({
      ...c,
      pageBreakAfter: true,
    }));
  }, [captions, isWordByWord]);

  const { pages } = useMemo(() => {
    if (!preparedCaptions.length) return { pages: [] };
    return createTikTokStyleCaptions({
      captions: preparedCaptions,
      combineTokensWithinMilliseconds: effectiveWindowMs,
    });
  }, [preparedCaptions, effectiveWindowMs]);

  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      {pages.map((page, index) => {
        const nextPage = pages[index + 1] ?? null;
        const lastToken = page.tokens[page.tokens.length - 1];
        const nextStartMs = nextPage
          ? nextPage.startMs
          : page.startMs + page.durationMs;
        const pageEndMs = lastToken
          ? lastToken.toMs
          : page.startMs + page.durationMs;

        // Hold caption while spoken + up to 150ms of trailing pause, or until next page starts
        const endMs = Math.min(
          nextStartMs,
          Math.max(pageEndMs, page.startMs + 100) + 150
        );

        const startFrame = (page.startMs / 1000) * fps;
        const endFrame = (endMs / 1000) * fps;
        const durationInFrames = Math.max(1, Math.round(endFrame - startFrame));

        if (durationInFrames <= 0) return null;

        return (
          <Sequence
            key={`caption-seq-${index}-${page.startMs}`}
            from={Math.round(startFrame)}
            durationInFrames={durationInFrames}
          >
            <CaptionPage
              page={page}
              emojiCandidate={emojiCandidate}
              highlightColor={highlightColor}
              fontSize={responsiveFontSize}
              position={position}
              bottomOffsetPercent={bottomOffsetPercent}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
