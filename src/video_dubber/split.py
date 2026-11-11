"""Split long transcript segments at silence gaps.

A long segment is the dub assembler's worst case: the whole line must be
stretched or squeezed into one window, so a 20-second sentence in a
12-second window needs a 0.6x fit and the quality report flags it. Splitting
the segment at a real pause gives the assembler two shorter windows that
each fit comfortably, and the subtitles read better too.

Splitting never cuts inside a word. A boundary between two words is a
split candidate when the pause there is at least ``min_gap`` seconds long
— read off the word timings themselves, and, when the source audio is
given, confirmed against the VAD: a boundary counts even when the word
timings are coarse if the boundary falls inside a real silence gap in the
audio. When a piece runs over ``max_duration`` with no silence candidate,
the split falls back to the longest pause inside the piece, and only as a
last resort to a forced cut at the word nearest ``max_duration`` — forced
cuts are counted in the report so the caller can see them, not hidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from video_dubber.audio import AudioData
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.vad import VadConfig, detect_speech


class SplitError(ValueError):
    """Raised for invalid split configuration or input."""


@dataclass(frozen=True)
class SplitRecord:
    """One cut: which segment, where in time, and why that boundary."""

    segment_index: int
    at: float
    reason: str  # "silence" | "longest-gap" | "forced"


@dataclass
class SplitConfig:
    """Tuning knobs for :func:`split_transcript`."""

    # A segment longer than this is split (seconds).
    max_duration: float = 8.0
    # A pause at least this long is a clean split point (seconds).
    min_gap: float = 0.4
    # Never leave a piece shorter than this (seconds) when a choice exists.
    min_segment: float = 1.0
    vad_config: VadConfig | None = None

    def __post_init__(self) -> None:
        if self.max_duration <= 0:
            raise SplitError(f"max_duration must be positive, got {self.max_duration}")
        if self.min_gap < 0:
            raise SplitError(f"min_gap must not be negative, got {self.min_gap}")
        if self.min_segment < 0:
            raise SplitError(f"min_segment must not be negative, got {self.min_segment}")
        if self.min_segment >= self.max_duration:
            raise SplitError(
                f"min_segment ({self.min_segment}) must be below "
                f"max_duration ({self.max_duration})"
            )


@dataclass
class SplitReport:
    """What :func:`split_transcript` did, for the CLI and the caller."""

    segments_before: int
    segments_after: int
    splits: list[SplitRecord] = field(default_factory=list)
    max_duration_before: float = 0.0
    max_duration_after: float = 0.0

    @property
    def forced_splits(self) -> int:
        return sum(1 for s in self.splits if s.reason == "forced")

    @property
    def total_splits(self) -> int:
        return len(self.splits)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> SplitReport:
        return cls(
            segments_before=int(data["segments_before"]),
            segments_after=int(data["segments_after"]),
            splits=[SplitRecord(**s) for s in data.get("splits", [])],
            max_duration_before=float(data.get("max_duration_before", 0.0)),
            max_duration_after=float(data.get("max_duration_after", 0.0)),
        )

    @classmethod
    def from_json(cls, data: str) -> SplitReport:
        try:
            return cls.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SplitError(f"invalid split report JSON: {exc}") from exc

    def to_markdown(self) -> str:
        """Human-readable report: headline numbers plus one row per cut."""
        lines = ["# Segment split", ""]
        lines.append(
            f"- Segments: {self.segments_before} -> {self.segments_after} "
            f"({self.total_splits} split{'s' if self.total_splits != 1 else ''})"
        )
        lines.append(
            f"- Longest segment: {self.max_duration_before:.2f}s -> "
            f"{self.max_duration_after:.2f}s"
        )
        if self.forced_splits:
            lines.append(
                f"- Forced cuts (no pause found): {self.forced_splits} — "
                "the source has continuous speech there; consider a "
                "shorter max-duration or re-recording the line"
            )
        if self.splits:
            lines.append("")
            lines.append("| Segment | Cut at | Reason |")
            lines.append("| --- | --- | --- |")
            for split in self.splits:
                lines.append(
                    f"| {split.segment_index + 1} | {split.at:.2f}s | "
                    f"{split.reason} |"
                )
        return "\n".join(lines) + "\n"


def _silence_intervals(audio: AudioData, config: SplitConfig) -> list[tuple[float, float]]:
    """Complement of the VAD speech segments: the real pauses in the audio."""
    speech = detect_speech(audio, config.vad_config)
    intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for seg in speech:
        if seg.start > cursor:
            intervals.append((cursor, seg.start))
        cursor = max(cursor, seg.end)
    if cursor < audio.duration:
        intervals.append((cursor, audio.duration))
    return intervals


def _candidate_boundaries(
    words: list[Word],
    silence: list[tuple[float, float]] | None,
    config: SplitConfig,
) -> dict[int, str]:
    """Word-boundary index -> reason, for every splittable boundary.

    A boundary after word ``i`` is splittable when the pause to the next
    word is at least ``min_gap``, or — when audio was given — when the
    boundary falls inside a real silence interval at least ``min_gap``
    long (this covers coarse word timings that hide a real pause). VAD
    edges are frame-quantised (30 ms frames) plus hangover, so a boundary
    up to 150 ms before a silence interval starts still counts.
    """
    candidates: dict[int, str] = {}
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap >= config.min_gap:
            candidates[i] = "silence"
            continue
        if silence is not None:
            point = words[i].end
            for start, end in silence:
                if (
                    end - start >= config.min_gap
                    and start - 0.15 <= point <= end
                ):
                    candidates[i] = "silence"
                    break
    return candidates


def _split_words(
    words: list[Word],
    candidates: dict[int, str],
    segment_index: int,
    config: SplitConfig,
) -> tuple[list[list[Word]], list[SplitRecord]]:
    """Greedy split of one segment's words into pieces <= max_duration.

    Walks forward; when the running piece runs over ``max_duration`` it
    cuts at the latest silence candidate inside the piece, else the
    longest pause, else a forced cut at the word nearest ``max_duration``.
    A single word longer than ``max_duration`` is emitted alone — a word
    can never be cut.
    """
    pieces: list[list[Word]] = []
    splits: list[SplitRecord] = []
    start = 0
    n = len(words)
    while start < n:
        piece_start = words[start].start
        # Grow until the piece is too long, or the words run out.
        end = start
        while end + 1 < n and words[end + 1].end - piece_start <= config.max_duration:
            end += 1
        if end == n - 1:
            pieces.append(words[start:])
            break
        # words[start..end] fits; adding words[end + 1] would overflow.
        # Choose the cut boundary inside [start, end].
        cut: int | None = None
        reason = "forced"
        for b in range(end, start - 1, -1):
            if b in candidates and words[b].end - piece_start >= config.min_segment:
                cut = b
                reason = candidates[b]
                break
        if cut is None:
            # Longest pause inside the piece beats a forced cut. The
            # boundary after the piece's last word is a valid cut too.
            best_gap = 0.0
            for b in range(start, end + 1):
                gap = words[b + 1].start - words[b].end
                if gap > best_gap and words[b].end - piece_start >= config.min_segment:
                    best_gap = gap
                    cut = b
            if cut is not None and best_gap > 0.0:
                reason = "longest-gap"
        if cut is None:
            # No pause at all: cut at the word nearest max_duration.
            target = piece_start + config.max_duration
            cut = min(
                range(start, end + 1),
                key=lambda b: abs(words[b].end - target),
            )
            reason = "forced"
        pieces.append(words[start : cut + 1])
        splits.append(
            SplitRecord(
                segment_index=segment_index,
                at=words[cut].end,
                reason=reason,
            )
        )
        start = cut + 1
    return pieces, splits


def split_transcript(
    transcript: Transcript,
    audio: AudioData | None = None,
    config: SplitConfig | None = None,
) -> tuple[Transcript, SplitReport]:
    """Split every over-long segment at silence gaps.

    Returns a new transcript (the input is untouched) plus a report.
    Segments within ``max_duration`` pass through unchanged, with speaker
    labels kept on every piece. ``audio`` is optional: without it, split
    points come from the word timings alone; with it, VAD silence gaps
    confirm boundaries the timings hide.
    """
    config = config or SplitConfig()
    silence = _silence_intervals(audio, config) if audio is not None else None
    segments: list[Segment] = []
    splits: list[SplitRecord] = []
    max_before = 0.0
    max_after = 0.0
    for index, segment in enumerate(transcript.segments):
        max_before = max(max_before, segment.duration)
        if len(segment.words) < 2 or segment.duration <= config.max_duration:
            segments.append(segment)
            max_after = max(max_after, segment.duration)
            continue
        candidates = _candidate_boundaries(segment.words, silence, config)
        pieces, piece_splits = _split_words(
            segment.words, candidates, index, config
        )
        splits.extend(piece_splits)
        for piece in pieces:
            segments.append(Segment(words=piece, speaker=segment.speaker))
            max_after = max(max_after, piece[-1].end - piece[0].start)
    report = SplitReport(
        segments_before=len(transcript.segments),
        segments_after=len(segments),
        splits=splits,
        max_duration_before=max_before,
        max_duration_after=max_after,
    )
    return (
        Transcript(segments=segments, language=transcript.language, source=transcript.source),
        report,
    )
