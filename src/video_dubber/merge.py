"""Merge over-split transcript segments back into readable lines.

Splitting is the right call for a long line, but transcription and VAD
over-segment too: a hesitant speaker arrives as five 0.4 s fragments,
and dubbing each one separately means five tiny syntheses, five fit
factors, and subtitles that flash past faster than anyone can read.
Merging is split's counterpart: adjacent segments join back together
when the pause between them is short and the combined line still fits
one dub window.

Merging never crosses a speaker change — two voices in one cue would
dub in one voice and read as one speaker — and never cuts inside a
word: a merge only joins whole segments, keeping every word's timing.
The input transcript is untouched; the merged transcript and a report
come back instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from video_dubber.transcript import Segment, Transcript, Word


class MergeError(ValueError):
    """Raised for invalid merge configuration or input."""


@dataclass(frozen=True)
class MergeRecord:
    """One join: which original segments merged, and why it was allowed."""

    segment_indices: list[int]
    reason: str  # "short-gap" (pause under max_gap, combined line fits)


@dataclass
class MergeConfig:
    """Tuning knobs for :func:`merge_transcript`."""

    # Join across a pause at most this long (seconds).
    max_gap: float = 0.4
    # Never produce a merged segment longer than this (seconds).
    max_duration: float = 8.0

    def __post_init__(self) -> None:
        if self.max_gap < 0:
            raise MergeError(f"max_gap must not be negative, got {self.max_gap}")
        if self.max_duration <= 0:
            raise MergeError(f"max_duration must be positive, got {self.max_duration}")


def merge_transcript(
    transcript: Transcript, config: MergeConfig | None = None
) -> tuple[Transcript, "MergeReport"]:
    """Join adjacent short segments into readable lines.

    A left-to-right greedy pass: a segment joins the group before it
    when all three hold — same speaker label (``None`` counts as a
    label, so two unlabelled lines merge but Asha never merges into
    Ravi), the pause between them is at most ``config.max_gap``, and
    the combined line is at most ``config.max_duration`` long. A
    merged group keeps every word in order with its original timings,
    so the result still dubs and subtitles exactly like the input.
    Wordless segments are copied through unchanged and break a group:
    they carry no timing, so they can neither join nor be joined across.

    Returns the merged transcript and a :class:`MergeReport`.
    """
    cfg = config or MergeConfig()
    segments = transcript.segments
    merged: list[Segment] = []
    merges: list[MergeRecord] = []
    group: list[Segment] = []
    group_indices: list[int] = []

    def flush_group() -> None:
        if not group:
            return
        words: list[Word] = [w for segment in group for w in segment.words]
        merged.append(Segment(words=words, speaker=group[0].speaker))
        if len(group) > 1:
            merges.append(MergeRecord(segment_indices=list(group_indices), reason="short-gap"))
        group.clear()
        group_indices.clear()

    for index, segment in enumerate(segments):
        if not segment.words:
            flush_group()
            merged.append(Segment(words=[], speaker=segment.speaker))
            continue
        if not group:
            group, group_indices = [segment], [index]
            continue
        gap = segment.start - group[-1].end
        combined = segment.end - group[0].start
        if segment.speaker == group[0].speaker and gap <= cfg.max_gap and combined <= cfg.max_duration:
            group.append(segment)
            group_indices.append(index)
        else:
            flush_group()
            group, group_indices = [segment], [index]
    flush_group()

    merged_transcript = Transcript(
        segments=merged, language=transcript.language, source=transcript.source
    )
    report = MergeReport(
        segments_before=len(segments),
        segments_after=len(merged),
        merges=merges,
        max_duration_before=max((s.duration for s in segments), default=0.0),
        max_duration_after=max((s.duration for s in merged), default=0.0),
    )
    return merged_transcript, report


@dataclass
class MergeReport:
    """What :func:`merge_transcript` did, for the CLI and the caller."""

    segments_before: int
    segments_after: int
    merges: list[MergeRecord] = field(default_factory=list)
    max_duration_before: float = 0.0
    max_duration_after: float = 0.0

    @property
    def total_merges(self) -> int:
        return len(self.merges)

    @property
    def segments_merged(self) -> int:
        """How many original segments disappeared into a neighbour."""
        return sum(len(m.segment_indices) - 1 for m in self.merges)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "MergeReport":
        return cls(
            segments_before=int(data["segments_before"]),
            segments_after=int(data["segments_after"]),
            merges=[MergeRecord(**m) for m in data.get("merges", [])],
            max_duration_before=float(data.get("max_duration_before", 0.0)),
            max_duration_after=float(data.get("max_duration_after", 0.0)),
        )

    @classmethod
    def from_json(cls, data: str) -> "MergeReport":
        try:
            return cls.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MergeError(f"invalid merge report JSON: {exc}") from exc

    def to_markdown(self) -> str:
        """Human-readable report: headline numbers plus one row per join."""
        lines = ["# Segment merge", ""]
        lines.append(
            f"- Segments: {self.segments_before} -> {self.segments_after} "
            f"({self.total_merges} merge{'s' if self.total_merges != 1 else ''}, "
            f"{self.segments_merged} segment{'s' if self.segments_merged != 1 else ''} joined)"
        )
        lines.append(
            f"- Longest segment: {self.max_duration_before:.2f}s -> {self.max_duration_after:.2f}s"
        )
        if self.merges:
            lines += ["", "| Segments | Reason |", "| --- | --- |"]
            for merge in self.merges:
                joined = "+".join(str(i + 1) for i in merge.segment_indices)
                lines.append(f"| {joined} | {merge.reason} |")
        return "\n".join(lines) + "\n"
