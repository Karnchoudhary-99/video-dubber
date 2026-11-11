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
