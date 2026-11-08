"""Transcript data model: timed words, segments, and subtitle serialisation.

A :class:`Transcript` is what the transcriber produces and everything
downstream (translation, voice cloning, dub assembly) consumes. It
serialises to JSON for the pipeline, and to SRT / WebVTT for humans and
video players.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Word:
    """One transcribed word with timings (seconds) and a confidence 0–1."""

    text: str
    start: float
    end: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"word end {self.end} precedes start {self.start}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@dataclass
class Segment:
    """A sentence/utterance: a run of words, optionally with a speaker label."""

    words: list[Word] = field(default_factory=list)
    speaker: str | None = None

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)


@dataclass
class Transcript:
    """A full transcription: ordered segments, source language, provenance."""

    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    source: str | None = None

    @property
    def words(self) -> list[Word]:
        return [w for segment in self.segments for w in segment.words]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def duration(self) -> float:
        words = self.words
        return words[-1].end - words[0].start if words else 0.0

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    @property
    def words_per_minute(self) -> float:
        if self.duration <= 0:
            return 0.0
        return self.word_count / self.duration * 60.0

    def to_json(self) -> str:
        """Serialise to JSON (stable key order, 2-space indent)."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> Transcript:
        """Parse :meth:`to_json` output. Raises ValueError on bad input."""
        try:
            raw = json.loads(data)
            segments = [
                Segment(
                    words=[Word(**word) for word in seg.get("words", [])],
                    speaker=seg.get("speaker"),
                )
                for seg in raw.get("segments", [])
            ]
            return cls(
                segments=segments,
                language=raw.get("language"),
                source=raw.get("source"),
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid transcript JSON: {exc}") from exc

    def to_srt(self) -> str:
        """Serialise segments as SubRip (one cue per segment)."""
        cues = []
        number = 0
        for segment in self.segments:
            if not segment.words:
                continue
            number += 1
            cues.append(
                f"{number}\n"
                f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n"
                f"{segment.text}\n"
            )
        return "\n".join(cues)

    def to_vtt(self) -> str:
        """Serialise segments as WebVTT (one cue per segment)."""
        cues = ["WEBVTT\n"]
        for segment in self.segments:
            if not segment.words:
                continue
            cues.append(
                f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n"
                f"{segment.text}\n"
            )
        return "\n".join(cues)


def _srt_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT/VTT share this layout)."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def from_words(
    words: list[Word],
    max_gap: float = 0.8,
    language: str | None = None,
    source: str | None = None,
) -> Transcript:
    """Group a flat word list into segments at pauses longer than ``max_gap``."""
    segments: list[Segment] = []
    current: list[Word] = []
    for word in words:
        if current and word.start - current[-1].end > max_gap:
            segments.append(Segment(current))
            current = []
        current.append(word)
    if current:
        segments.append(Segment(current))
    return Transcript(segments=segments, language=language, source=source)
