"""Speaker diarization: label who is speaking in each transcript segment.

Transcription produces timed words, but no speaker labels — a transcript
of a two-person interview is one flat stream of segments. Diarization
fills that gap by matching each segment's audio against a
:class:`~video_dubber.voice.VoiceLibrary`: the segment's embedding is
compared with every stored profile, and the best match above the
threshold becomes the segment's ``speaker`` label.

This is diarization-lite, and the docstring is honest about it:

* The embedding is the same stdlib spectral fingerprint
  :func:`~video_dubber.voice.extract_embedding` computes for voice
  cloning — a timbre fingerprint, not a neural speaker encoder. It
  separates clearly different voices well, but two similar voices (or
  very short segments, under ~0.3 s) can land close together, and then
  the segment is labelled ``unknown`` rather than guessed.
* A segment whose audio is silent (or too short to analyse) is labelled
  ``unknown`` and never fails the run — a bad label is worse than none.
* Segments that already carry a speaker label are left alone unless
  ``overwrite=True``; diarization is a labelling pass, not a relabelling
  one. Labelled transcripts feed straight into multi-speaker dubbing
  (:mod:`video_dubber.assemble` ``profiles=``), which is the point:
  ``transcribe -> diarize -> translate -> dub`` with no hand-labelling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from video_dubber.audio import AudioData
from video_dubber.transcript import Segment, Transcript
from video_dubber.voice import VoiceCloneError, VoiceLibrary, cosine_similarity, extract_embedding

__all__ = [
    "DiarizationReport",
    "DiarizeError",
    "SegmentLabel",
    "diarize_transcript",
]

DEFAULT_THRESHOLD = 0.90


class DiarizeError(Exception):
    """Raised when diarization cannot run at all (empty library, bad input)."""


@dataclass
class SegmentLabel:
    """What diarization decided for one segment.

    ``speaker`` is the matched name, or ``None`` when the segment is
    unknown. ``similarity`` is the best cosine similarity found (``None``
    when the segment was kept as-is or had no usable audio).
    ``status`` is ``matched`` / ``unknown`` / ``kept`` / ``silent``.
    """

    index: int
    speaker: str | None
    similarity: float | None
    status: str
    start: float
    end: float
    text: str


@dataclass
class DiarizationReport:
    """Per-segment labels plus counts, JSON/Markdown serialisable."""

    labels: list[SegmentLabel] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return sum(1 for label in self.labels if label.status == "matched")

    @property
    def unknown(self) -> int:
        return sum(1 for label in self.labels if label.status in ("unknown", "silent"))

    @property
    def kept(self) -> int:
        return sum(1 for label in self.labels if label.status == "kept")

    @property
    def speakers(self) -> dict[str, int]:
        """Segment counts per speaker label (unknown segments excluded)."""
        counts: dict[str, int] = {}
        for label in self.labels:
            if label.speaker is not None:
                counts[label.speaker] = counts.get(label.speaker, 0) + 1
        return counts

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Diarization report",
            "",
            (
                f"{len(self.labels)} segments: {self.matched} matched, "
                f"{self.kept} kept, {self.unknown} unknown."
            ),
            "",
        ]
        for label in self.labels:
            speaker = label.speaker or "unknown"
            similarity = (
                f" (similarity {label.similarity:.3f})" if label.similarity is not None else ""
            )
            lines.append(
                f"- [{label.start:7.2f} - {label.end:7.2f}] {speaker}{similarity} — {label.text}"
            )
        return "\n".join(lines) + "\n"


def _best_match(
    embedding: list[float], library: VoiceLibrary, threshold: float
) -> tuple[str | None, float]:
    """Best-matching speaker name and its similarity (None below threshold)."""
    best_name, best_score = None, threshold
    for name in library.names():
        score = cosine_similarity(embedding, library.get(name).embedding)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score


def diarize_transcript(
    transcript: Transcript,
    audio: AudioData,
    library: VoiceLibrary,
    threshold: float = DEFAULT_THRESHOLD,
    overwrite: bool = False,
) -> tuple[Transcript, DiarizationReport]:
    """Label each segment of ``transcript`` with a speaker from ``library``.

    The segment's audio (``audio`` sliced to the segment's word timings)
    is embedded and matched against the library; the best match above
    ``threshold`` wins. Returns a new transcript (the input is untouched)
    plus a :class:`DiarizationReport`. Segments already labelled are kept
    unless ``overwrite`` is set; empty segments are skipped as silent.
    """
    if len(library) == 0:
        raise DiarizeError("cannot diarize with an empty voice library")
    if not 0.0 <= threshold <= 1.0:
        raise DiarizeError(f"match threshold must be in [0, 1], got {threshold}")

    labelled_segments: list[Segment] = []
    labels: list[SegmentLabel] = []
    for index, segment in enumerate(transcript.segments):
        if not segment.words:
            labels.append(
                SegmentLabel(
                    index=index,
                    speaker=segment.speaker,
                    similarity=None,
                    status="silent",
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )
            labelled_segments.append(segment)
            continue
        if segment.speaker is not None and not overwrite:
            labels.append(
                SegmentLabel(
                    index=index,
                    speaker=segment.speaker,
                    similarity=None,
                    status="kept",
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )
            labelled_segments.append(segment)
            continue
        try:
            clip = audio.slice(segment.start, segment.end)
            embedding = extract_embedding(clip)
        except VoiceCloneError:
            labels.append(
                SegmentLabel(
                    index=index,
                    speaker=None,
                    similarity=None,
                    status="silent",
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
            )
            labelled_segments.append(Segment(words=segment.words, speaker=None))
            continue
        name, similarity = _best_match(embedding, library, threshold)
        status = "matched" if name is not None else "unknown"
        labels.append(
            SegmentLabel(
                index=index,
                speaker=name,
                similarity=similarity if name is not None else None,
                status=status,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )
        labelled_segments.append(Segment(words=segment.words, speaker=name))
    labelled = Transcript(
        segments=labelled_segments,
        language=transcript.language,
        source=transcript.source,
    )
    return labelled, DiarizationReport(labels=labels)
