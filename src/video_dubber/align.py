"""Align subtitle cues to the speech actually in the rendered dub audio.

:func:`video_dubber.subtitles.retime_transcript` retimes cues to the
*placed* spans the assembler planned — but the synthesiser does not
always start speaking exactly at the window edge. A segment whose
synthesis opens with 300 ms of leading silence (or a breath, or a
fade-in) leaves its subtitle on screen over silence, and a segment
that finishes early leaves the cue lingering after the voice stops.

This module fixes that after the fact, on the finished track:

1. Run the energy VAD (:mod:`video_dubber.vad`) once over the whole
   dubbed track — one pass, so the adaptive noise floor comes from
   the track as a whole, not from a single window that might be all
   speech.
2. For each timed segment, take the speech that overlaps its planned
   window: the first overlapping onset becomes the cue start, the
   last overlapping offset becomes the cue end.
3. Clamp each move to ``max_shift_s`` — a cue is never dragged further
   than that from where the dubber planned it, so a mis-detected onset
   cannot throw the subtitles out of sync with the video.
4. Hand the aligned spans to :func:`retime_transcript`, which rescales
   the word timings into them and stretches too-short cues to a
   readable minimum without ever overlapping the next cue.

A segment whose window holds no detected speech keeps its planned
timing and is reported as ``no-speech`` instead of being silently
dropped or zeroed — the dub may genuinely be silent there (a pause
the script kept), and the report says so. Wordless segments are
skipped, as everywhere else: they produce no cue anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from video_dubber.audio import AudioData
from video_dubber.subtitles import DEFAULT_MIN_CUE_S, retime_transcript
from video_dubber.transcript import Transcript
from video_dubber.vad import VadConfig, detect_speech

__all__ = [
    "DEFAULT_MAX_SHIFT_S",
    "AlignConfig",
    "AlignError",
    "AlignmentReport",
    "AlignmentRow",
    "align_transcript",
]

DEFAULT_MAX_SHIFT_S = 1.0  # never drag a cue further than this from planned


class AlignError(Exception):
    """Raised when alignment cannot run (bad parameters, empty input)."""


@dataclass(frozen=True)
class AlignConfig:
    """Tuning knobs for :func:`align_transcript`."""

    max_shift_s: float = DEFAULT_MAX_SHIFT_S
    min_cue_s: float = DEFAULT_MIN_CUE_S
    vad: VadConfig | None = None


@dataclass(frozen=True)
class AlignmentRow:
    """One segment's planned timing vs its aligned timing.

    ``shift_ms`` is the cue-start move in milliseconds (positive =
    the cue starts later, the usual case when synthesis opens with
    leading silence). ``status`` is one of ``aligned`` (moved),
    ``unchanged`` (speech already started at the window edge),
    ``clamped`` (the detected move exceeded ``max_shift_s`` and was
    capped), or ``no-speech`` (nothing detected in the window — the
    planned timing was kept).
    """

    index: int  # 1-based segment number in the transcript
    planned_start: float
    planned_end: float
    aligned_start: float
    aligned_end: float
    shift_ms: float
    status: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "planned_start": round(self.planned_start, 3),
            "planned_end": round(self.planned_end, 3),
            "aligned_start": round(self.aligned_start, 3),
            "aligned_end": round(self.aligned_end, 3),
            "shift_ms": round(self.shift_ms, 1),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlignmentRow:
        """Rebuild from :meth:`to_dict` output."""
        try:
            return cls(
                index=int(data["index"]),
                planned_start=float(data["planned_start"]),
                planned_end=float(data["planned_end"]),
                aligned_start=float(data["aligned_start"]),
                aligned_end=float(data["aligned_end"]),
                shift_ms=float(data["shift_ms"]),
                status=str(data["status"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid alignment row: {exc}") from exc


@dataclass
class AlignmentReport:
    """What alignment did to a transcript's cues (JSON + Markdown)."""

    max_shift_s: float
    min_cue_s: float
    segments_aligned: int
    segments_clamped: int
    segments_no_speech: int
    segments_unchanged: int
    mean_abs_shift_ms: float
    max_abs_shift_ms: float
    rows: list[AlignmentRow] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "max_shift_s": self.max_shift_s,
                "min_cue_s": self.min_cue_s,
                "segments_aligned": self.segments_aligned,
                "segments_clamped": self.segments_clamped,
                "segments_no_speech": self.segments_no_speech,
                "segments_unchanged": self.segments_unchanged,
                "mean_abs_shift_ms": round(self.mean_abs_shift_ms, 1),
                "max_abs_shift_ms": round(self.max_abs_shift_ms, 1),
                "segments": [row.to_dict() for row in self.rows],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Subtitle alignment report",
            "",
            f"- Max shift allowed: {self.max_shift_s:.2f}s",
            f"- Min cue length: {self.min_cue_s:.2f}s",
            f"- Mean |start shift|: {self.mean_abs_shift_ms:.0f} ms",
            f"- Max |start shift|: {self.max_abs_shift_ms:.0f} ms",
            f"- Segments aligned: {self.segments_aligned}",
            f"- Segments clamped at max shift: {self.segments_clamped}",
            f"- Segments with no detected speech (kept as planned): "
            f"{self.segments_no_speech}",
            f"- Segments unchanged: {self.segments_unchanged}",
        ]
        for row in self.rows:
            if row.status == "no-speech":
                lines.append(
                    f"- Segment {row.index} ({row.planned_start:.2f}-"
                    f"{row.planned_end:.2f}s): no speech detected, kept as planned"
                )
            else:
                clamped = " (clamped)" if row.status == "clamped" else ""
                lines.append(
                    f"- Segment {row.index} ({row.planned_start:.2f}-"
                    f"{row.planned_end:.2f}s): cue {row.aligned_start:.2f}-"
                    f"{row.aligned_end:.2f}s, start shift "
                    f"{row.shift_ms:+.0f} ms{clamped}"
                )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> AlignmentReport:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("alignment report must be a JSON object")
        try:
            return cls(
                max_shift_s=float(data["max_shift_s"]),
                min_cue_s=float(data["min_cue_s"]),
                segments_aligned=int(data["segments_aligned"]),
                segments_clamped=int(data["segments_clamped"]),
                segments_no_speech=int(data["segments_no_speech"]),
                segments_unchanged=int(data["segments_unchanged"]),
                mean_abs_shift_ms=float(data["mean_abs_shift_ms"]),
                max_abs_shift_ms=float(data["max_abs_shift_ms"]),
                rows=[AlignmentRow.from_dict(row) for row in data.get("segments", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid alignment report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> AlignmentReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def align_transcript(
    audio: AudioData,
    transcript: Transcript,
    config: AlignConfig | None = None,
) -> tuple[Transcript, AlignmentReport]:
    """Shift each cue to the speech detected inside its planned window.

    Returns ``(aligned_transcript, report)``. The aligned transcript
    keeps every segment's words and speaker; only the word timings
    move, rescaled into the aligned span by
    :func:`video_dubber.subtitles.retime_transcript`.
    """
    config = config or AlignConfig()
    if config.max_shift_s < 0:
        raise AlignError(f"max_shift_s must be non-negative, got {config.max_shift_s}")
    if config.min_cue_s < 0:
        raise AlignError(f"min_cue_s must be non-negative, got {config.min_cue_s}")
    timed = [segment for segment in transcript.segments if segment.words]
    if not timed:
        raise AlignError("cannot align a transcript with no timed segments")

    speech = detect_speech(audio, config.vad)

    placed: list[tuple[float, float]] = []
    rows: list[AlignmentRow] = []
    prev_end = 0.0
    for i, segment in enumerate(timed, start=1):
        start, end = segment.start, segment.end
        overlapping = [s for s in speech if s.end > start and s.start < end]
        if not overlapping:
            placed.append((start, end - start))
            rows.append(
                AlignmentRow(
                    index=i,
                    planned_start=start,
                    planned_end=end,
                    aligned_start=start,
                    aligned_end=end,
                    shift_ms=0.0,
                    status="no-speech",
                )
            )
            prev_end = end
            continue

        onset = max(start, overlapping[0].start)
        offset = min(end, overlapping[-1].end)
        new_start, new_end = onset, offset
        clamped = False
        if new_start - start > config.max_shift_s:
            new_start = start + config.max_shift_s
            clamped = True
        if end - new_end > config.max_shift_s:
            new_end = end - config.max_shift_s
            clamped = True
        # Never overlap the previous cue: the detected onset can sit
        # inside the previous cue when the VAD bridges a short gap.
        if new_start < prev_end:
            new_start = prev_end
            clamped = True
        if new_end <= new_start:
            # Detected speech is a sliver (or the clamp ate it): keep
            # the planned timing rather than emitting a zero cue.
            placed.append((start, end - start))
            rows.append(
                AlignmentRow(
                    index=i,
                    planned_start=start,
                    planned_end=end,
                    aligned_start=start,
                    aligned_end=end,
                    shift_ms=0.0,
                    status="unchanged",
                )
            )
            prev_end = end
            continue
        placed.append((new_start, new_end - new_start))
        shift_ms = (new_start - start) * 1000.0
        if clamped:
            status = "clamped"
        elif abs(shift_ms) < 0.5 and abs(new_end - end) < 0.001:
            status = "unchanged"
        else:
            status = "aligned"
        rows.append(
            AlignmentRow(
                index=i,
                planned_start=start,
                planned_end=end,
                aligned_start=new_start,
                aligned_end=new_end,
                shift_ms=shift_ms,
                status=status,
            )
        )
        prev_end = new_end

    aligned = retime_transcript(transcript, placed, min_cue_s=config.min_cue_s)
    # retime_transcript may stretch a too-short cue to the readable
    # minimum; report the final on-screen timings, not the raw spans.
    final_rows: list[AlignmentRow] = []
    timed_aligned = [s for s in aligned.segments if s.words]
    for row, segment in zip(rows, timed_aligned):
        final_rows.append(
            AlignmentRow(
                index=row.index,
                planned_start=row.planned_start,
                planned_end=row.planned_end,
                aligned_start=segment.start,
                aligned_end=segment.end,
                shift_ms=(segment.start - row.planned_start) * 1000.0,
                status=row.status,
            )
        )

    shifts = [abs(row.shift_ms) for row in final_rows]
    report = AlignmentReport(
        max_shift_s=config.max_shift_s,
        min_cue_s=config.min_cue_s,
        segments_aligned=sum(1 for r in final_rows if r.status == "aligned"),
        segments_clamped=sum(1 for r in final_rows if r.status == "clamped"),
        segments_no_speech=sum(1 for r in final_rows if r.status == "no-speech"),
        segments_unchanged=sum(1 for r in final_rows if r.status == "unchanged"),
        mean_abs_shift_ms=sum(shifts) / len(shifts) if shifts else 0.0,
        max_abs_shift_ms=max(shifts) if shifts else 0.0,
        rows=final_rows,
    )
    return aligned, report
