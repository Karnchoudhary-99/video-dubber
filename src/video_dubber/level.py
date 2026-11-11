"""Per-segment loudness levelling for dubbed tracks.

A dub is assembled from separately synthesised segments, and even
though :func:`video_dubber.assemble.loudness_match` matches each
segment to the original recording, the *dubbed* segments can still
jump in volume against each other — one line whispered, the next
shouted — because the original's own levels vary. This module evens
them out after the fact, on the finished track:

1. **Measure** each timed segment's RMS/peak (silence stays silent).
2. **Gain** each segment toward a common target RMS (default -16 dBFS,
   the same target as :func:`video_dubber.enhance.normalize_loudness`),
   clamped to ``±max_gain_db`` and to the peak ceiling — a segment
   that would need more gain than the cap allows is reported as
   ``at_cap`` instead of being pushed into clipping.
3. **Ramp** the gain over 5 ms at each segment edge so the level
   changes never click, and soft-clip the result at the ceiling
   (tanh, as in :mod:`video_dubber.enhance`) as a final safety net.

Levelling never moves timing: a segment keeps its start, end and
duration, so a levelled dub still lines up with its subtitles and
its quality report. Wordless segments are skipped (nothing to
level); a segment whose window holds only silence is reported, not
gained.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from video_dubber.audio import AudioData
from video_dubber.transcript import Transcript

__all__ = [
    "DEFAULT_CEILING_DBFS",
    "DEFAULT_MAX_GAIN_DB",
    "DEFAULT_TARGET_DBFS",
    "LevelError",
    "LevelReport",
    "SegmentLevel",
    "level_segments",
    "measure_segment_levels",
]

DEFAULT_TARGET_DBFS = -16.0
DEFAULT_MAX_GAIN_DB = 12.0
DEFAULT_CEILING_DBFS = -1.0
_FADE_MS = 5.0  # gain ramp at each segment edge, so levels never click


class LevelError(Exception):
    """Raised when levelling cannot run (bad parameters, empty input)."""


def _dbfs(value: float) -> float:
    """Linear amplitude -> dBFS (``-inf`` at zero)."""
    if value <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(value)


@dataclass(frozen=True)
class SegmentLevel:
    """One segment's measured level and the gain applied to it.

    ``gain_db`` is 0 for silent segments (nothing to level) and for
    segments already at the target. ``at_cap`` means the gain was
    clamped — by ``max_gain_db`` or by the peak ceiling — so the
    segment may still sit off-target; the report counts these.
    """

    index: int  # 1-based segment number in the transcript
    start: float
    end: float
    rms_dbfs: float
    peak_dbfs: float
    gain_db: float = 0.0
    at_cap: bool = False
    silent: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "rms_dbfs": None if self.rms_dbfs == float("-inf") else round(self.rms_dbfs, 2),
            "peak_dbfs": None if self.peak_dbfs == float("-inf") else round(self.peak_dbfs, 2),
            "gain_db": round(self.gain_db, 2),
            "at_cap": self.at_cap,
            "silent": self.silent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SegmentLevel:
        """Rebuild from :meth:`to_dict` output."""
        try:
            rms = data["rms_dbfs"]
            peak = data["peak_dbfs"]
            return cls(
                index=int(data["index"]),
                start=float(data["start"]),
                end=float(data["end"]),
                rms_dbfs=float("-inf") if rms is None else float(rms),
                peak_dbfs=float("-inf") if peak is None else float(peak),
                gain_db=float(data.get("gain_db", 0.0)),
                at_cap=bool(data.get("at_cap", False)),
                silent=bool(data.get("silent", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid segment level: {exc}") from exc


@dataclass
class LevelReport:
    """What levelling did to a track (JSON + Markdown)."""

    target_dbfs: float
    max_deviation_before_db: float  # worst |segment RMS - target| before
    max_deviation_after_db: float  # same, measured on the levelled track
    segments_levelled: int
    segments_at_cap: int
    segments_silent: int
    levels: list[SegmentLevel] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "target_dbfs": self.target_dbfs,
                "max_deviation_before_db": round(self.max_deviation_before_db, 2),
                "max_deviation_after_db": round(self.max_deviation_after_db, 2),
                "segments_levelled": self.segments_levelled,
                "segments_at_cap": self.segments_at_cap,
                "segments_silent": self.segments_silent,
                "segments": [level.to_dict() for level in self.levels],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Segment levelling report",
            "",
            f"- Target RMS: {self.target_dbfs:.1f} dBFS",
            f"- Max deviation from target: {self.max_deviation_before_db:.2f} dB -> "
            f"{self.max_deviation_after_db:.2f} dB",
            f"- Segments levelled: {self.segments_levelled}",
            f"- Segments at the gain cap: {self.segments_at_cap}",
            f"- Silent segments (untouched): {self.segments_silent}",
        ]
        for level in self.levels:
            if level.silent:
                lines.append(
                    f"- Segment {level.index} ({level.start:.2f}-{level.end:.2f}s): silent"
                )
            else:
                cap = " (at cap)" if level.at_cap else ""
                lines.append(
                    f"- Segment {level.index} ({level.start:.2f}-{level.end:.2f}s): "
                    f"RMS {level.rms_dbfs:.1f} dBFS, gain {level.gain_db:+.1f} dB{cap}"
                )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> LevelReport:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("level report must be a JSON object")
        try:
            return cls(
                target_dbfs=float(data["target_dbfs"]),
                max_deviation_before_db=float(data["max_deviation_before_db"]),
                max_deviation_after_db=float(data["max_deviation_after_db"]),
                segments_levelled=int(data["segments_levelled"]),
                segments_at_cap=int(data["segments_at_cap"]),
                segments_silent=int(data["segments_silent"]),
                levels=[SegmentLevel.from_dict(row) for row in data.get("segments", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid level report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> LevelReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def _slice_stats(samples: list[float], start: int, end: int) -> tuple[float, float]:
    """RMS and peak dBFS of ``samples[start:end]`` (-inf when silent)."""
    window = samples[start:end]
    if not window:
        return float("-inf"), float("-inf")
    rms = math.sqrt(sum(s * s for s in window) / len(window))
    peak = max(abs(s) for s in window)
    return _dbfs(rms), _dbfs(peak)


def measure_segment_levels(audio: AudioData, transcript: Transcript) -> list[SegmentLevel]:
    """Measure every timed segment's RMS/peak (no gain applied).

    Wordless segments are skipped. A segment that starts at or past
    the end of the audio raises :class:`LevelError` — that is a caller
    bug (the transcript does not match the track); a segment that
    merely runs past the end is measured on the samples that exist.
    """
    timed = [s for s in transcript.segments if s.words]
    if not timed:
        raise LevelError("cannot level a transcript with no timed segments")
    levels: list[SegmentLevel] = []
    for i, segment in enumerate(timed, start=1):
        start_sample = int(segment.start * audio.sample_rate)
        end_sample = int(segment.end * audio.sample_rate)
        if start_sample >= len(audio.samples):
            raise LevelError(
                f"segment {i} starts at {segment.start:.2f}s, past the "
                f"{audio.duration:.2f}s audio"
            )
        end_sample = min(end_sample, len(audio.samples))
        rms_dbfs, peak_dbfs = _slice_stats(audio.samples, start_sample, end_sample)
        levels.append(
            SegmentLevel(
                index=i,
                start=segment.start,
                end=segment.end,
                rms_dbfs=rms_dbfs,
                peak_dbfs=peak_dbfs,
                silent=rms_dbfs == float("-inf"),
            )
        )
    return levels


def level_segments(
    audio: AudioData,
    transcript: Transcript,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    max_gain_db: float = DEFAULT_MAX_GAIN_DB,
    ceiling_dbfs: float = DEFAULT_CEILING_DBFS,
) -> tuple[AudioData, LevelReport]:
    """Gain each segment toward ``target_dbfs`` RMS; returns (audio, report).

    The gain for a segment is ``target - rms``, clamped to
    ``±max_gain_db`` and further capped so the segment's peak lands
    on the ceiling at most. The gain ramps in/out over 5 ms at the
    segment edges, and the whole track is soft-clipped at the ceiling
    (tanh) as a final safety net — overlapping windows can stack
    gains, so the ceiling is enforced on the output, not assumed.
    Silence is untouched, timing is untouched, and the report records
    the before/after deviation honestly: a segment that hit the cap
    still shows up off-target.
    """
    if target_dbfs > 0.0 or ceiling_dbfs > 0.0:
        raise LevelError("target and ceiling must be <= 0 dBFS")
    if target_dbfs > ceiling_dbfs:
        raise LevelError("target RMS must sit below the peak ceiling")
    if max_gain_db <= 0.0:
        raise LevelError(f"max gain must be positive, got {max_gain_db}")
    levels = measure_segment_levels(audio, transcript)

    sr = audio.sample_rate
    fade = max(1, int(sr * _FADE_MS / 1000.0))
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    envelope = [1.0] * len(audio.samples)
    applied: list[SegmentLevel] = []
    for level in levels:
        if level.silent:
            applied.append(level)
            continue
        gain_db = target_dbfs - level.rms_dbfs
        capped = False
        if abs(gain_db) > max_gain_db:
            gain_db = math.copysign(max_gain_db, gain_db)
            capped = True
        # Cap the gain so the segment's peak lands on the ceiling.
        peak_headroom_db = ceiling_dbfs - level.peak_dbfs
        if gain_db > peak_headroom_db:
            gain_db = peak_headroom_db
            capped = True
        applied.append(
            SegmentLevel(
                index=level.index,
                start=level.start,
                end=level.end,
                rms_dbfs=level.rms_dbfs,
                peak_dbfs=level.peak_dbfs,
                gain_db=gain_db,
                at_cap=capped,
            )
        )
        gain = 10.0 ** (gain_db / 20.0)
        start_sample = int(level.start * sr)
        end_sample = min(int(level.end * sr), len(audio.samples))
        for n in range(start_sample, end_sample):
            # Ramp the gain in over the first 5 ms and back out over
            # the last 5 ms; a segment shorter than 10 ms gets a
            # triangular envelope instead of a click.
            pos = n - start_sample
            tail = end_sample - 1 - n
            ramp = min(1.0, pos / fade, tail / fade)
            envelope[n] = 1.0 + (gain - 1.0) * ramp

    out: list[float] = []
    for sample, gain in zip(audio.samples, envelope):
        value = sample * gain
        if abs(value) > ceiling:
            value = math.tanh(value / ceiling) * ceiling
        out.append(value)
    levelled = AudioData(out, sr)

    def _max_deviation(levels_before: list[SegmentLevel], track: AudioData) -> float:
        after = measure_segment_levels(track, transcript)
        deviations = [
            abs(a.rms_dbfs - target_dbfs)
            for a, b in zip(levels_before, after)
            if not a.silent and b.rms_dbfs != float("-inf")
        ]
        return max(deviations, default=0.0)

    before = max(
        (abs(level.rms_dbfs - target_dbfs) for level in levels if not level.silent),
        default=0.0,
    )
    report = LevelReport(
        target_dbfs=target_dbfs,
        max_deviation_before_db=before,
        max_deviation_after_db=_max_deviation(levels, levelled),
        segments_levelled=sum(1 for level in applied if not level.silent),
        segments_at_cap=sum(1 for level in applied if level.at_cap),
        segments_silent=sum(1 for level in applied if level.silent),
        levels=applied,
    )
    return levelled, report
