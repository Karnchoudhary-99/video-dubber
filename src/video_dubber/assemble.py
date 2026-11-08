"""Dub assembly: fit synthesised speech to the original timing, mix the
dub track, mux it back to video, and report on quality.

The assembly pipeline for one translated transcript:

1. **Synthesise** each segment in the cloned voice (the caller supplies
   the :class:`~video_dubber.voice.VoiceCloner`; the demo double renders
   formant tones).
2. **Fit** each segment's audio into its original timing window with
   :func:`fit_to_window` (time-stretch; a gentle stretch sounds better
   than cutting words off, and the fit factor is reported honestly).
3. **Loudness-match** every segment to the original clip's RMS so the dub
   does not jump in volume between segments.
4. **Place** the segments on a silence bed at their original start times
   (:func:`assemble_dub`) and, when a video is given, **mux** the new
   audio track back in with ffmpeg (:func:`mux_audio`).
5. **Report** (:func:`build_report`): sync drift, fit factors, loudness
   error, and translation coverage — the numbers a reviewer checks
   before the dub ships.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from video_dubber.audio import AudioData
from video_dubber.extract import ExtractionError, _run
from video_dubber.transcript import Transcript
from video_dubber.voice import SpeakerProfile, VoiceCloner, time_stretch

__all__ = [
    "AssemblyError",
    "DubReport",
    "assemble_dub",
    "build_report",
    "dub_transcript",
    "fit_to_window",
    "loudness_match",
    "mux_audio",
]

MAX_STRETCH = 1.6  # beyond this a dub sounds rushed/dragged; reported, not hidden
MIN_STRETCH = 0.6


class AssemblyError(Exception):
    """Raised when dub assembly or muxing fails."""


def fit_to_window(audio: AudioData, target_duration: float) -> tuple[AudioData, float]:
    """Time-stretch ``audio`` to ``target_duration``; returns (audio, factor).

    ``factor`` is ``target / actual``: > 1 slows the speech down, < 1
    speeds it up. Factors outside :data:`MIN_STRETCH`–:data:`MAX_STRETCH`
    are clamped — a dub that needs more than that has a translation or
    speaking-rate problem, and the report flags it instead of producing
    unintelligible audio. Empty audio returns empty audio with factor 1.
    """
    if target_duration <= 0:
        raise AssemblyError(f"target duration must be positive, got {target_duration}")
    if not audio.samples or audio.duration <= 0:
        return AudioData([], audio.sample_rate), 1.0
    factor = target_duration / audio.duration
    clamped = min(MAX_STRETCH, max(MIN_STRETCH, factor))
    return time_stretch(audio, clamped), clamped


def loudness_match(audio: AudioData, reference: AudioData) -> AudioData:
    """Scale ``audio`` so its RMS matches the reference's RMS.

    Silence stays silent; a silent reference leaves the audio untouched.
    Returns a new :class:`AudioData` (peak-clamped to 0.95 to avoid
    clipping when the reference was louder than the source).
    """
    if not audio.samples:
        return AudioData([], audio.sample_rate)
    ref_rms = math.sqrt(sum(s * s for s in reference.samples) / len(reference.samples)) if reference.samples else 0.0
    if ref_rms <= 0.0:
        return AudioData(list(audio.samples), audio.sample_rate)
    rms = math.sqrt(sum(s * s for s in audio.samples) / len(audio.samples))
    if rms <= 0.0:
        return AudioData(list(audio.samples), audio.sample_rate)
    gain = ref_rms / rms
    scaled = [max(-0.95, min(0.95, s * gain)) for s in audio.samples]
    return AudioData(scaled, audio.sample_rate)


def assemble_dub(
    segments: list[tuple[float, AudioData]],
    total_duration: float,
    sample_rate: int = 16_000,
) -> AudioData:
    """Place ``(start_time, audio)`` clips on a silence bed.

    Each clip starts at its segment's original start time; overlapping
    clips are mixed (added, then clamped). The bed is ``total_duration``
    long. Raises :class:`AssemblyError` on a negative start or a clip
    that runs past the end of the bed.
    """
    if total_duration <= 0:
        raise AssemblyError(f"total duration must be positive, got {total_duration}")
    bed = [0.0] * int(total_duration * sample_rate)
    for start, clip in segments:
        if start < 0:
            raise AssemblyError(f"segment starts before 0: {start}")
        offset = int(start * sample_rate)
        if offset + len(clip.samples) > len(bed):
            raise AssemblyError(
                f"segment at {start:.2f}s overruns the {total_duration:.2f}s bed "
                f"by {(offset + len(clip.samples) - len(bed)) / sample_rate:.2f}s"
            )
        for i, sample in enumerate(clip.samples):
            bed[offset + i] = max(-1.0, min(1.0, bed[offset + i] + sample))
    return AudioData(bed, sample_rate)


def dub_transcript(
    transcript: Transcript,
    cloner: VoiceCloner,
    profile: SpeakerProfile,
    original: AudioData | None = None,
) -> tuple[AudioData, list[float], list[tuple[float, float]]]:
    """Render a full dubbed track for a translated transcript.

    Each segment is synthesised in the cloned voice, stretched to its
    original window, loudness-matched to the original recording when one
    is given, and placed at its original start time. Returns
    ``(dub_audio, fit_factors, placed)`` — the per-segment stretch
    factors and the ``(start, duration)`` of every placed clip, so the
    report can measure sync drift. The dub bed spans the transcript's
    full duration (last word end).
    """
    if not transcript.segments:
        raise AssemblyError("cannot dub an empty transcript")
    total = transcript.segments[-1].end
    if total <= 0:
        raise AssemblyError("transcript has no timed segments to dub")
    placed: list[tuple[float, AudioData]] = []
    fit_factors: list[float] = []
    for segment in transcript.segments:
        if not segment.words:
            continue
        spoken = cloner.synthesize(segment.text, profile)
        fitted, factor = fit_to_window(spoken, segment.duration)
        fit_factors.append(factor)
        if original is not None:
            fitted = loudness_match(fitted, original)
        placed.append((segment.start, fitted))
    placed_info = [(start, clip.duration) for start, clip in placed]
    return assemble_dub(placed, total), fit_factors, placed_info


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """Replace a video's audio track with ``audio_path`` (ffmpeg).

    The video stream is copied untouched; the new audio is encoded as
    AAC and the output is cut to the shorter of the two streams
    (``-shortest``) so a dub that runs long does not freeze the last
    frame. Returns ``out_path``. Raises :class:`AssemblyError` when
    ffmpeg is missing or fails.
    """
    try:
        result = _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                out_path,
            ],
            "mux",
        )
    except ExtractionError as exc:
        raise AssemblyError(str(exc)) from exc
    if result.returncode != 0:
        raise AssemblyError(
            f"ffmpeg mux failed for {video_path}: {result.stderr.strip()[:300]}"
        )
    return out_path


# --- Quality report -----------------------------------------------------------

@dataclass
class DubReport:
    """Quality numbers for one dubbed transcript (JSON + Markdown)."""

    segments_dubbed: int
    total_duration: float
    mean_sync_drift: float
    max_sync_drift: float
    segments_over_drift: int  # segments with drift > 0.10s
    mean_fit_factor: float
    max_fit_factor: float
    extreme_fits: int  # segments with factor outside [0.8, 1.25]
    loudness_error_db: float | None  # dub RMS vs original RMS, None if no original
    translation_coverage: float | None  # dictionary coverage, None if unknown
    verdict: str  # "good" | "review" — review when drift/fit/loudness is off

    def to_json(self) -> str:
        return json.dumps(
            {
                "segments_dubbed": self.segments_dubbed,
                "total_duration": round(self.total_duration, 3),
                "mean_sync_drift_s": round(self.mean_sync_drift, 3),
                "max_sync_drift_s": round(self.max_sync_drift, 3),
                "segments_over_drift": self.segments_over_drift,
                "mean_fit_factor": round(self.mean_fit_factor, 3),
                "max_fit_factor": round(self.max_fit_factor, 3),
                "extreme_fits": self.extreme_fits,
                "loudness_error_db": (
                    None if self.loudness_error_db is None else round(self.loudness_error_db, 2)
                ),
                "translation_coverage": self.translation_coverage,
                "verdict": self.verdict,
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Dub quality report",
            "",
            f"- Segments dubbed: {self.segments_dubbed}",
            f"- Total duration: {self.total_duration:.2f}s",
            f"- Mean sync drift: {self.mean_sync_drift:.3f}s (max {self.max_sync_drift:.3f}s)",
            f"- Segments over 0.10s drift: {self.segments_over_drift}",
            f"- Mean fit factor: {self.mean_fit_factor:.3f} (max {self.max_fit_factor:.3f})",
            f"- Extreme fits (outside 0.8–1.25): {self.extreme_fits}",
        ]
        if self.loudness_error_db is not None:
            lines.append(f"- Loudness error vs original: {self.loudness_error_db:.2f} dB")
        if self.translation_coverage is not None:
            lines.append(f"- Translation coverage: {self.translation_coverage:.0%}")
        lines.append(f"- Verdict: **{self.verdict}**")
        return "\n".join(lines) + "\n"


def build_report(
    transcript: Transcript,
    placed: list[tuple[float, float]],
    fit_factors: list[float],
    dub_audio: AudioData,
    original: AudioData | None = None,
    translation_coverage: float | None = None,
) -> DubReport:
    """Score a dub from the placed clips and the fit factors.

    Sync drift per segment is ``|placed end - segment end|`` — zero when
    the stretch fit the window exactly, positive when the stretch had to
    be clamped. The verdict is ``review`` when max drift exceeds 0.25s,
    any fit is extreme, or loudness is off by more than 3 dB; otherwise
    ``good``.
    """
    timed = [s for s in transcript.segments if s.words]
    drifts = [
        abs((start + duration) - segment.end)
        for segment, (start, duration) in zip(timed, placed)
    ]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    max_drift = max(drifts, default=0.0)
    mean_fit = sum(fit_factors) / len(fit_factors) if fit_factors else 1.0
    max_fit = max(fit_factors, default=1.0)
    extreme = sum(1 for f in fit_factors if f < 0.8 or f > 1.25)
    loudness_error: float | None = None
    if original is not None and original.samples and dub_audio.samples:
        dub_rms = math.sqrt(sum(s * s for s in dub_audio.samples) / len(dub_audio.samples))
        orig_rms = math.sqrt(sum(s * s for s in original.samples) / len(original.samples))
        if dub_rms > 0 and orig_rms > 0:
            loudness_error = 20 * math.log10(dub_rms / orig_rms)
    verdict = "good"
    if (
        max_drift > 0.25
        or extreme > 0
        or (loudness_error is not None and abs(loudness_error) > 3.0)
    ):
        verdict = "review"
    return DubReport(
        segments_dubbed=len(timed),
        total_duration=transcript.segments[-1].end if transcript.segments else 0.0,
        mean_sync_drift=mean_drift,
        max_sync_drift=max_drift,
        segments_over_drift=sum(1 for d in drifts if d > 0.10),
        mean_fit_factor=mean_fit,
        max_fit_factor=max_fit,
        extreme_fits=extreme,
        loudness_error_db=loudness_error,
        translation_coverage=translation_coverage,
        verdict=verdict,
    )
