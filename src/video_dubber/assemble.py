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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from video_dubber.audio import AudioData, concat, load_wav, make_silence, save_wav
from video_dubber.extract import ExtractionError, _run
from video_dubber.transcript import Transcript
from video_dubber.voice import VoiceCloner, time_stretch

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
