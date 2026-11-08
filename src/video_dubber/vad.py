"""Energy-based voice activity detection (VAD).

Splits audio into speech segments using per-frame RMS energy against an
adaptive threshold derived from the recording's own noise floor. This is a
lightweight, dependency-free baseline — good enough to find where people are
talking so the transcriber and the dub assembler can work segment by
segment. It is not a neural VAD and will treat loud non-speech (music,
applause) as speech; that limitation is documented, not hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from video_dubber.audio import AudioData


@dataclass(frozen=True)
class SpeechSegment:
    """One contiguous stretch of detected speech, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VadConfig:
    """Tuning knobs for :func:`detect_speech`."""

    frame_ms: float = 30.0
    # Energy threshold. None = adaptive: noise floor x factor + floor.
    energy_threshold: float | None = None
    threshold_factor: float = 4.0
    energy_floor: float = 1e-4
    # Speech shorter than this is dropped (clicks, pops).
    min_speech_ms: float = 120.0
    # Silence gaps shorter than this are bridged (pauses inside a sentence).
    max_silence_gap_ms: float = 300.0
    # Frames of speech kept before/after the detected edges (context).
    hangover_ms: float = 60.0


def _frame_energies(audio: AudioData, frame_ms: float) -> list[float]:
    frame_len = max(1, int(audio.sample_rate * frame_ms / 1000.0))
    energies: list[float] = []
    for i in range(0, len(audio.samples), frame_len):
        frame = audio.samples[i : i + frame_len]
        if not frame:
            continue
        energies.append(sum(s * s for s in frame) / len(frame))
    return energies


def _adaptive_threshold(energies: list[float], config: VadConfig) -> float:
    if config.energy_threshold is not None:
        return config.energy_threshold
    if not energies:
        return config.energy_floor
    ordered = sorted(energies)
    # 10th percentile ≈ the recording's noise floor (assumes some silence).
    noise = ordered[max(0, int(0.1 * len(ordered)) - 1)]
    return max(noise * config.threshold_factor, config.energy_floor)


def detect_speech(audio: AudioData, config: VadConfig | None = None) -> list[SpeechSegment]:
    """Return speech segments for ``audio`` (seconds, sorted, non-overlapping).

    Empty or silent audio returns an empty list. Segment edges are snapped
    to frame boundaries and extended by ``hangover_ms`` of context.
    """
    config = config or VadConfig()
    if not audio.samples:
        return []
    frame_s = config.frame_ms / 1000.0
    energies = _frame_energies(audio, config.frame_ms)
    threshold = _adaptive_threshold(energies, config)
    is_speech = [e >= threshold for e in energies]

    # Raw runs of speech frames.
    runs: list[list[int]] = []
    for i, active in enumerate(is_speech):
        if active and runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        elif active:
            runs.append([i])

    min_frames = max(1, math.ceil(config.min_speech_ms / config.frame_ms))
    gap_frames = max(0, int(config.max_silence_gap_ms / config.frame_ms))
    hang_frames = max(0, round(config.hangover_ms / config.frame_ms))

    # Drop short runs, then merge runs separated by a short silence gap.
    runs = [r for r in runs if len(r) >= min_frames]
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][-1] - 1 <= gap_frames:
            merged[-1].extend(range(merged[-1][-1] + 1, run[-1] + 1))
        else:
            merged.append(list(run))

    segments: list[SpeechSegment] = []
    for run in merged:
        start = max(0.0, (run[0] - hang_frames) * frame_s)
        end = min(audio.duration, (run[-1] + 1 + hang_frames) * frame_s)
        if end > start:
            segments.append(SpeechSegment(start, end))
    return segments


def speech_ratio(audio: AudioData, config: VadConfig | None = None) -> float:
    """Fraction of the audio that is speech (0.0–1.0)."""
    if audio.duration <= 0:
        return 0.0
    spoken = sum(s.duration for s in detect_speech(audio, config))
    return min(1.0, spoken / audio.duration)
