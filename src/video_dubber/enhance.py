"""Audio enhancement for dubbed tracks — stdlib only.

A synthesized dub voice is rarely broadcast-ready straight out of the
cloner: levels wander between segments, low-frequency hum creeps in from
the source recording, and pauses carry room noise. This module cleans
the finished dub (or any WAV) before it ships:

1. **Measure** (:func:`measure_loudness`): RMS/peak levels in dBFS,
   crest factor and DC offset — the numbers the rest of the pipeline
   reasons about.
2. **Normalize** (:func:`normalize_loudness`): gain to a target RMS
   with a peak ceiling; samples that would exceed the ceiling are
   soft-limited (tanh) instead of hard-clipped, so loud peaks round
   off instead of squaring off.
3. **High-pass** (:func:`high_pass`): one-pole filter that removes
   sub-voice rumble/hum (default 80 Hz) without touching speech.
4. **Noise gate** (:func:`noise_gate`): per-frame RMS envelope with
   attack/release smoothing; frames below the threshold are pushed
   down to a floor instead of cut to zero, so gating never clicks.
5. **Trim** (:func:`trim_silence`): drop leading/trailing silence with
   a small margin, so files start on the first word.
6. **Pipeline** (:func:`enhance_audio`): the stages above in order,
   plus an :class:`EnhancementReport` with before/after numbers.

Everything is pure float math on :class:`AudioData`; no third-party
deps, so it runs anywhere the rest of the pipeline does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from video_dubber.audio import AudioData

__all__ = [
    "EnhanceError",
    "EnhancementReport",
    "LoudnessStats",
    "enhance_audio",
    "high_pass",
    "measure_loudness",
    "noise_gate",
    "normalize_loudness",
    "trim_silence",
]

DEFAULT_TARGET_DBFS = -16.0  # podcast-style loudness target
DEFAULT_CEILING_DBFS = -1.0  # never let a peak past this
DEFAULT_HIGH_PASS_HZ = 80.0
DEFAULT_GATE_THRESHOLD_DBFS = -45.0
DEFAULT_GATE_FLOOR_DB = -60.0


class EnhanceError(ValueError):
    """Raised when enhancement parameters are out of range."""


@dataclass
class LoudnessStats:
    """Level summary of an audio clip, all in dBFS (0 = full scale)."""

    rms_dbfs: float  # -inf for silence
    peak_dbfs: float  # -inf for silence
    crest_db: float  # peak - rms; 0.0 for silence
    dc_offset: float  # mean sample value; ~0 for centred audio
    duration: float

    def to_dict(self) -> dict:
        return {
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "crest_db": self.crest_db,
            "dc_offset": self.dc_offset,
            "duration": self.duration,
        }


def _dbfs(ratio: float) -> float:
    """Linear amplitude ratio -> dBFS (``-inf`` at zero)."""
    if ratio <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(ratio)


def measure_loudness(audio: AudioData) -> LoudnessStats:
    """Measure RMS/peak levels, crest factor and DC offset."""
    samples = audio.samples
    if not samples:
        return LoudnessStats(float("-inf"), float("-inf"), 0.0, 0.0, 0.0)
    mean = sum(samples) / len(samples)
    mean_square = sum((s - mean) ** 2 for s in samples) / len(samples)
    rms = math.sqrt(mean_square)
    peak = max(abs(s) for s in samples)
    return LoudnessStats(
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        crest_db=(_dbfs(peak) - _dbfs(rms)) if rms > 0 else 0.0,
        dc_offset=mean,
        duration=audio.duration,
    )


def normalize_loudness(
    audio: AudioData,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    ceiling_dbfs: float = DEFAULT_CEILING_DBFS,
) -> tuple[AudioData, float]:
    """Gain ``audio`` to ``target_dbfs`` RMS, soft-limiting at the ceiling.

    Returns ``(audio, gain_db)``. Silence is returned unchanged with
    0 dB gain. The ceiling is enforced in two steps: first the gain is
    capped so the *peak* lands exactly on the ceiling, then any sample
    still above the ceiling (only possible when the crest factor is
    huge) is rounded off with a tanh soft clip instead of a hard clip.
    """
    if target_dbfs > 0.0 or ceiling_dbfs > 0.0:
        raise EnhanceError("target and ceiling must be <= 0 dBFS")
    if target_dbfs > ceiling_dbfs:
        raise EnhanceError("target RMS must sit below the peak ceiling")
    stats = measure_loudness(audio)
    if not audio.samples or stats.rms_dbfs == float("-inf"):
        return AudioData(list(audio.samples), audio.sample_rate), 0.0

    gain_db = target_dbfs - stats.rms_dbfs
    # Cap the gain so the peak lands on the ceiling, not past it.
    peak_headroom_db = ceiling_dbfs - stats.peak_dbfs
    gain_db = min(gain_db, peak_headroom_db)
    gain = 10.0 ** (gain_db / 20.0)
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)

    out: list[float] = []
    for sample in audio.samples:
        value = sample * gain
        if abs(value) > ceiling:
            # tanh soft clip: continuous, monotonic, rounds the peak.
            value = math.tanh(value / ceiling) * ceiling
        out.append(value)
    return AudioData(out, audio.sample_rate), gain_db
