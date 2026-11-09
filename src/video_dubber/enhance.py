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


def high_pass(audio: AudioData, cutoff_hz: float = DEFAULT_HIGH_PASS_HZ) -> AudioData:
    """One-pole high-pass filter — removes rumble/hum below ``cutoff_hz``.

    ``y[n] = a * (y[n-1] + x[n] - x[n-1])`` with ``a`` from the RC time
    constant, run through two cascaded stages for a 12 dB/octave
    rolloff. A 440 Hz tone passes essentially untouched while a 50 Hz
    hum is cut by ~11 dB at the default 80 Hz cutoff.
    """
    if cutoff_hz <= 0.0:
        raise EnhanceError(f"high-pass cutoff must be positive, got {cutoff_hz}")
    if cutoff_hz >= audio.sample_rate / 2:
        raise EnhanceError("high-pass cutoff must be below the Nyquist frequency")
    if not audio.samples:
        return AudioData([], audio.sample_rate)
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / audio.sample_rate
    alpha = rc / (rc + dt)

    def _one_pole(samples: list[float]) -> list[float]:
        out: list[float] = []
        prev_in = 0.0
        prev_out = 0.0
        for sample in samples:
            value = alpha * (prev_out + sample - prev_in)
            out.append(value)
            prev_in = sample
            prev_out = value
        return out

    return AudioData(_one_pole(_one_pole(audio.samples)), audio.sample_rate)


def noise_gate(
    audio: AudioData,
    threshold_dbfs: float = DEFAULT_GATE_THRESHOLD_DBFS,
    floor_db: float = DEFAULT_GATE_FLOOR_DB,
    attack_ms: float = 5.0,
    release_ms: float = 100.0,
) -> AudioData:
    """Attenuate quiet frames to ``floor_db`` instead of cutting them.

    Frames are 10 ms; a frame's RMS decides open vs closed, and the
    gain envelope moves with attack/release smoothing so the gate never
    clicks. The floor (default -60 dBFS) keeps gated audio present but
    inaudible rather than digitally silent, which sounds more natural
    under a dub.
    """
    if threshold_dbfs > 0.0 or floor_db > 0.0:
        raise EnhanceError("gate threshold and floor must be <= 0 dBFS")
    if floor_db > threshold_dbfs:
        raise EnhanceError("gate floor must sit below the threshold")
    if attack_ms <= 0.0 or release_ms <= 0.0:
        raise EnhanceError("attack and release must be positive")
    if not audio.samples:
        return AudioData([], audio.sample_rate)

    sr = audio.sample_rate
    frame = max(1, int(sr * 0.010))  # 10 ms frames
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    floor_gain = 10.0 ** (floor_db / 20.0)
    # Per-sample smoothing: attack fast, release slow.
    attack_coef = 1.0 - math.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coef = 1.0 - math.exp(-1.0 / (sr * release_ms / 1000.0))

    out: list[float] = []
    gain = 1.0
    for start in range(0, len(audio.samples), frame):
        chunk = audio.samples[start : start + frame]
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        target = 1.0 if rms >= threshold else floor_gain
        coef = attack_coef if target > gain else release_coef
        for sample in chunk:
            gain += (target - gain) * coef
            out.append(sample * gain)
    return AudioData(out, sr)


def trim_silence(
    audio: AudioData,
    threshold_dbfs: float = DEFAULT_GATE_THRESHOLD_DBFS,
    margin_s: float = 0.05,
) -> tuple[AudioData, float]:
    """Drop leading/trailing silence, keeping a ``margin_s`` cushion.

    Returns ``(audio, trimmed_seconds)``. A frame counts as silence
    when its 10 ms RMS sits below the threshold. Fully silent audio
    returns empty audio and the full duration as trimmed.
    """
    if threshold_dbfs > 0.0:
        raise EnhanceError("trim threshold must be <= 0 dBFS")
    if margin_s < 0.0:
        raise EnhanceError("trim margin must be >= 0")
    if not audio.samples:
        return AudioData([], audio.sample_rate), 0.0

    sr = audio.sample_rate
    frame = max(1, int(sr * 0.010))
    threshold = 10.0 ** (threshold_dbfs / 20.0)

    def _frame_is_silent(start: int) -> bool:
        chunk = audio.samples[start : start + frame]
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        return rms < threshold

    first = 0
    while first < len(audio.samples) and _frame_is_silent(first):
        first += frame
    last = len(audio.samples)
    while last > first and _frame_is_silent(max(first, last - frame)):
        last -= frame

    if first >= last:
        # No speech at all — the whole clip is silence.
        return AudioData([], audio.sample_rate), audio.duration

    margin = int(margin_s * sr)
    lo = max(0, first - margin)
    hi = min(len(audio.samples), last + margin)
    trimmed = audio.duration - (hi - lo) / sr
    return AudioData(audio.samples[lo:hi], sr), max(0.0, trimmed)


@dataclass
class EnhancementReport:
    """Before/after numbers for one :func:`enhance_audio` run."""

    before: LoudnessStats
    after: LoudnessStats
    gain_db: float
    trimmed_s: float
    high_pass_hz: float
    gate_threshold_dbfs: float

    def to_dict(self) -> dict:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "gain_db": self.gain_db,
            "trimmed_s": self.trimmed_s,
            "high_pass_hz": self.high_pass_hz,
            "gate_threshold_dbfs": self.gate_threshold_dbfs,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Audio enhancement report",
            "",
            f"- RMS: {self.before.rms_dbfs:.1f} dBFS -> {self.after.rms_dbfs:.1f} dBFS",
            f"- Peak: {self.before.peak_dbfs:.1f} dBFS -> {self.after.peak_dbfs:.1f} dBFS",
            f"- Gain applied: {self.gain_db:+.1f} dB",
            f"- Silence trimmed: {self.trimmed_s:.2f} s",
            f"- High-pass: {self.high_pass_hz:.0f} Hz, gate: {self.gate_threshold_dbfs:.0f} dBFS",
        ]
        return "\n".join(lines) + "\n"


def enhance_audio(
    audio: AudioData,
    target_dbfs: float = DEFAULT_TARGET_DBFS,
    ceiling_dbfs: float = DEFAULT_CEILING_DBFS,
    high_pass_hz: float = DEFAULT_HIGH_PASS_HZ,
    gate_threshold_dbfs: float = DEFAULT_GATE_THRESHOLD_DBFS,
    trim: bool = True,
) -> tuple[AudioData, EnhancementReport]:
    """Run the full enhancement chain: trim -> high-pass -> gate -> normalize.

    Returns ``(audio, report)``. ``trim=False`` keeps the duration
    untouched — use that when the audio sits on a timed bed (e.g. a
    dub track whose segments must stay at their original start times),
    where trimming would shift everything after the first word.
    """
    before = measure_loudness(audio)
    trimmed_s = 0.0
    work = audio
    if trim:
        work, trimmed_s = trim_silence(work)
    if high_pass_hz > 0.0:
        work = high_pass(work, high_pass_hz)
    work = noise_gate(work, gate_threshold_dbfs)
    work, gain_db = normalize_loudness(work, target_dbfs, ceiling_dbfs)
    after = measure_loudness(work)
    report = EnhancementReport(
        before=before,
        after=after,
        gain_db=gain_db,
        trimmed_s=trimmed_s,
        high_pass_hz=high_pass_hz,
        gate_threshold_dbfs=gate_threshold_dbfs,
    )
    return work, report
