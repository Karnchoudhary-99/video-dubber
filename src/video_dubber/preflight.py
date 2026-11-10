"""Pre-flight checks for source audio, before a dub is attempted.

Dubbing a clipped, silent, or DC-offset recording wastes a whole pipeline
run and produces a bad dub. :func:`run_preflight` measures the audio that
is about to be dubbed and reports one check per failure mode, each with a
plain-language detail so the fix is obvious:

* ``duration`` — empty or very short audio
* ``sample_rate`` — anything but 16 kHz is resampled (a warning, not a failure)
* ``clipping`` — samples pinned at full scale (distortion the dub inherits)
* ``dc_offset`` — a constant bias that eats headroom and clicks on edits
* ``level`` — silent or very quiet recordings
* ``speech_coverage`` — how much of the file the VAD hears as speech
* ``snr`` — speech RMS against the silence between segments
* ``edge_silence`` — long dead air at the start or end (trimmable)

The verdict is ``pass`` when every check passes, ``pass_with_warnings``
when nothing failed but something is worth fixing, and ``fail`` when any
check failed. The ``dub`` command's ``--preflight`` flag aborts on a
``fail`` verdict before any synthesis happens.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from video_dubber.audio import DEFAULT_SAMPLE_RATE, AudioData
from video_dubber.vad import VadConfig, detect_speech, speech_ratio

__all__ = [
    "PreflightCheck",
    "PreflightConfig",
    "PreflightReport",
    "run_preflight",
]


@dataclass
class PreflightConfig:
    """Thresholds for the pre-flight checks (all tunable by callers)."""

    min_duration_s: float = 1.0
    # Samples with |value| >= this count as clipped.
    clip_threshold: float = 0.999
    # Fail above this fraction of clipped samples, warn above the lower one.
    max_clipping_ratio: float = 0.001
    warn_clipping_ratio: float = 0.0001
    # DC offset (mean sample value): warn above, fail above.
    warn_dc_offset: float = 0.01
    max_dc_offset: float = 0.05
    # Peak level: warn below this, fail at silence (peak == 0).
    min_peak: float = 0.10
    # Speech coverage: warn below this fraction, fail at none.
    min_speech_ratio: float = 0.20
    # SNR in dB: warn below, fail below.
    min_snr_db: float = 10.0
    fail_snr_db: float = 5.0
    # Leading/trailing silence worth trimming.
    max_edge_silence_s: float = 2.0


@dataclass(frozen=True)
class PreflightCheck:
    """One pre-flight check: a name, a status, and what it found."""

    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    value: float | None = None  # the measured number, when there is one


@dataclass
class PreflightReport:
    """The outcome of :func:`run_preflight` for one audio file."""

    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """``pass`` | ``pass_with_warnings`` | ``fail``."""
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "warn" for c in self.checks):
            return "pass_with_warnings"
        return "pass"

    @property
    def failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "warn"]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "value": c.value,
                }
                for c in self.checks
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> PreflightReport:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("preflight report must be a JSON object")
        try:
            return cls(
                checks=[
                    PreflightCheck(
                        name=str(item["name"]),
                        status=str(item["status"]),
                        detail=str(item["detail"]),
                        value=item.get("value"),
                    )
                    for item in data["checks"]
                ]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid preflight report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> PreflightReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

    def to_markdown(self) -> str:
        """One line per check plus a verdict line, for the CLI and demos."""
        lines = [f"# Pre-flight report — {self.verdict}", ""]
        for check in self.checks:
            mark = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}[check.status]
            value = "" if check.value is None else f" ({check.value:.3g})"
            lines.append(f"- [{mark}] {check.name}{value}: {check.detail}")
        return "\n".join(lines) + "\n"


def _check_speech_coverage(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    ratio = speech_ratio(audio)
    if ratio <= 0.0:
        return PreflightCheck(
            "speech_coverage",
            "fail",
            "no speech detected — check this is a voice recording, not music or silence",
            ratio,
        )
    if ratio < config.min_speech_ratio:
        return PreflightCheck(
            "speech_coverage",
            "warn",
            f"only {ratio:.0%} of the file is speech — mostly dead air or noise",
            ratio,
        )
    return PreflightCheck("speech_coverage", "pass", f"{ratio:.0%} of the file is speech", ratio)


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _check_snr(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    """Speech RMS against the RMS of the gaps between speech segments.

    A recording with no silence to measure against (wall-to-wall speech)
    passes — there is nothing to compare with, and that is fine.
    """
    segments = detect_speech(audio)
    if not segments:
        return PreflightCheck("snr", "warn", "cannot estimate — no speech detected", None)
    rate = audio.sample_rate
    speech_samples: list[float] = []
    gap_samples: list[float] = []
    cursor = 0
    for seg in segments:
        start = int(seg.start * rate)
        end = int(seg.end * rate)
        gap_samples.extend(audio.samples[cursor:start])
        speech_samples.extend(audio.samples[start:end])
        cursor = end
    gap_samples.extend(audio.samples[cursor:])
    speech_rms = _rms(speech_samples)
    gap_rms = _rms(gap_samples)
    if gap_rms <= 0.0:
        return PreflightCheck("snr", "pass", "no silence between segments to measure against", None)
    snr_db = 20 * math.log10(speech_rms / gap_rms) if speech_rms > 0.0 else float("-inf")
    if snr_db < config.fail_snr_db:
        return PreflightCheck(
            "snr",
            "fail",
            f"speech is only {snr_db:.1f} dB above the room noise — transcription will struggle",
            snr_db,
        )
    if snr_db < config.min_snr_db:
        return PreflightCheck(
            "snr",
            "warn",
            f"speech is {snr_db:.1f} dB above the room noise — consider a quieter room",
            snr_db,
        )
    return PreflightCheck("snr", "pass", f"speech {snr_db:.1f} dB above the room noise", snr_db)


def _check_duration(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    duration = audio.duration
    if duration <= 0.0:
        return PreflightCheck("duration", "fail", "audio is empty", duration)
    if duration < config.min_duration_s:
        return PreflightCheck(
            "duration",
            "warn",
            f"only {duration:.2f}s long — shorter than {config.min_duration_s:.0f}s",
            duration,
        )
    return PreflightCheck("duration", "pass", f"{duration:.2f}s", duration)


def _check_clipping(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    if not audio.samples:
        return PreflightCheck("clipping", "fail", "no samples to measure", 0.0)
    clipped = sum(1 for s in audio.samples if abs(s) >= config.clip_threshold)
    ratio = clipped / len(audio.samples)
    if ratio > config.max_clipping_ratio:
        return PreflightCheck(
            "clipping",
            "fail",
            f"{clipped} samples ({ratio:.2%}) pinned at full scale — distortion the dub inherits",
            ratio,
        )
    if ratio > config.warn_clipping_ratio:
        return PreflightCheck(
            "clipping",
            "warn",
            f"{clipped} samples ({ratio:.2%}) pinned at full scale",
            ratio,
        )
    return PreflightCheck("clipping", "pass", f"{clipped} clipped samples", ratio)


def _check_dc_offset(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    if not audio.samples:
        return PreflightCheck("dc_offset", "fail", "no samples to measure", 0.0)
    mean = sum(audio.samples) / len(audio.samples)
    offset = abs(mean)
    if offset > config.max_dc_offset:
        return PreflightCheck(
            "dc_offset",
            "fail",
            f"mean sample {mean:+.3f} — a bias this large eats headroom and clicks on edits",
            offset,
        )
    if offset > config.warn_dc_offset:
        return PreflightCheck(
            "dc_offset", "warn", f"mean sample {mean:+.3f} — consider removing the DC bias", offset
        )
    return PreflightCheck("dc_offset", "pass", f"mean sample {mean:+.4f}", offset)


def _check_level(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    peak = audio.peak
    if peak <= 0.0:
        return PreflightCheck("level", "fail", "silence — peak level is 0", peak)
    if peak < config.min_peak:
        return PreflightCheck(
            "level",
            "warn",
            f"peak {peak:.3f} is very quiet — the dub will need a large gain",
            peak,
        )
    return PreflightCheck("level", "pass", f"peak {peak:.3f}", peak)


def _check_sample_rate(audio: AudioData, config: PreflightConfig) -> PreflightCheck:
    del config  # the threshold is the pipeline's fixed rate, not tunable
    if audio.sample_rate == DEFAULT_SAMPLE_RATE:
        return PreflightCheck(
            "sample_rate", "pass", f"{audio.sample_rate} Hz", float(audio.sample_rate)
        )
    return PreflightCheck(
        "sample_rate",
        "warn",
        f"{audio.sample_rate} Hz — will be resampled to {DEFAULT_SAMPLE_RATE} Hz",
        float(audio.sample_rate),
    )
