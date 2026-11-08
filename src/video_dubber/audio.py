"""Audio loading, saving and resampling — stdlib only (``wave`` + ``struct``).

The pipeline works on 16 kHz mono float audio internally. Files on disk may
be any PCM WAV (8/16/24/32-bit, mono or stereo); video files go through
:mod:`video_dubber.extract` (ffmpeg) first and arrive here as WAV.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field

DEFAULT_SAMPLE_RATE = 16_000


@dataclass
class AudioData:
    """In-memory mono audio: float samples in ``[-1, 1]`` plus a sample rate."""

    samples: list[float] = field(default_factory=list)
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def duration(self) -> float:
        """Length in seconds."""
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    @property
    def peak(self) -> float:
        """Maximum absolute sample value (0.0 for empty audio)."""
        return max((abs(s) for s in self.samples), default=0.0)

    def slice(self, start: float, end: float) -> AudioData:
        """Return the audio between ``start`` and ``end`` (seconds, clamped)."""
        start = max(0.0, start)
        end = min(self.duration, max(start, end))
        lo = int(start * self.sample_rate)
        hi = int(end * self.sample_rate)
        return AudioData(self.samples[lo:hi], self.sample_rate)

    def normalized(self, target_peak: float = 0.9) -> AudioData:
        """Return a peak-normalized copy (silence stays silent)."""
        peak = self.peak
        if peak <= 0.0 or target_peak <= 0.0:
            return AudioData(list(self.samples), self.sample_rate)
        gain = target_peak / peak
        return AudioData([s * gain for s in self.samples], self.sample_rate)


def _decode_pcm(raw: bytes, width: int, channels: int) -> list[float]:
    """Decode interleaved PCM bytes to mono float samples in ``[-1, 1]``."""
    if width == 1:  # unsigned 8-bit
        frames = [(b - 128) / 128.0 for b in raw]
    elif width == 2:
        frames = [v / 32767.0 for v in struct.unpack(f"<{len(raw) // 2}h", raw)]
    elif width == 3:  # 24-bit, little-endian, sign-extended by hand
        frames = []
        for i in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[i : i + 3], "little", signed=True)
            frames.append(value / 8388608.0)
    elif width == 4:
        frames = [v / 2147483648.0 for v in struct.unpack(f"<{len(raw) // 4}i", raw)]
    else:  # pragma: no cover - wave module only reports widths 1..4
        raise ValueError(f"unsupported sample width: {width}")

    if channels == 1:
        return frames
    # Mix down to mono by averaging channels.
    return [
        sum(frames[i * channels : (i + 1) * channels]) / channels
        for i in range(len(frames) // channels)
    ]


def load_wav(path: str) -> AudioData:
    """Load a PCM WAV file as mono float audio.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a readable PCM WAV.
    """
    try:
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a readable WAV file: {path}") from exc
    if width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported sample width {width} in {path}")
    return AudioData(_decode_pcm(frames, width, channels), rate)


def save_wav(path: str, audio: AudioData) -> None:
    """Write mono float audio as 16-bit PCM WAV."""
    clipped = [max(-32768, min(32767, round(s * 32767))) for s in audio.samples]
    raw = struct.pack(f"<{len(clipped)}h", *clipped)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(raw)


def resample(audio: AudioData, target_rate: int) -> AudioData:
    """Resample to ``target_rate`` with linear interpolation (identity if equal)."""
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if target_rate == audio.sample_rate or not audio.samples:
        return AudioData(list(audio.samples), target_rate)
    ratio = audio.sample_rate / target_rate
    count = max(1, round(len(audio.samples) / ratio))
    out: list[float] = []
    for i in range(count):
        pos = i * ratio
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, len(audio.samples) - 1)
        out.append(audio.samples[lo] * (1.0 - frac) + audio.samples[hi] * frac)
    return AudioData(out, target_rate)


def make_sine(
    freq: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> AudioData:
    """Generate a pure sine tone — handy for tests and demos."""
    count = int(duration * sample_rate)
    return AudioData(
        [amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate) for i in range(count)],
        sample_rate,
    )


def make_silence(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioData:
    """Generate silence of the given duration."""
    return AudioData([0.0] * int(duration * sample_rate), sample_rate)


def concat(*parts: AudioData) -> AudioData:
    """Concatenate audio clips (all resampled to the first clip's rate)."""
    if not parts:
        return AudioData([], DEFAULT_SAMPLE_RATE)
    rate = parts[0].sample_rate
    samples: list[float] = []
    for part in parts:
        samples.extend(resample(part, rate).samples)
    return AudioData(samples, rate)
