"""Voice cloning: speaker profiles and text-to-speech synthesis.

A :class:`SpeakerProfile` is the portable identity of a voice: a small
embedding vector extracted from reference recordings, plus the prosody
the dub needs (base pitch, speaking rate). Downstream, dub assembly
renders translated text in that voice, timed to the original segments.

This module is deliberately honest about what it is:

* :func:`extract_embedding` computes a real, deterministic embedding
  from reference audio — stdlib spectral features (band energies, zero
  crossing, spectral centroid, pitch estimate), not a neural encoder.
  It captures timbre well enough to tell speakers apart and to match a
  profile against a new sample, but it is not a production voice clone.
* :class:`ScriptedVoiceCloner` is the demo double: it synthesises
  formant-like speech (a harmonic tone shaped per vowel, with per-word
  timing from the transcript) in the profile's pitch and rate. It sounds
  like a voice, not like the person — a labelled stand-in until a real
  TTS backend (Coqui XTTS / OpenVoice) is wired in as a
  :class:`VoiceCloner` implementation, the same way Whisper and Argos are
  adapters for transcription and translation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Protocol

from video_dubber.audio import AudioData, concat, make_silence, resample

__all__ = [
    "SpeakerProfile",
    "ScriptedVoiceCloner",
    "VoiceCloneError",
    "VoiceCloner",
    "clone_from_references",
    "cosine_similarity",
    "extract_embedding",
    "extract_features",
    "synthesize_text",
]

EMBEDDING_DIM = 16
DEFAULT_SAMPLE_RATE = 16_000


class VoiceCloneError(Exception):
    """Raised when a voice profile or synthesis request is unusable."""


@dataclass
class SpeakerProfile:
    """Portable voice identity: embedding + prosody, JSON-serialisable.

    ``embedding`` is a unit-length vector of :data:`EMBEDDING_DIM` floats
    from :func:`extract_embedding` (or a demo double). ``base_pitch_hz``
    is the speaker's median fundamental; ``speaking_rate_wpm`` their
    words-per-minute. ``reference_files`` records provenance only — the
    profile never stores raw audio.
    """

    name: str
    embedding: list[float]
    base_pitch_hz: float = 120.0
    speaking_rate_wpm: float = 150.0
    reference_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.embedding) != EMBEDDING_DIM:
            raise VoiceCloneError(
                f"embedding must be {EMBEDDING_DIM} floats, got {len(self.embedding)}"
            )
        if self.base_pitch_hz <= 0:
            raise VoiceCloneError(f"base_pitch_hz must be positive, got {self.base_pitch_hz}")
        if self.speaking_rate_wpm <= 0:
            raise VoiceCloneError(
                f"speaking_rate_wpm must be positive, got {self.speaking_rate_wpm}"
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "embedding": self.embedding,
                "base_pitch_hz": self.base_pitch_hz,
                "speaking_rate_wpm": self.speaking_rate_wpm,
                "reference_files": self.reference_files,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, data: str) -> SpeakerProfile:
        try:
            raw = json.loads(data)
            return cls(
                name=raw["name"],
                embedding=[float(v) for v in raw["embedding"]],
                base_pitch_hz=float(raw.get("base_pitch_hz", 120.0)),
                speaking_rate_wpm=float(raw.get("speaking_rate_wpm", 150.0)),
                reference_files=list(raw.get("reference_files", [])),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise VoiceCloneError(f"invalid speaker profile JSON: {exc}") from exc


# --- Spectral features (stdlib only) -----------------------------------------
# Frames are 25 ms with a 10 ms hop. Per frame we measure RMS, zero-crossing
# rate, band energies at 8 log-spaced frequencies (Goertzel — cheap and
# dependency-free), a spectral centroid from those bands, and a pitch
# estimate by autocorrelation. Aggregated over voiced frames, these
# separate speakers by timbre well enough for matching and demo synthesis.

_FRAME_MS = 25
_HOP_MS = 10
_BAND_FREQS = (100, 200, 400, 800, 1600, 3200, 4800, 7000)


def _goertzel(samples: list[float], sample_rate: int, freq: float) -> float:
    """Magnitude of one DFT bin, computed without an FFT library."""
    if not samples:
        return 0.0
    coeff = 2.0 * math.cos(2.0 * math.pi * freq / sample_rate)
    s1 = s2 = 0.0
    for sample in samples:
        s0 = sample + coeff * s1 - s2
        s2, s1 = s1, s0
    power = s1 * s1 + s2 * s2 - coeff * s1 * s2
    return math.sqrt(max(0.0, power)) / len(samples)


def _autocorr_pitch(frame: list[float], sample_rate: int) -> float:
    """Fundamental estimate (Hz) by autocorrelation, 0.0 if unvoiced.

    Searches lags for 50–400 Hz; a frame counts as voiced only when the
    best normalised autocorrelation exceeds 0.3 (silence and noise stay 0).
    """
    n = len(frame)
    if n < 4:
        return 0.0
    mean = sum(frame) / n
    centred = [s - mean for s in frame]
    energy = sum(s * s for s in centred)
    if energy <= 0.0:
        return 0.0
    min_lag = max(1, sample_rate // 400)
    max_lag = min(n - 1, sample_rate // 50)
    best_lag, best_value = 0, 0.0
    for lag in range(min_lag, max_lag + 1):
        corr = sum(centred[i] * centred[i + lag] for i in range(n - lag)) / energy
        if corr > best_value:
            best_value, best_lag = corr, lag
    if best_value < 0.3:
        return 0.0
    return sample_rate / best_lag


def _frame_features(frame: list[float], sample_rate: int) -> dict[str, float]:
    n = len(frame)
    rms = math.sqrt(sum(s * s for s in frame) / n) if n else 0.0
    crossings = sum(
        1 for a, b in zip(frame, frame[1:]) if (a < 0) != (b < 0)
    )
    zcr = crossings / max(1, n - 1)
    bands = [_goertzel(frame, sample_rate, f) for f in _BAND_FREQS]
    band_total = sum(bands) or 1.0
    centroid = sum(f * b for f, b in zip(_BAND_FREQS, bands)) / band_total
    return {
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "pitch": _autocorr_pitch(frame, sample_rate),
        **{f"band{i}": b / band_total for i, b in enumerate(bands)},
    }


def extract_features(audio: AudioData) -> dict[str, float]:
    """Aggregate per-frame spectral features over the voiced frames.

    Returns means of RMS, zero-crossing rate, spectral centroid, pitch
    (voiced frames only) and the 8 normalised band energies. Raises
    :class:`VoiceCloneError` for empty or silent audio.
    """
    rate = audio.sample_rate
    frame_len = max(64, int(rate * _FRAME_MS / 1000))
    hop = max(32, int(rate * _HOP_MS / 1000))
    frames = [
        audio.samples[i : i + frame_len]
        for i in range(0, max(1, len(audio.samples) - frame_len + 1), hop)
    ]
    if not frames or not any(frames):
        raise VoiceCloneError("cannot extract features from empty audio")
    per_frame = [_frame_features(f, rate) for f in frames]
    voiced = [f for f in per_frame if f["rms"] > 0.01]
    if not voiced:
        raise VoiceCloneError("reference audio is silent (no frame above the noise floor)")
    keys = list(per_frame[0])
    out = {k: sum(f[k] for f in voiced) / len(voiced) for k in keys}
    out["voiced_ratio"] = len(voiced) / len(per_frame)
    return out


def extract_embedding(audio: AudioData) -> list[float]:
    """Deterministic speaker embedding from reference audio.

    16 dims: log-RMS, zero-crossing rate, normalised centroid, normalised
    pitch, voiced ratio, spectral flatness, band-energy spread, the 8
    normalised band energies, and the low-frequency share. The vector is
    L2-normalised so :func:`cosine_similarity` is a plain dot product.
    """
    feats = extract_features(audio)
    bands = [feats[f"band{i}"] for i in range(len(_BAND_FREQS))]
    band_mean = sum(bands) / len(bands) or 1.0
    geo_mean = math.exp(sum(math.log(b + 1e-9) for b in bands) / len(bands))
    flatness = geo_mean / band_mean
    spread = math.sqrt(sum((b - band_mean) ** 2 for b in bands) / len(bands))
    raw = [
        (math.log10(feats["rms"] + 1e-6) + 3.0) / 3.0,
        feats["zcr"],
        feats["centroid"] / 7000.0,
        feats["pitch"] / 400.0,
        feats["voiced_ratio"],
        flatness,
        spread,
        *bands,
        bands[0] + bands[1],
    ]
    assert len(raw) == EMBEDDING_DIM, f"embedding dim drift: {len(raw)}"
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two embeddings in ``[-1, 1]`` (1 = same voice)."""
    if len(a) != len(b):
        raise VoiceCloneError(f"cannot compare embeddings of dim {len(a)} and {len(b)}")
    denom = math.sqrt(sum(v * v for v in a)) * math.sqrt(sum(v * v for v in b))
    if denom == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def clone_from_references(
    name: str,
    audios: list[AudioData],
    reference_files: list[str] | None = None,
) -> SpeakerProfile:
    """Build a speaker profile by averaging embeddings across references.

    Base pitch is the mean voiced pitch across the references (120 Hz
    fallback when no frame is voiced); speaking rate defaults to 150 wpm
    — estimating it needs a transcript, which the CLI supplies with
    ``--wpm`` when known. Needs at least one reference clip.
    """
    if not audios:
        raise VoiceCloneError("need at least one reference recording to clone a voice")
    embeddings = [extract_embedding(a) for a in audios]
    mean = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(v * v for v in mean)) or 1.0
    pitches = [extract_features(a)["pitch"] for a in audios]
    voiced_pitches = [p for p in pitches if p > 0]
    base_pitch = sum(voiced_pitches) / len(voiced_pitches) if voiced_pitches else 120.0
    return SpeakerProfile(
        name=name,
        embedding=[v / norm for v in mean],
        base_pitch_hz=base_pitch,
        speaking_rate_wpm=150.0,
        reference_files=list(reference_files or []),
    )


class VoiceCloner(Protocol):
    """Clones a voice from reference audio and synthesises speech in it.

    Implementations raise :class:`VoiceCloneError` on failure. The
    shipped implementation is :class:`ScriptedVoiceCloner` (a labelled
    demo double); a real TTS backend slots in here behind the same
    protocol, exactly like Whisper/Argos do for STT/translation.
    """

    def clone(
        self,
        name: str,
        audios: list[AudioData],
        reference_files: list[str] | None = None,
    ) -> SpeakerProfile: ...

    def synthesize(
        self, text: str, profile: SpeakerProfile, words_per_minute: float | None = None
    ) -> AudioData: ...

    @property
    def name(self) -> str: ...


# --- Formant-ish synthesis (demo double) -------------------------------------
# A word is a short harmonic tone: the fundamental is the speaker's base
# pitch (varied deterministically per word, the way real intonation rises
# and falls), and the harmonic amplitudes are boosted near the first two
# formants of the word's dominant vowel. An attack/decay envelope keeps
# words from clicking. It is recognisably speech-shaped, not speech — the
# docstring says so, and the class name says so.

_VOWEL_FORMANTS: dict[str, tuple[float, float]] = {
    "a": (730, 1090),
    "e": (530, 1840),
    "i": (270, 2290),
    "o": (570, 840),
    "u": (300, 870),
}


def _word_pitch(base_pitch: float, word: str) -> float:
    """Deterministic per-word pitch: base pitch varied ±10% by word hash."""
    digest = hashlib.md5(word.encode("utf-8")).digest()
    jitter = (digest[0] / 255.0 - 0.5) * 0.2  # [-0.1, +0.1]
    return base_pitch * (1.0 + jitter)


def _word_tone(word: str, pitch: float, duration: float, sample_rate: int) -> AudioData:
    vowel = next((ch for ch in word.lower() if ch in _VOWEL_FORMANTS), "a")
    f1, f2 = _VOWEL_FORMANTS[vowel]
    count = max(1, int(duration * sample_rate))
    samples: list[float] = []
    for i in range(count):
        t = i / sample_rate
        value = 0.0
        for harmonic in range(1, 7):
            freq = pitch * harmonic
            # Boost harmonics near either formant, fall off elsewhere.
            boost = 1.0 + 2.0 * math.exp(-((freq - f1) / 300.0) ** 2)
            boost += 1.5 * math.exp(-((freq - f2) / 400.0) ** 2)
            value += (boost / harmonic) * math.sin(2.0 * math.pi * freq * t)
        # Attack/decay envelope: 10 ms in, 30 ms out.
        attack = min(1.0, t / 0.01)
        decay = min(1.0, (duration - t) / 0.03) if duration > 0.04 else 1.0
        samples.append(0.25 * value * attack * max(0.0, decay))
    return AudioData(samples, sample_rate)


def synthesize_text(
    text: str,
    profile: SpeakerProfile,
    words_per_minute: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioData:
    """Render ``text`` as audio in the profile's voice (demo synthesis).

    Word duration follows the speaking rate (5 chars ≈ 1 word); words are
    separated by a 50 ms gap. The result's duration is therefore
    predictable from the text, which is what dub assembly relies on to
    stretch it into the segment window. Empty text returns silence.
    """
    wpm = profile.speaking_rate_wpm if words_per_minute is None else words_per_minute
    if wpm <= 0:
        raise VoiceCloneError(f"words_per_minute must be positive, got {wpm}")
    words = text.split()
    if not words:
        return AudioData([], sample_rate)
    parts: list[AudioData] = []
    for word in words:
        duration = max(0.12, (len(word) / 5.0) * (60.0 / wpm))
        parts.append(_word_tone(word, _word_pitch(profile.base_pitch_hz, word), duration, sample_rate))
        parts.append(make_silence(0.05, sample_rate))
    return concat(*parts).normalized(0.8)


@dataclass
class ScriptedVoiceCloner:
    """Demo voice cloner: real embeddings, formant-style synthesis.

    ``clone`` extracts a genuine embedding from the references (same
    features as :func:`extract_embedding`); ``synthesize`` renders text
    with :func:`synthesize_text`. The voice *identity* is real — two
    different speakers get measurably different embeddings — while the
    *sound* is a labelled stand-in for a neural TTS backend.
    """

    @property
    def name(self) -> str:
        return "scripted"

    def clone(
        self,
        name: str,
        audios: list[AudioData],
        reference_files: list[str] | None = None,
    ) -> SpeakerProfile:
        return clone_from_references(name, audios, reference_files)

    def synthesize(
        self, text: str, profile: SpeakerProfile, words_per_minute: float | None = None
    ) -> AudioData:
        return synthesize_text(text, profile, words_per_minute)


def adjust_prosody(
    profile: SpeakerProfile,
    semitones: float = 0.0,
    words_per_minute: float | None = None,
) -> SpeakerProfile:
    """Return a copy of ``profile`` with shifted pitch and/or speaking rate.

    Pitch shifts by ``semitones`` (``+12`` = one octave up, via
    ``2 ** (semitones / 12)``); rate replaces the profile's wpm when
    given. The embedding is untouched — prosody is performance, not
    identity — so a shifted profile still matches the original speaker.
    """
    if words_per_minute is not None and words_per_minute <= 0:
        raise VoiceCloneError(
            f"words_per_minute must be positive, got {words_per_minute}"
        )
    return SpeakerProfile(
        name=profile.name,
        embedding=list(profile.embedding),
        base_pitch_hz=profile.base_pitch_hz * (2.0 ** (semitones / 12.0)),
        speaking_rate_wpm=words_per_minute or profile.speaking_rate_wpm,
        reference_files=list(profile.reference_files),
    )


def time_stretch(audio: AudioData, factor: float) -> AudioData:
    """Change duration by ``factor`` without changing the sample rate.

    ``factor > 1`` makes the audio longer (slower); ``factor < 1`` makes
    it shorter. Linear-interpolation resampling: quality is adequate for
    the demo double's tones, and the function is what dub assembly uses
    to fit synthesised speech into its segment window. Raises
    :class:`VoiceCloneError` on a non-positive factor.
    """
    if factor <= 0:
        raise VoiceCloneError(f"time-stretch factor must be positive, got {factor}")
    if not audio.samples or factor == 1.0:
        return AudioData(list(audio.samples), audio.sample_rate)
    stretched = AudioData(list(audio.samples), audio.sample_rate)
    # Resample to a higher rate then relabel: more samples at the same
    # rate means a longer clip at the same pitch contour shape.
    resampled = resample(stretched, max(1, round(audio.sample_rate * factor)))
    return AudioData(resampled.samples, audio.sample_rate)
