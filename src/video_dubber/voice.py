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
import struct
from dataclasses import dataclass, field
from typing import Protocol

from video_dubber.audio import AudioData, concat, make_silence, make_sine

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
