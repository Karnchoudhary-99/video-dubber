"""Dubbing style presets: named delivery styles for a dubbed voice.

A style is performance, not identity: it scales the speaking rate and
shifts the pitch of a cloned voice (via
:func:`video_dubber.voice.adjust_prosody`), so the same speaker can
read a documentary slowly and a trailer fast without re-cloning. The
embedding is untouched, so a styled profile still matches its speaker
— a style never turns Asha into someone else, it just changes how
fast and how high she reads.

Styles matter because fit is the dub's hard constraint: a fast style
fits long translations into short windows with less time-stretching,
while a slow style needs wider windows. Pick the style before
estimating, and the estimate matches the dub.
"""

from __future__ import annotations

from dataclasses import dataclass

from video_dubber.voice import SpeakerProfile, VoiceCloneError, adjust_prosody

__all__ = [
    "BUILTIN_STYLES",
    "DubStyle",
    "StyleError",
    "apply_style",
    "get_style",
    "list_styles",
]


class StyleError(Exception):
    """Raised for an unknown style name or an invalid style definition."""


@dataclass(frozen=True)
class DubStyle:
    """One delivery style: a rate multiplier and a pitch shift.

    ``rate_factor`` multiplies the speaker's words-per-minute (``1.15``
    reads 15% faster); ``pitch_semitones`` shifts the base pitch
    (``+2`` is two semitones up, ``-2`` two down). Bounds keep styles
    in the range a voice can still carry: rate 0.5–2.0x, pitch ±12
    semitones (one octave either way).
    """

    name: str
    description: str
    rate_factor: float = 1.0
    pitch_semitones: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise StyleError("style name must not be empty")
        if not 0.5 <= self.rate_factor <= 2.0:
            raise StyleError(
                f"style {self.name!r}: rate_factor must be between 0.5 and 2.0, "
                f"got {self.rate_factor}"
            )
        if not -12.0 <= self.pitch_semitones <= 12.0:
            raise StyleError(
                f"style {self.name!r}: pitch_semitones must be between -12 and 12, "
                f"got {self.pitch_semitones}"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "rate_factor": self.rate_factor,
            "pitch_semitones": self.pitch_semitones,
        }


BUILTIN_STYLES: tuple[DubStyle, ...] = (
    DubStyle(
        name="documentary",
        description="Measured, steady delivery for documentaries and explainers.",
        rate_factor=0.95,
        pitch_semitones=0.0,
    ),
    DubStyle(
        name="conversational",
        description="Natural everyday speech — the default when no style is given.",
        rate_factor=1.0,
        pitch_semitones=0.0,
    ),
    DubStyle(
        name="energetic",
        description="Fast, bright delivery for trailers, ads and sports.",
        rate_factor=1.15,
        pitch_semitones=2.0,
    ),
    DubStyle(
        name="narration",
        description="Slow, low audiobook-style narration.",
        rate_factor=0.90,
        pitch_semitones=-1.0,
    ),
    DubStyle(
        name="dramatic",
        description="Deliberate, low delivery for drama and tension.",
        rate_factor=0.85,
        pitch_semitones=-2.0,
    ),
)


def list_styles() -> list[DubStyle]:
    """All built-in styles, in display order."""
    return list(BUILTIN_STYLES)


def get_style(name: str) -> DubStyle:
    """Look a style up by name (case-insensitive, surrounding space ignored).

    Raises :class:`StyleError` naming the known styles when the name
    is not one of them — a typo should fail loudly, not silently dub
    in the default style.
    """
    wanted = name.strip().lower()
    for style in BUILTIN_STYLES:
        if style.name == wanted:
            return style
    known = ", ".join(style.name for style in BUILTIN_STYLES)
    raise StyleError(f"unknown dub style {name!r} (known styles: {known})")


def apply_style(profile: SpeakerProfile, style: DubStyle | str) -> SpeakerProfile:
    """Return ``profile`` read in ``style`` (rate scaled, pitch shifted).

    ``style`` may be a :class:`DubStyle` or its name. The returned
    profile keeps the original name and embedding — only the prosody
    changes — so voice-library matching still identifies the speaker.
    """
    resolved = style if isinstance(style, DubStyle) else get_style(style)
    try:
        return adjust_prosody(
            profile,
            semitones=resolved.pitch_semitones,
            words_per_minute=profile.speaking_rate_wpm * resolved.rate_factor,
        )
    except VoiceCloneError as exc:
        raise StyleError(str(exc)) from exc
