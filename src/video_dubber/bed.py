"""Background-music bed: keep the original recording's non-speech audio
(music, ambience, crowd noise) under the dubbed speech.

A dub that replaces the whole soundtrack with silence between lines
sounds dead — real dubs keep the original music bed audible, ducked
under the new voice. The pipeline:

1. **Extract** (:func:`extract_bed`): copy the original audio and mute
   the speech windows (the transcript's segment ranges), leaving the
   gaps — music and ambience — untouched. Short fades at the cut edges
   keep the bed click-free.
2. **Duck** (:func:`duck_bed`): attenuate the bed to ``duck_level``
   underneath the placed dubbed clips, with fades, so the new voice
   stays intelligible while the music is still audible.
3. **Mix** (:func:`mix_bed`): add the ducked bed under the dubbed track
   at ``bed_level`` gain, peak-clamped.

All three steps are pure stdlib float math on :class:`AudioData`, so
they run anywhere the rest of the pipeline does.
"""

from __future__ import annotations

from video_dubber.audio import AudioData, resample

__all__ = [
    "BedError",
    "duck_bed",
    "extract_bed",
    "mix_bed",
]

FADE_MS = 30.0  # fade length at speech-window edges when extracting a bed
DUCK_FADE_MS = 50.0  # fade length when ducking under a dubbed clip


class BedError(Exception):
    """Raised when bed extraction, ducking or mixing is misconfigured."""


def _check_level(level: float, name: str) -> None:
    if not 0.0 <= level <= 1.0:
        raise BedError(f"{name} must be in [0, 1], got {level}")


def _fade_samples(sample_rate: int, fade_ms: float) -> int:
    return max(1, int(sample_rate * fade_ms / 1000.0))


def _apply_fades(
    gains: list[float],
    start: int,
    end: int,
    fade: int,
    inside: float,
    outside: float,
) -> None:
    """Set ``gains[start:end]`` to ``inside``, ramping from ``outside``.

    The ramp runs over ``fade`` samples at each edge so a cut never
    starts or ends with a step (a step is an audible click).
    """
    if start >= end:
        return
    for i in range(start, end):
        gain = inside
        if i - start < fade:
            t = (i - start + 1) / fade
            gain = outside + (inside - outside) * t
        elif end - 1 - i < fade:
            t = (end - i) / fade
            gain = outside + (inside - outside) * t
        gains[i] = gain


def extract_bed(
    original: AudioData,
    segments: list[tuple[float, float]],
    fade_ms: float = FADE_MS,
) -> AudioData:
    """Return the original audio with its speech windows muted.

    ``segments`` are ``(start, end)`` seconds of speech (e.g. the
    transcript's segment ranges). Everything outside them — music,
    ambience, room tone — is kept sample-for-sample; inside them the
    bed is silent, with a short linear fade at each edge. Segment
    ranges are clamped to the audio length; a range that starts at or
    after the end of the audio, or ends before 0, is ignored. Empty
    audio returns empty audio.
    """
    if fade_ms <= 0:
        raise BedError(f"fade_ms must be positive, got {fade_ms}")
    if not original.samples:
        return AudioData([], original.sample_rate)
    gains = [1.0] * len(original.samples)
    fade = _fade_samples(original.sample_rate, fade_ms)
    for start, end in segments:
        if end <= 0 or start >= original.duration:
            continue
        lo = max(0, int(start * original.sample_rate))
        hi = min(len(original.samples), int(end * original.sample_rate))
        _apply_fades(gains, lo, hi, fade, inside=0.0, outside=1.0)
    return AudioData([s * g for s, g in zip(original.samples, gains)], original.sample_rate)


def duck_bed(
    bed: AudioData,
    placed: list[tuple[float, float]],
    duck_level: float = 0.15,
    fade_ms: float = DUCK_FADE_MS,
) -> AudioData:
    """Attenuate the bed to ``duck_level`` under placed dubbed clips.

    ``placed`` are ``(start, duration)`` pairs of the dubbed segments
    (the ``placed`` list :func:`dub_transcript` returns). Outside them
    the bed plays at full level; under speech it drops to
    ``duck_level`` (0.15 by default — audible but well under the voice),
    with a short fade at each boundary. A ``duck_level`` of 0 silences
    the bed under speech entirely; 1.0 leaves it untouched.
    """
    _check_level(duck_level, "duck_level")
    if fade_ms <= 0:
        raise BedError(f"fade_ms must be positive, got {fade_ms}")
    if not bed.samples:
        return AudioData([], bed.sample_rate)
    gains = [1.0] * len(bed.samples)
    fade = _fade_samples(bed.sample_rate, fade_ms)
    for start, duration in placed:
        if duration <= 0 or start >= bed.duration:
            continue
        lo = max(0, int(start * bed.sample_rate))
        hi = min(len(bed.samples), int((start + duration) * bed.sample_rate))
        _apply_fades(gains, lo, hi, fade, inside=duck_level, outside=1.0)
    return AudioData([s * g for s, g in zip(bed.samples, gains)], bed.sample_rate)


def mix_bed(dub: AudioData, bed: AudioData, bed_level: float = 0.3) -> AudioData:
    """Mix the (ducked) bed under the dubbed track at ``bed_level`` gain.

    The bed is resampled to the dub's sample rate when needed, then
    added sample-by-sample and clamped to ``[-1, 1]``. A bed shorter
    than the dub is zero-padded; a longer one is truncated to the dub's
    length. ``bed_level`` 0 returns the dub unchanged; 1.0 mixes the
    bed at full strength.
    """
    _check_level(bed_level, "bed_level")
    if not dub.samples:
        return AudioData([], dub.sample_rate)
    if not bed.samples or bed_level == 0.0:
        return AudioData(list(dub.samples), dub.sample_rate)
    bed = resample(bed, dub.sample_rate) if bed.sample_rate != dub.sample_rate else bed
    out: list[float] = []
    for i, sample in enumerate(dub.samples):
        bed_sample = bed.samples[i] if i < len(bed.samples) else 0.0
        out.append(max(-1.0, min(1.0, sample + bed_sample * bed_level)))
    return AudioData(out, dub.sample_rate)
