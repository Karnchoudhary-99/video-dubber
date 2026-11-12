"""Re-dub one segment of an existing dub, without re-running the whole dub.

A dub is rarely wrong everywhere: usually one line lands badly — a
mistranslation, a name the synthesiser mangled, a fit that clamped so
hard the line is rushed. Re-running ``dub`` for that one line wastes
the good segments (and, with a real TTS backend, the money).

:func:`patch_transcript` swaps one segment's text in the translated
transcript, retiming its words proportionally across the segment's
original window so subtitles and reports stay consistent.
:func:`patch_dub` re-synthesis only that segment, fits it to the same
window, loudness-matches it to the clip it replaces (or to the
original recording when one is given), and splices it into the
existing dub audio — only the old placed span is zeroed, so a music
bed under the rest of the dub is untouched. :func:`refresh_report`
updates the patched row in a saved :class:`DubReport` and recomputes
its aggregates, so the report still describes the audio on disk.
"""

from __future__ import annotations

from dataclasses import replace

from video_dubber.assemble import (
    AssemblyError,
    DubReport,
    SegmentRow,
    fit_to_window,
    loudness_match,
)
from video_dubber.audio import AudioData
from video_dubber.quality import mos_proxy as _mos_proxy
from video_dubber.style import DubStyle, apply_style, get_style
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import SpeakerProfile, VoiceCloner

__all__ = [
    "PatchError",
    "patch_dub",
    "patch_transcript",
    "refresh_report",
    "split_words_across_span",
]

# Drift / fit limits are the report's own: a segment counts as over-drift
# past 0.10s, and the verdict flips to "review" past 0.25s max drift.
OVER_DRIFT_S = 0.10
REVIEW_DRIFT_S = 0.25
EXTREME_FIT_LOW = 0.8
EXTREME_FIT_HIGH = 1.25


class PatchError(Exception):
    """Raised when a segment cannot be patched (bad index, empty text, ...)."""


def split_words_across_span(text: str, start: float, end: float) -> list[Word]:
    """Split ``text`` into words timed across ``[start, end]``.

    Words share the span in proportion to their character length (a
    long word gets more time than a short one); the last word's end is
    pinned to ``end`` so the cue covers the whole window. Raises
    :class:`PatchError` on blank text or a non-positive span — the
    caller has no real words or no window to put them in.
    """
    words = text.split()
    if not words:
        raise PatchError("patched segment text must not be empty")
    if end <= start:
        raise PatchError(f"segment span must be positive, got {start:.2f} -> {end:.2f}")
    total_chars = sum(len(w) for w in words)
    timed: list[Word] = []
    cursor = start
    for i, word in enumerate(words):
        share = (end - start) * len(word) / total_chars
        word_end = end if i == len(words) - 1 else cursor + share
        timed.append(Word(word, cursor, word_end))
        cursor = word_end
    return timed


def _timed_segments(transcript: Transcript) -> list[Segment]:
    """Segments that carry words, in transcript order (the patchable ones)."""
    return [segment for segment in transcript.segments if segment.words]


def patch_transcript(transcript: Transcript, index: int, new_text: str) -> Transcript:
    """Return a copy of ``transcript`` with timed segment ``index`` re-worded.

    ``index`` is 1-based among the segments that have words — the same
    numbering the quality report's per-segment rows use, so ``report
    --format`` and ``patch --segment`` agree on which line is which.
    The segment keeps its speaker label and its original window; only
    its words change, retimed across that window by
    :func:`split_words_across_span`. Raises :class:`PatchError` for an
    out-of-range index or blank text. The input transcript is untouched.
    """
    timed = _timed_segments(transcript)
    if index < 1 or index > len(timed):
        raise PatchError(
            f"segment {index} out of range: transcript has {len(timed)} timed segments"
        )
    target = timed[index - 1]
    new_words = split_words_across_span(new_text, target.start, target.end)
    segments: list[Segment] = []
    seen = 0
    for segment in transcript.segments:
        if not segment.words:
            segments.append(Segment(words=[], speaker=segment.speaker))
            continue
        seen += 1
        if seen == index:
            segments.append(Segment(words=new_words, speaker=segment.speaker))
        else:
            segments.append(segment)
    return Transcript(segments=segments, language=transcript.language, source=transcript.source)


def patch_dub(
    dub_audio: AudioData,
    transcript: Transcript,
    index: int,
    new_text: str,
    cloner: VoiceCloner,
    profile: SpeakerProfile,
    original: AudioData | None = None,
    profiles: dict[str, SpeakerProfile] | None = None,
    placed: list[tuple[float, float]] | None = None,
    style: DubStyle | str | None = None,
) -> tuple[AudioData, float, float]:
    """Re-dub timed segment ``index`` inside an existing dub track.

    The new line is synthesised in the segment speaker's voice (the
    ``profiles`` map, falling back to ``profile`` — the same rule
    :func:`~video_dubber.assemble.dub_transcript` uses), stretched to
    its original window, and spliced in: only the old placed span is
    zeroed before the new clip is mixed in, so the rest of the track —
    including any music bed — is bit-identical. ``placed`` is the
    ``(start, duration)`` list :func:`dub_transcript` returned for the
    original dub; without it the whole segment window is zeroed, which
    also erases bed audio in the gap after a clamped clip.

    Loudness: with ``original`` the new clip is matched to the original
    recording (as a full dub is); without it, to the RMS of the clip it
    replaces, so the patched line does not jump in volume against its
    neighbours.

    Returns ``(patched_audio, fit_factor, placed_duration)``. Raises
    :class:`PatchError` for a bad index or blank text, and
    :class:`~video_dubber.assemble.AssemblyError` when the dub track
    does not cover the segment's window.

    With ``style``, the replacement line is read in that delivery
    style — the same style the original dub used, so the patched line
    does not sound like a different performance dropped into the track.
    """
    timed = _timed_segments(transcript)
    if index < 1 or index > len(timed):
        raise PatchError(
            f"segment {index} out of range: transcript has {len(timed)} timed segments"
        )
    segment = timed[index - 1]
    if not new_text.split():
        raise PatchError("patched segment text must not be empty")
    if placed is not None and len(placed) != len(timed):
        raise PatchError(f"placed has {len(placed)} spans for {len(timed)} timed segments")
    if dub_audio.duration < segment.end:
        raise AssemblyError(
            f"dub audio ({dub_audio.duration:.2f}s) does not cover segment {index} "
            f"ending at {segment.end:.2f}s"
        )
    resolved_style = get_style(style) if isinstance(style, str) else style
    voice = profile
    if profiles is not None and segment.speaker is not None:
        voice = profiles.get(segment.speaker, profile)
    if resolved_style is not None:
        voice = apply_style(voice, resolved_style)
    spoken = cloner.synthesize(new_text, voice)
    fitted, factor = fit_to_window(spoken, segment.duration)
    if original is not None:
        fitted = loudness_match(fitted, original)
    else:
        # Match the clip being replaced: its RMS is the volume the
        # neighbours were mixed against.
        old_start = int(segment.start * dub_audio.sample_rate)
        old_end = int(segment.end * dub_audio.sample_rate)
        old_clip = AudioData(dub_audio.samples[old_start:old_end], dub_audio.sample_rate)
        fitted = loudness_match(fitted, old_clip)
    if placed is not None:
        old_start, old_duration = placed[index - 1]
    else:
        old_start, old_duration = segment.start, segment.duration
    samples = list(dub_audio.samples)
    zero_start = int(old_start * dub_audio.sample_rate)
    zero_end = int((old_start + old_duration) * dub_audio.sample_rate)
    for i in range(zero_start, min(zero_end, len(samples))):
        samples[i] = 0.0
    offset = int(segment.start * dub_audio.sample_rate)
    for i, sample in enumerate(fitted.samples):
        if offset + i < len(samples):
            samples[offset + i] = max(-1.0, min(1.0, samples[offset + i] + sample))
    return AudioData(samples, dub_audio.sample_rate), factor, fitted.duration


def refresh_report(report: DubReport, row: SegmentRow) -> DubReport:
    """Return ``report`` with one per-segment row replaced, aggregates redone.

    ``row`` must carry the ``index`` of an existing row (1-based, as in
    :func:`patch_transcript`); every aggregate that depends on the rows
    — mean/max drift, over-drift count, mean/max fit, extreme fits,
    MOS proxy, verdict — is recomputed from the updated rows. Fields
    the patch cannot change (loudness error, WER, coverage, bed level,
    enhancement) are carried over untouched. The verdict rule mirrors
    :func:`~video_dubber.assemble.build_report`: ``review`` when max
    drift exceeds 0.25s, any fit is extreme, loudness is off by more
    than 3 dB, or WER exceeds 0.25.
    """
    rows = list(report.segments)
    positions = [i for i, existing in enumerate(rows) if existing.index == row.index]
    if not positions:
        raise PatchError(f"report has no segment row {row.index} to refresh")
    rows[positions[0]] = row
    drifts = [r.drift for r in rows]
    fits = [r.fit_factor for r in rows]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    max_drift = max(drifts, default=0.0)
    mean_fit = sum(fits) / len(fits) if fits else 1.0
    max_fit = max(fits, default=1.0)
    extreme = sum(1 for f in fits if f < EXTREME_FIT_LOW or f > EXTREME_FIT_HIGH)
    verdict = "good"
    if (
        max_drift > REVIEW_DRIFT_S
        or extreme > 0
        or (report.loudness_error_db is not None and abs(report.loudness_error_db) > 3.0)
        or (report.word_error_rate is not None and report.word_error_rate > 0.25)
    ):
        verdict = "review"
    return replace(
        report,
        mean_sync_drift=mean_drift,
        max_sync_drift=max_drift,
        segments_over_drift=sum(1 for d in drifts if d > OVER_DRIFT_S),
        mean_fit_factor=mean_fit,
        max_fit_factor=max_fit,
        extreme_fits=extreme,
        mos_proxy=_mos_proxy(
            mean_drift,
            max_drift,
            mean_fit,
            extreme,
            len(rows),
            report.loudness_error_db,
            report.word_error_rate,
        ),
        verdict=verdict,
        segments=rows,
    )
