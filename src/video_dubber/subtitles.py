"""Dub-synced subtitles: retime subtitle cues to the dubbed audio.

``dub --subtitles`` used to write SRT/VTT straight from the translated
transcript, so every cue kept its *original* timing window — but the dub
does not speak inside that window exactly. ``dub_transcript`` stretches
each segment to fit its window, and when the stretch has to be clamped
(a segment that needs more than 0.6x–1.6x) the placed clip comes out
shorter than its window: the dubbed speech ends early and the subtitle
lingers on screen over silence (or over the next line's music bed).

:func:`retime_transcript` fixes that: it takes the ``placed`` spans
``dub_transcript`` returns — ``(start, duration)`` per timed segment —
and rescales each segment's word timings proportionally into its placed
span, so the SRT/VTT cues match what the listener actually hears.
Short cues are stretched to a readable minimum (default 0.5s) without
ever overlapping the next cue.
"""

from __future__ import annotations

from video_dubber.transcript import Segment, Transcript, Word

__all__ = ["SubtitleError", "retime_transcript"]

DEFAULT_MIN_CUE_S = 0.5  # a cue shorter than this is unreadable on screen


class SubtitleError(Exception):
    """Raised when subtitles cannot be retimed (bad placed spans)."""


def retime_transcript(
    transcript: Transcript,
    placed: list[tuple[float, float]],
    min_cue_s: float = DEFAULT_MIN_CUE_S,
) -> Transcript:
    """Return a copy of ``transcript`` retimed to the dubbed audio.

    ``placed`` is the ``(start, duration)`` list ``dub_transcript``
    returns, one entry per segment that has words, in transcript order.
    Wordless segments have no placed entry and are copied unchanged
    (they produce no subtitle cue anyway). Word timings inside a
    segment are rescaled proportionally, so a word that sat halfway
    through the original window still sits halfway through the dub.

    Raises :class:`SubtitleError` when the placed list does not line up
    with the timed segments, or when a span is negative or zero-length.
    ``min_cue_s`` must be non-negative; a cue shorter than it is
    extended up to the minimum, but never past the next cue's start.
    """
    if min_cue_s < 0:
        raise SubtitleError(f"min_cue_s must be non-negative, got {min_cue_s}")
    timed = [segment for segment in transcript.segments if segment.words]
    if len(placed) != len(timed):
        raise SubtitleError(f"placed has {len(placed)} spans for {len(timed)} timed segments")
    for start, duration in placed:
        if start < 0:
            raise SubtitleError(f"placed span starts before 0: {start}")
        if duration <= 0:
            raise SubtitleError(f"placed span has non-positive duration: {duration}")

    # A short cue may extend into the silence gap, but never into the
    # next cue — so work out each span's capped end up front.
    ends: list[float] = []
    for i, (start, duration) in enumerate(placed):
        end = start + duration
        if duration < min_cue_s:
            end = start + min_cue_s
            if i + 1 < len(placed):
                end = min(end, placed[i + 1][0])
            end = max(end, start)
        ends.append(end)

    retimed: list[Segment] = []
    placed_iter = iter(zip(placed, ends))
    for segment in transcript.segments:
        if not segment.words:
            retimed.append(Segment(words=[], speaker=segment.speaker))
            continue
        (start, _duration), end = next(placed_iter)
        words = _rescale_words(segment, start, end)
        retimed.append(Segment(words=words, speaker=segment.speaker))
    return Transcript(segments=retimed, language=transcript.language, source=transcript.source)


def _rescale_words(segment: Segment, start: float, end: float) -> list[Word]:
    """Rescale one segment's words proportionally into ``[start, end]``.

    A zero-duration segment (all words at one instant) collapses its
    words onto ``start``; the min-cue extension in
    :func:`retime_transcript` then gives the cue its readable length
    by stretching the last word's end.
    """
    original = segment.duration
    words: list[Word] = []
    if original <= 0:
        for word in segment.words:
            words.append(Word(word.text, start, start, word.confidence))
    else:
        for word in segment.words:
            offset_start = (word.start - segment.start) / original
            offset_end = (word.end - segment.start) / original
            words.append(
                Word(
                    word.text,
                    start + offset_start * (end - start),
                    start + offset_end * (end - start),
                    word.confidence,
                )
            )
    # The cue's on-screen end is the last word's end: stretch it to the
    # span end when the words fall short (min-cue extension, or a
    # zero-duration segment whose words collapsed onto the start).
    if words and words[-1].end < end:
        last = words[-1]
        words[-1] = Word(last.text, last.start, end, last.confidence)
    return words
