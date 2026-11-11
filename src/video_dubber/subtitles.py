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

import re
from dataclasses import dataclass, field

from video_dubber.transcript import Segment, Transcript, Word

__all__ = [
    "AssStyle",
    "SubtitleError",
    "assign_speaker_styles",
    "retime_transcript",
    "to_ass",
]

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


# --- Advanced SubStation Alpha (ASS) export ---------------------------------
# SRT/VTT carry one cue per segment and no styling at all, which is fine
# for a plain player but useless for a styled render: `render` burns
# subtitles with a single ASS style for every line, so a two-speaker dub
# reads as one voice. `to_ass` writes a real .ass file — one named style
# per speaker, each with its own colour — which ffmpeg's `subtitles`
# filter (libass) and any ASS-capable player render directly.

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

# Distinct, readable-on-video colours handed out to speakers in order.
SPEAKER_PALETTE: tuple[str, ...] = (
    "#FFFFFF",  # white — the default voice
    "#FFFF00",  # yellow
    "#00FFFF",  # cyan
    "#00FF00",  # green
    "#FF00FF",  # magenta
    "#FFA500",  # orange
    "#FF8080",  # light red
    "#80FF80",  # light green
)


@dataclass(frozen=True)
class AssStyle:
    """One ASS style: a named look for a speaker's cues.

    Colours are ``#RRGGBB`` (the web form) and are converted to ASS's
    ``&HAABBGGRR`` on output, with full opacity (``AA = 00``).
    ``alignment`` is the ASS numpad position (2 = bottom-centre, the
    usual subtitle spot). All values are validated — a bad colour or
    an alignment outside 1–9 raises :class:`SubtitleError` at
    construction, never halfway through a file write.
    """

    name: str
    font: str = "Arial"
    size: int = 20
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#FF0000"
    outline_color: str = "#000000"
    back_color: str = "#000000"
    bold: bool = False
    italic: bool = False
    alignment: int = 2
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 10
    outline: float = 1.0
    shadow: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise SubtitleError("ASS style name must not be empty")
        if "," in self.name:
            raise SubtitleError(f"ASS style name must not contain a comma: {self.name!r}")
        for label, color in (
            ("primary_color", self.primary_color),
            ("secondary_color", self.secondary_color),
            ("outline_color", self.outline_color),
            ("back_color", self.back_color),
        ):
            if not _HEX_COLOR.match(color):
                raise SubtitleError(
                    f"ASS {label} must be #RRGGBB, got {color!r} (style {self.name!r})"
                )
        if self.size <= 0:
            raise SubtitleError(f"ASS font size must be positive, got {self.size}")
        if not 1 <= self.alignment <= 9:
            raise SubtitleError(f"ASS alignment must be 1-9 (numpad), got {self.alignment}")
        if self.outline < 0 or self.shadow < 0:
            raise SubtitleError(
                f"ASS outline/shadow must be non-negative, got {self.outline}/{self.shadow}"
            )
        if min(self.margin_l, self.margin_r, self.margin_v) < 0:
            raise SubtitleError("ASS margins must be non-negative")

    @staticmethod
    def _ass_color(hex_color: str) -> str:
        """``#RRGGBB`` -> ASS ``&H00BBGGRR`` (alpha 00 = fully opaque)."""
        red, green, blue = hex_color[1:3], hex_color[3:5], hex_color[5:7]
        return f"&H00{blue}{green}{red}".upper()

    def to_ass_line(self) -> str:
        """Render this style as one ``Style:`` line for the [V4+ Styles] section."""
        fields = [
            self.name,
            self.font,
            str(self.size),
            self._ass_color(self.primary_color),
            self._ass_color(self.secondary_color),
            self._ass_color(self.outline_color),
            self._ass_color(self.back_color),
            "-1" if self.bold else "0",
            "-1" if self.italic else "0",
            "0",  # underline
            "0",  # strikeout
            "100",  # scale X
            "100",  # scale Y
            "0",  # spacing
            "0",  # angle
            "1",  # border style (outline + shadow)
            f"{self.outline:g}",
            f"{self.shadow:g}",
            str(self.alignment),
            str(self.margin_l),
            str(self.margin_r),
            str(self.margin_v),
            "1",  # encoding (ANSI)
        ]
        return "Style: " + ",".join(fields)
