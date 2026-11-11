"""Render a finished dubbed video: burn subtitles into the picture.

``dub`` muxes the dubbed audio back into the video, but a dub is only
reviewable when you can *watch* it — with the translated lines on
screen, timed to the dubbed speech. :func:`render_video` does that:
it takes the original video, the dubbed audio track and a retimed
subtitle file (the SRT ``dub --subtitles`` writes) and produces a
final MP4 with the subtitles burned into the picture via ffmpeg's
``subtitles`` filter (libass), so the file plays anywhere — no player
subtitle support needed.

Two modes:

- **Full render** (:func:`render_video`): the whole video, subtitles
  burned in with a configurable :class:`SubtitleStyle`.
- **Preview** (:func:`render_preview`): a short slice (``--preview-start``
  / ``--preview-duration``) so a long dub can be spot-checked in
  seconds instead of re-encoding the whole file.

ffmpeg is an optional runtime dependency, like extraction: every entry
point raises :class:`RenderError` with a clear message when it is
missing, when the build has no ``subtitles`` filter, or when the video
has no video stream.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass

from video_dubber.extract import ExtractionError, _run, ffmpeg_available, probe

__all__ = [
    "RenderError",
    "RenderReport",
    "SubtitleStyle",
    "count_srt_cues",
    "escape_filter_path",
    "render_preview",
    "render_video",
    "subtitles_filter_available",
]

# x264 presets, slowest to fastest; "veryfast" is the default — a preview
# render should not take longer than the dub it is checking.
PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)

_COLOR_RE = re.compile(r"^&H[0-9A-Fa-f]{8}$")
_SRT_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")


class RenderError(Exception):
    """Raised when a video cannot be rendered (missing ffmpeg, bad input)."""


@dataclass(frozen=True)
class SubtitleStyle:
    """How burned-in subtitles look (ASS ``force_style`` values).

    Colours are ASS ``&HAABBGGRR`` (alpha, blue, green, red): the
    defaults are opaque white text with a black outline on a
    semi-transparent black box, readable over any footage.
    ``alignment`` follows the ASS numpad layout (2 = bottom-centre).
    """

    font_size: int = 24
    font: str | None = None  # font family name; None lets libass pick
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    outline: float = 1.0
    shadow: float = 0.0
    bold: bool = False
    alignment: int = 2
    margin_v: int = 24  # pixels from the bottom edge

    def __post_init__(self) -> None:
        if not 8 <= self.font_size <= 96:
            raise RenderError(f"font size must be 8-96, got {self.font_size}")
        if not 1 <= self.alignment <= 9:
            raise RenderError(f"alignment must be 1-9 (ASS numpad), got {self.alignment}")
        if self.margin_v < 0:
            raise RenderError(f"margin_v must be non-negative, got {self.margin_v}")
        if self.outline < 0:
            raise RenderError(f"outline must be non-negative, got {self.outline}")
        if self.shadow < 0:
            raise RenderError(f"shadow must be non-negative, got {self.shadow}")
        for name, value in (
            ("primary_color", self.primary_color),
            ("outline_color", self.outline_color),
            ("back_color", self.back_color),
        ):
            if not _COLOR_RE.match(value):
                raise RenderError(f"{name} must be ASS &HAABBGGRR, got {value!r}")

    def to_force_style(self) -> str:
        """Serialise to the ``subtitles`` filter's ``force_style`` value."""
        parts = [
            f"FontSize={self.font_size}",
            f"PrimaryColour={self.primary_color}",
            f"OutlineColour={self.outline_color}",
            f"BackColour={self.back_color}",
            f"Outline={self.outline:g}",
            f"Shadow={self.shadow:g}",
            f"Bold={1 if self.bold else 0}",
            f"Alignment={self.alignment}",
            f"MarginV={self.margin_v}",
        ]
        if self.font is not None:
            parts.append(f"FontName={self.font}")
        return ",".join(parts)


def escape_filter_path(path: str) -> str:
    """Escape a file path for use inside an ffmpeg filter argument.

    Inside a filtergraph, ``:`` separates filter options, ``,`` separates
    filters and ``'`` quotes values, so all three (plus backslashes
    themselves) must be escaped — otherwise a path with a space or a
    colon breaks the filter, or worse, silently renders the wrong file.
    """
    return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")


def subtitles_filter_available() -> bool:
    """True when this ffmpeg build ships the ``subtitles`` (libass) filter."""
    if not ffmpeg_available():
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return any(" subtitles " in line for line in result.stdout.splitlines())


def count_srt_cues(path: str) -> int:
    """Count the subtitle cues in an SRT file (one timestamp line each)."""
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if _SRT_TIMESTAMP_RE.match(line.strip()))


@dataclass(frozen=True)
class RenderReport:
    """What one render produced, for the CLI and for JSON reports."""

    out_path: str
    duration: float
    subtitle_cues: int  # cues burned in; 0 when no subtitles file was given
    filter_used: str  # "subtitles" | "none"
    video_codec: str
    crf: int
    is_preview: bool = False

    def to_dict(self) -> dict:
        return {
            "out_path": self.out_path,
            "duration_s": round(self.duration, 3),
            "subtitle_cues": self.subtitle_cues,
            "filter_used": self.filter_used,
            "video_codec": self.video_codec,
            "crf": self.crf,
            "is_preview": self.is_preview,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Video render report",
            "",
            f"- Output: {self.out_path}",
            f"- Duration: {self.duration:.2f}s" + (" (preview slice)" if self.is_preview else ""),
            f"- Subtitle cues burned in: {self.subtitle_cues} (filter: {self.filter_used})",
            f"- Video codec: {self.video_codec} (CRF {self.crf})",
        ]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> RenderReport:
        """Rebuild from :meth:`to_dict` output; malformed input raises ValueError."""
        if not isinstance(data, dict):
            raise TypeError("render report must be a JSON object")
        try:
            return cls(
                out_path=str(data["out_path"]),
                duration=float(data["duration_s"]),
                subtitle_cues=int(data["subtitle_cues"]),
                filter_used=str(data["filter_used"]),
                video_codec=str(data["video_codec"]),
                crf=int(data["crf"]),
                is_preview=bool(data.get("is_preview", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid render report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> RenderReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def render_video(
    video_path: str,
    audio_path: str,
    out_path: str,
    subtitles_path: str | None = None,
    style: SubtitleStyle | None = None,
    crf: int = 23,
    preset: str = "veryfast",
    start: float | None = None,
    duration: float | None = None,
) -> RenderReport:
    """Render a dubbed video, optionally burning subtitles into the picture.

    The video stream is re-encoded (libx264, ``crf``, ``preset``) so the
    subtitles filter can draw on it; the dubbed audio is encoded as AAC
    and the output is cut to the shorter stream (``-shortest``).
    With ``subtitles_path`` given, cues are burned in with
    ``style`` (or the default :class:`SubtitleStyle`); without it the
    video is simply re-encoded with the new audio (``filter_used`` is
    then ``"none"``).

    ``start``/``duration`` (seconds) slice the output — the preview
    path, see :func:`render_preview`. Both must be non-negative;
    ``duration`` must be positive when given.

    Raises :class:`RenderError` when a file is missing, the video has
    no video stream, ffmpeg is missing or has no ``subtitles`` filter
    (when one is needed), or ffmpeg fails.
    """
    for label, path in (("video", video_path), ("audio", audio_path)):
        if not os.path.isfile(path):
            raise RenderError(f"{label} file not found: {path}")
    if subtitles_path is not None and not os.path.isfile(subtitles_path):
        raise RenderError(f"subtitles file not found: {subtitles_path}")
    if not 0 <= crf <= 51:
        raise RenderError(f"crf must be 0-51, got {crf}")
    if preset not in PRESETS:
        raise RenderError(f"unknown x264 preset: {preset!r} (try one of {', '.join(PRESETS)})")
    if start is not None and start < 0:
        raise RenderError(f"start must be non-negative, got {start}")
    if duration is not None and duration <= 0:
        raise RenderError(f"duration must be positive, got {duration}")

    info = probe(video_path)
    if not info.has_video:
        raise RenderError(f"no video stream in {video_path}")
    style = style or SubtitleStyle()
    filter_used = "none"
    video_filter: list[str] = []
    if subtitles_path is not None:
        if not subtitles_filter_available():
            raise RenderError(
                "this ffmpeg build has no 'subtitles' filter (libass missing) — "
                "cannot burn subtitles in; install an ffmpeg with libass, or "
                "render without --subtitles"
            )
        escaped = escape_filter_path(os.path.abspath(subtitles_path))
        video_filter = [
            "-vf",
            f"subtitles='{escaped}':force_style='{style.to_force_style()}'",
        ]
        filter_used = "subtitles"

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path, "-i", audio_path]
    cmd += video_filter
    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
    ]
    if duration is None:
        # Cut to the shorter stream — but only when no explicit slice
        # length is given: with -t, the slice length bounds the output,
        # and -shortest would cut it short whenever the dubbed audio
        # ends before the slice does (a preview of a long video over a
        # short dub track came out 0.5s instead of the requested 1.0s).
        cmd.append("-shortest")
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(out_path)
    try:
        result = _run(cmd, "render")
    except ExtractionError as exc:
        raise RenderError(str(exc)) from exc
    if result.returncode != 0:
        raise RenderError(f"ffmpeg render failed for {video_path}: {result.stderr.strip()[:300]}")
    out_info = probe(out_path)
    cues = count_srt_cues(subtitles_path) if subtitles_path is not None else 0
    return RenderReport(
        out_path=out_path,
        duration=out_info.duration,
        subtitle_cues=cues,
        filter_used=filter_used,
        video_codec=out_info.video_codec or "libx264",
        crf=crf,
        is_preview=start is not None or duration is not None,
    )


def render_preview(
    video_path: str,
    audio_path: str,
    out_path: str,
    subtitles_path: str | None = None,
    style: SubtitleStyle | None = None,
    start: float = 0.0,
    duration: float = 10.0,
    crf: int = 23,
    preset: str = "veryfast",
) -> RenderReport:
    """Render a short slice of the dubbed video for a quick visual check.

    Same as :func:`render_video` with ``start``/``duration`` set — the
    slice keeps A/V in sync (the seek is applied to the output, not to
    one input), so the preview shows exactly what the full render
    shows at that timestamp. ``duration`` defaults to 10 seconds.
    """
    return render_video(
        video_path,
        audio_path,
        out_path,
        subtitles_path=subtitles_path,
        style=style,
        crf=crf,
        preset=preset,
        start=start,
        duration=duration,
    )
