"""Subtitle QA: readability checks over the cues a dub will show on screen.

The assembler, the splitter and the aligner all move cues around — a
line can end up too fast to read, too long for the screen, or
overlapping the next cue. This module checks the finished transcript
against the usual broadcast readability rules *before* the video is
muxed, when a fix is still cheap:

* **too-fast** — characters per second above ``max_cps`` (20 by
  default; most broadcasters cap reading speed at 17–20 cps).
* **too-many-lines** — the text wraps to more than ``max_lines`` lines
  at ``max_line_chars`` characters per line (42 by default, two lines).
* **long-word** — a single word longer than one line, so no wrap can
  save it; it will overflow the frame or be cut.
* **too-short** — a cue shorter than ``min_cue_s`` (1 s); the eye never
  gets to settle on it.
* **too-long** — a cue longer than ``max_cue_s`` (7 s); it sits on
  screen long after the viewer has finished reading it.
* **overlap** — a cue starting before the previous one ends; the
  player shows two lines at once (an *error*, everything else is a
  *warning*).
* **tiny-gap** — less than ``min_gap_s`` (80 ms, two frames at 25 fps)
  between cues; the viewer never sees the line change.

The verdict is ``pass`` when there are no issues at all, ``review``
otherwise. Wordless segments are skipped everywhere else in the
pipeline — they produce no cue — so they are skipped here too and
reported in the header count, not as issues.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field

from video_dubber.transcript import Transcript

__all__ = [
    "DEFAULT_MAX_CPS",
    "DEFAULT_MAX_CUE_S",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_LINE_CHARS",
    "DEFAULT_MIN_CUE_S",
    "DEFAULT_MIN_GAP_S",
    "QaConfig",
    "QaError",
    "QaIssue",
    "QaReport",
    "qa_transcript",
]

DEFAULT_MAX_CPS = 20.0  # broadcast reading-speed cap, chars per second
DEFAULT_MAX_LINE_CHARS = 42  # safe line width for a 16:9 frame
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_CUE_S = 1.0
DEFAULT_MAX_CUE_S = 7.0
DEFAULT_MIN_GAP_S = 0.08  # two frames at 25 fps


class QaError(Exception):
    """Raised when a QA run cannot start (bad parameters, empty input)."""


@dataclass(frozen=True)
class QaConfig:
    """Tuning knobs for :func:`qa_transcript`."""

    max_cps: float = DEFAULT_MAX_CPS
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS
    max_lines: int = DEFAULT_MAX_LINES
    min_cue_s: float = DEFAULT_MIN_CUE_S
    max_cue_s: float = DEFAULT_MAX_CUE_S
    min_gap_s: float = DEFAULT_MIN_GAP_S


@dataclass(frozen=True)
class QaIssue:
    """One readability problem in one cue.

    ``kind`` is one of ``too-fast``, ``too-many-lines``, ``long-word``,
    ``too-short``, ``too-long``, ``overlap`` or ``tiny-gap``.
    ``severity`` is ``error`` for an overlap (two cues on screen at
    once) and ``warning`` for everything else.
    """

    index: int  # 1-based segment number in the transcript
    start: float
    end: float
    kind: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict) -> QaIssue:
        """Rebuild from :meth:`to_dict` output."""
        try:
            return cls(
                index=int(data["index"]),
                start=float(data["start"]),
                end=float(data["end"]),
                kind=str(data["kind"]),
                severity=str(data["severity"]),
                detail=str(data["detail"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid QA issue: {exc}") from exc


@dataclass
class QaReport:
    """What the readability checks found (JSON + Markdown)."""

    max_cps: float
    max_line_chars: int
    max_lines: int
    min_cue_s: float
    max_cue_s: float
    min_gap_s: float
    segments_checked: int
    segments_skipped: int
    errors: int
    warnings: int
    verdict: str  # "pass" or "review"
    issues: list[QaIssue] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "max_cps": self.max_cps,
                "max_line_chars": self.max_line_chars,
                "max_lines": self.max_lines,
                "min_cue_s": self.min_cue_s,
                "max_cue_s": self.max_cue_s,
                "min_gap_s": self.min_gap_s,
                "segments_checked": self.segments_checked,
                "segments_skipped": self.segments_skipped,
                "errors": self.errors,
                "warnings": self.warnings,
                "verdict": self.verdict,
                "issues": [issue.to_dict() for issue in self.issues],
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [
            "# Subtitle QA report",
            "",
            f"- Verdict: **{self.verdict}**",
            f"- Cues checked: {self.segments_checked}",
            f"- Wordless segments skipped: {self.segments_skipped}",
            f"- Errors: {self.errors}, warnings: {self.warnings}",
            f"- Reading speed cap: {self.max_cps:.1f} chars/s",
            f"- Line width: {self.max_line_chars} chars, max {self.max_lines} lines",
            (
                f"- Cue length: {self.min_cue_s:.2f}s – {self.max_cue_s:.2f}s, "
                f"min gap {self.min_gap_s * 1000:.0f} ms"
            ),
        ]
        if not self.issues:
            lines.append("- No issues found.")
        for issue in self.issues:
            lines.append(
                f"- Segment {issue.index} ({issue.start:.2f}-"
                f"{issue.end:.2f}s) [{issue.severity}] {issue.kind}: {issue.detail}"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, data: dict) -> QaReport:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("QA report must be a JSON object")
        try:
            return cls(
                max_cps=float(data["max_cps"]),
                max_line_chars=int(data["max_line_chars"]),
                max_lines=int(data["max_lines"]),
                min_cue_s=float(data["min_cue_s"]),
                max_cue_s=float(data["max_cue_s"]),
                min_gap_s=float(data["min_gap_s"]),
                segments_checked=int(data["segments_checked"]),
                segments_skipped=int(data["segments_skipped"]),
                errors=int(data["errors"]),
                warnings=int(data["warnings"]),
                verdict=str(data["verdict"]),
                issues=[QaIssue.from_dict(row) for row in data.get("issues", [])],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid QA report: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> QaReport:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def _wrap_lines(text: str, width: int) -> list[str]:
    """Word-aware wrap; a word longer than ``width`` stays whole."""
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def qa_transcript(
    transcript: Transcript,
    config: QaConfig | None = None,
) -> QaReport:
    """Run the readability checks over a transcript's cues.

    Returns a :class:`QaReport`; raises :class:`QaError` on a bad
    config or a transcript with no timed segments.
    """
    config = config or QaConfig()
    if config.max_cps <= 0:
        raise QaError(f"max_cps must be positive, got {config.max_cps}")
    if config.max_line_chars < 1:
        raise QaError(f"max_line_chars must be at least 1, got {config.max_line_chars}")
    if config.max_lines < 1:
        raise QaError(f"max_lines must be at least 1, got {config.max_lines}")
    if config.min_cue_s < 0:
        raise QaError(f"min_cue_s must be non-negative, got {config.min_cue_s}")
    if config.max_cue_s <= 0 or config.max_cue_s < config.min_cue_s:
        raise QaError(f"max_cue_s must be positive and >= min_cue_s, got {config.max_cue_s}")
    if config.min_gap_s < 0:
        raise QaError(f"min_gap_s must be non-negative, got {config.min_gap_s}")

    timed = [segment for segment in transcript.segments if segment.words]
    if not timed:
        raise QaError("cannot QA a transcript with no timed segments")

    issues: list[QaIssue] = []
    prev_end: float | None = None
    for i, segment in enumerate(timed, start=1):
        start, end = segment.start, segment.end
        duration = end - start
        text = segment.text

        # Ordering between cues: overlap is the only error.
        if prev_end is not None:
            gap = start - prev_end
            if gap < 0:
                issues.append(
                    QaIssue(
                        index=i,
                        start=start,
                        end=end,
                        kind="overlap",
                        severity="error",
                        detail=f"starts {abs(gap) * 1000:.0f} ms before the previous cue ends",
                    )
                )
            elif gap < config.min_gap_s:
                issues.append(
                    QaIssue(
                        index=i,
                        start=start,
                        end=end,
                        kind="tiny-gap",
                        severity="warning",
                        detail=(
                            f"only {gap * 1000:.0f} ms after the previous cue "
                            f"(min {config.min_gap_s * 1000:.0f} ms)"
                        ),
                    )
                )
        prev_end = max(prev_end, end) if prev_end is not None else end

        if duration < config.min_cue_s:
            issues.append(
                QaIssue(
                    index=i,
                    start=start,
                    end=end,
                    kind="too-short",
                    severity="warning",
                    detail=f"cue lasts {duration:.2f}s (min {config.min_cue_s:.2f}s)",
                )
            )
        if duration > config.max_cue_s:
            issues.append(
                QaIssue(
                    index=i,
                    start=start,
                    end=end,
                    kind="too-long",
                    severity="warning",
                    detail=f"cue lasts {duration:.2f}s (max {config.max_cue_s:.2f}s)",
                )
            )
        if duration > 0:
            cps = len(text) / duration
            if cps > config.max_cps:
                issues.append(
                    QaIssue(
                        index=i,
                        start=start,
                        end=end,
                        kind="too-fast",
                        severity="warning",
                        detail=(
                            f"{cps:.1f} chars/s over {len(text)} chars in "
                            f"{duration:.2f}s (max {config.max_cps:.1f})"
                        ),
                    )
                )
        if any(len(word) > config.max_line_chars for word in text.split()):
            issues.append(
                QaIssue(
                    index=i,
                    start=start,
                    end=end,
                    kind="long-word",
                    severity="warning",
                    detail=(
                        f"a word is longer than one {config.max_line_chars}-char line "
                        "and cannot be wrapped"
                    ),
                )
            )
        lines = _wrap_lines(text, config.max_line_chars)
        if len(lines) > config.max_lines:
            issues.append(
                QaIssue(
                    index=i,
                    start=start,
                    end=end,
                    kind="too-many-lines",
                    severity="warning",
                    detail=(
                        f"wraps to {len(lines)} lines at "
                        f"{config.max_line_chars} chars (max {config.max_lines})"
                    ),
                )
            )

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = len(issues) - errors
    return QaReport(
        max_cps=config.max_cps,
        max_line_chars=config.max_line_chars,
        max_lines=config.max_lines,
        min_cue_s=config.min_cue_s,
        max_cue_s=config.max_cue_s,
        min_gap_s=config.min_gap_s,
        segments_checked=len(timed),
        segments_skipped=len(transcript.segments) - len(timed),
        errors=errors,
        warnings=warnings,
        verdict="pass" if not issues else "review",
        issues=issues,
    )
