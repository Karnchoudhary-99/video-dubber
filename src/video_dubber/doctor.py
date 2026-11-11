"""Environment diagnostics: is this machine ready to dub?

The pipeline shells out to ffmpeg, ffprobe, whisper and argos-translate,
and a missing binary fails a run halfway through — after the
transcription, or after the dub, with an error that names the tool but
not the fix. ``dubber doctor`` checks everything up front and prints
one line per check with a plain-language verdict, so a fresh machine
gets a shopping list instead of a stack trace.

Each check is ``required`` or optional: a missing ffmpeg fails the
run (nothing in the pipeline works without it), a missing whisper only
warns (the scripted backend still demos the whole pipeline). The
report's verdict is ``ok``, ``ok_with_warnings`` (something optional
is missing), or ``fail`` (something required is missing).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field


class DoctorError(ValueError):
    """Raised for invalid doctor configuration or input."""


@dataclass(frozen=True)
class DoctorCheck:
    """One environment check: what was probed, and how it came out."""

    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    required: bool = True

    def __post_init__(self) -> None:
        if self.status not in ("pass", "warn", "fail"):
            raise DoctorError(f"check status must be pass/warn/fail, got {self.status!r}")


def _which(command: str) -> str | None:
    """Resolve ``command`` on PATH, or None when it is not installed."""
    return shutil.which(command)


def _first_line(command: list[str]) -> str | None:
    """First stdout line of ``command``, or None when it cannot run."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def _check_python() -> DoctorCheck:
    version = sys.version_info
    if version >= (3, 11):
        return DoctorCheck(
            name="python",
            status="pass",
            detail=f"Python {version.major}.{version.minor}.{version.micro} (>= 3.11)",
        )
    return DoctorCheck(
        name="python",
        status="fail",
        detail=f"Python {version.major}.{version.minor}.{version.micro} — needs 3.11+",
    )


def _check_binary(command: str, args: list[str], required: bool, hint: str) -> DoctorCheck:
    path = _which(command)
    if path is None:
        return DoctorCheck(
            name=command,
            status="fail" if required else "warn",
            detail=f"not found on PATH ({hint})",
            required=required,
        )
    version = _first_line([command, *args])
    detail = f"{version} ({path})" if version else f"found at {path} (version unreadable)"
    return DoctorCheck(name=command, status="pass", detail=detail, required=required)


def _check_subtitles_filter() -> DoctorCheck:
    """ffmpeg's ``subtitles`` filter is what ``render`` burns cues with."""
    if _which("ffmpeg") is None:
        return DoctorCheck(
            name="ffmpeg subtitles filter",
            status="fail",
            detail="cannot check without ffmpeg",
        )
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return DoctorCheck(
            name="ffmpeg subtitles filter",
            status="fail",
            detail="ffmpeg -filters did not run",
        )
    # Filter lines look like " .. subtitles   V->V  Render text subtitles ..."
    # — the name is the second whitespace-separated token.
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "subtitles":
            return DoctorCheck(
                name="ffmpeg subtitles filter",
                status="pass",
                detail="libass subtitles filter available (render can burn cues in)",
            )
    return DoctorCheck(
        name="ffmpeg subtitles filter",
        status="fail",
        detail="ffmpeg has no subtitles filter — render --subtitles will fail",
    )


def _check_disk_space(path: str, minimum_gib: float = 1.0) -> DoctorCheck:
    """Free space where dubs land; a full disk fails a render mid-write."""
    if minimum_gib <= 0:
        raise DoctorError(f"minimum free space must be positive, got {minimum_gib}")
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return DoctorCheck(
            name="disk space",
            status="warn",
            detail=f"cannot measure free space under {path!r}: {exc}",
            required=False,
        )
    free_gib = usage.free / (1024**3)
    if free_gib >= minimum_gib:
        return DoctorCheck(
            name="disk space",
            status="pass",
            detail=f"{free_gib:.1f} GiB free under {path} (>= {minimum_gib:.1f} GiB)",
            required=False,
        )
    return DoctorCheck(
        name="disk space",
        status="warn",
        detail=f"only {free_gib:.1f} GiB free under {path} (< {minimum_gib:.1f} GiB)",
        required=False,
    )
