"""Video/audio extraction via ffmpeg + ffprobe.

Extracts a video's audio track to 16 kHz mono WAV (the pipeline's internal
format) and probes media files for duration and stream info. ffmpeg is an
optional runtime dependency: every entry point raises
:class:`ExtractionError` with a clear message when it is missing, so the
rest of the package (WAV loading, VAD, transcripts) keeps working without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from video_dubber.audio import DEFAULT_SAMPLE_RATE, AudioData, load_wav


class ExtractionError(RuntimeError):
    """Raised when probing or extracting fails (missing binary, bad file)."""


def ffmpeg_available() -> bool:
    """True when both ffmpeg and ffprobe are on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ExtractionError(f"{what} needs ffmpeg/ffprobe on PATH (not found)") from exc


@dataclass(frozen=True)
class ProbeResult:
    """Media-file facts from ffprobe."""

    duration: float
    has_audio: bool
    has_video: bool
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None


def probe(path: str) -> ProbeResult:
    """Probe a media file with ffprobe (JSON). Raises ExtractionError on failure."""
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            path,
        ],
        "probe",
    )
    if result.returncode != 0:
        raise ExtractionError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if duration == 0.0 and audio is not None:
        duration = float(audio.get("duration") or 0.0)
    return ProbeResult(
        duration=duration,
        has_audio=audio is not None,
        has_video=video is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        audio_channels=int(audio["channels"]) if audio and audio.get("channels") else None,
        video_codec=video.get("codec_name") if video else None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
    )


def extract_audio(
    video_path: str,
    out_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Extract a video's audio track to a 16-bit mono WAV at ``out_path``.

    Returns ``out_path``. Raises ExtractionError if ffmpeg fails or the
    input has no audio stream.
    """
    info = probe(video_path)
    if not info.has_audio:
        raise ExtractionError(f"no audio stream in {video_path}")
    result = _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            out_path,
        ],
        "extract_audio",
    )
    if result.returncode != 0:
        raise ExtractionError(f"ffmpeg failed for {video_path}: {result.stderr.strip()}")
    return out_path


def extract_audio_data(video_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioData:
    """Extract a video's audio track straight into memory as :class:`AudioData`.

    Writes through a temporary WAV file (ffmpeg needs a file target).
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="video-dubber-") as tmpdir:
        wav_path = str(Path(tmpdir) / "audio.wav")
        extract_audio(video_path, wav_path, sample_rate=sample_rate)
        return load_wav(wav_path)
