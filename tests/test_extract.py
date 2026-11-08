import os
import subprocess

import pytest

from video_dubber.audio import load_wav, save_wav
from video_dubber.extract import (
    ExtractionError,
    extract_audio,
    extract_audio_data,
    ffmpeg_available,
    probe,
)

ffmpeg_missing = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")


def _make_test_video(path: str, duration: float = 1.0) -> None:
    """Generate a tiny MP4 (colour bars + sine tone) with ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s=64x64:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            path,
        ],
        check=True,
    )


@ffmpeg_missing
def test_probe_video(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _make_test_video(video, duration=1.0)
    info = probe(video)
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert info.audio_codec == "aac"
    assert info.video_codec == "h264"
    assert info.width == 64 and info.height == 64


@ffmpeg_missing
def test_extract_audio_round_trip(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _make_test_video(video, duration=1.0)
    wav = str(tmp_path / "audio.wav")
    assert extract_audio(video, wav) == wav
    audio = load_wav(wav)
    assert audio.sample_rate == 16_000
    assert audio.duration == pytest.approx(1.0, abs=0.05)
    assert audio.peak > 0.1  # the sine tone survived


@ffmpeg_missing
def test_extract_audio_data_in_memory(tmp_path):
    video = str(tmp_path / "clip.mp4")
    _make_test_video(video, duration=0.5)
    audio = extract_audio_data(video)
    assert audio.duration == pytest.approx(0.5, abs=0.05)
    assert audio.peak > 0.1


@ffmpeg_missing
def test_probe_missing_file_raises():
    with pytest.raises(ExtractionError, match="ffprobe failed"):
        probe("/definitely/not/here.mp4")


@ffmpeg_missing
def test_extract_silent_video_raises(tmp_path):
    video = str(tmp_path / "silent.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=0.5",
            "-t",
            "0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video,
        ],
        check=True,
    )
    with pytest.raises(ExtractionError, match="no audio stream"):
        extract_audio(video, str(tmp_path / "out.wav"))


def test_probe_without_ffmpeg(monkeypatch):
    monkeypatch.setattr("video_dubber.extract.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "video_dubber.extract.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("ffprobe")),
    )
    with pytest.raises(ExtractionError, match="needs ffmpeg"):
        probe("anything.mp4")


def test_probe_invalid_json(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = "not json"
        stderr = ""

    monkeypatch.setattr("video_dubber.extract.subprocess.run", lambda *a, **k: FakeResult())
    with pytest.raises(ExtractionError, match="invalid JSON"):
        probe("anything.mp4")


def test_extract_audio_data_uses_wav_loader(tmp_path, monkeypatch):
    """extract_audio_data loads whatever extract_audio wrote (no ffmpeg needed)."""
    wav = tmp_path / "tone.wav"
    from video_dubber.audio import make_sine

    save_wav(str(wav), make_sine(440.0, 0.25))
    monkeypatch.setattr(
        "video_dubber.extract.extract_audio",
        lambda video, out, sample_rate=16_000: os.replace(str(wav), out),
    )
    audio = extract_audio_data("fake.mp4")
    assert audio.duration == pytest.approx(0.25)
