import json
import subprocess
from types import SimpleNamespace

import pytest

from video_dubber.audio import AudioData, save_wav
from video_dubber.extract import ProbeResult, ffmpeg_available
from video_dubber.render import (
    RenderError,
    RenderReport,
    SubtitleStyle,
    count_srt_cues,
    escape_filter_path,
    render_preview,
    render_video,
    subtitles_filter_available,
)

ffmpeg_missing = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")
subtitles_missing = pytest.mark.skipif(
    not subtitles_filter_available(), reason="ffmpeg build has no subtitles filter"
)

SRT = """1
00:00:00,000 --> 00:00:01,000
Hello world

2
00:00:01,500 --> 00:00:02,500
Welcome to Mars
"""


def _make_test_video(path: str, duration: float = 2.0) -> None:
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


def _make_wav(path: str, seconds: float = 2.0, freq: float = 440.0) -> None:
    import math

    rate = 16_000
    samples = [0.3 * math.sin(2 * math.pi * freq * i / rate) for i in range(int(seconds * rate))]
    save_wav(path, AudioData(samples, rate))


def _video_probe(monkeypatch):
    """Fake probe: input has video+audio, output is 2.0s of h264."""

    def fake_probe(path):
        if path.endswith("out.mp4"):
            return ProbeResult(duration=2.0, has_audio=True, has_video=True, video_codec="h264")
        return ProbeResult(duration=2.0, has_audio=True, has_video=True, video_codec="h264")

    monkeypatch.setattr("video_dubber.render.probe", fake_probe)


def _ok_run(monkeypatch, calls):
    def fake_run(cmd, what):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("video_dubber.render._run", fake_run)


# --- SubtitleStyle ------------------------------------------------------------


def test_style_defaults_produce_force_style():
    style = SubtitleStyle()
    text = style.to_force_style()
    assert "FontSize=24" in text
    assert "PrimaryColour=&H00FFFFFF" in text
    assert "Alignment=2" in text
    assert "MarginV=24" in text
    assert "Bold=0" in text
    assert "FontName" not in text


def test_style_font_and_bold_in_force_style():
    style = SubtitleStyle(font="DejaVu Sans", bold=True, font_size=18, alignment=1, margin_v=0)
    text = style.to_force_style()
    assert "FontName=DejaVu Sans" in text
    assert "Bold=1" in text
    assert "FontSize=18" in text
    assert "Alignment=1" in text
    assert "MarginV=0" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"font_size": 7},
        {"font_size": 97},
        {"alignment": 0},
        {"alignment": 10},
        {"margin_v": -1},
        {"outline": -0.5},
        {"shadow": -1},
        {"primary_color": "white"},
        {"primary_color": "&HFFFFFF"},
        {"back_color": "&HZZ000000"},
    ],
)
def test_style_validation_rejects_bad_values(kwargs):
    with pytest.raises(RenderError):
        SubtitleStyle(**kwargs)


# --- Path escaping + SRT counting ----------------------------------------------


def test_escape_filter_path():
    # Backslashes, quotes, colons and commas all break a filtergraph raw.
    assert escape_filter_path("/tmp/my file.srt") == "/tmp/my file.srt"
    assert escape_filter_path("/tmp/a:b,c'd.srt") == "/tmp/a\\:b\\,c\\'d.srt"
    assert escape_filter_path("C:\\vid\\out.srt") == "C\\:\\\\vid\\\\out.srt"


def test_count_srt_cues(tmp_path):
    srt = tmp_path / "subs.srt"
    srt.write_text(SRT, encoding="utf-8")
    assert count_srt_cues(str(srt)) == 2
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    assert count_srt_cues(str(empty)) == 0


# --- render_video command building (mocked ffmpeg) ----------------------------


def test_render_video_builds_subtitles_command(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    srt = tmp_path / "subs.srt"
    srt.write_text(SRT, encoding="utf-8")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    monkeypatch.setattr("video_dubber.render.subtitles_filter_available", lambda: True)
    calls: list = []
    _ok_run(monkeypatch, calls)

    report = render_video(video, audio, str(tmp_path / "out.mp4"), subtitles_path=str(srt))

    assert len(calls) == 1
    cmd = calls[0]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("subtitles='")
    assert "force_style='FontSize=24" in vf
    assert "-c:v" in cmd and "libx264" in cmd
    assert cmd[cmd.index("-crf") + 1] == "23"
    assert "-shortest" in cmd
    assert report.filter_used == "subtitles"
    assert report.subtitle_cues == 2
    assert report.video_codec == "h264"
    assert report.duration == pytest.approx(2.0)


def test_render_video_without_subtitles_skips_filter(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    calls: list = []
    _ok_run(monkeypatch, calls)

    report = render_video(video, audio, str(tmp_path / "out.mp4"), crf=18, preset="fast")

    cmd = calls[0]
    assert "-vf" not in cmd
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert report.filter_used == "none"
    assert report.subtitle_cues == 0


def test_render_preview_slices_output(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    calls: list = []
    _ok_run(monkeypatch, calls)

    report = render_preview(video, audio, str(tmp_path / "out.mp4"), start=5.0, duration=10.0)

    cmd = calls[0]
    # The seek is an output option (after the inputs) so audio and video
    # stay in sync — it must not appear before the first -i.
    assert cmd.index("-ss") > cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "5.0"
    assert cmd[cmd.index("-t") + 1] == "10.0"
    # A sliced render is bounded by -t, not -shortest: -shortest would
    # cut the slice short when the dub audio ends before the slice does.
    assert "-shortest" not in cmd
    assert report.is_preview is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crf": -1},
        {"crf": 52},
        {"preset": "turbo"},
        {"start": -1.0},
        {"duration": 0.0},
    ],
)
def test_render_video_validates_options(tmp_path, monkeypatch, kwargs):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    with pytest.raises(RenderError):
        render_video(video, audio, str(tmp_path / "out.mp4"), **kwargs)


def test_render_video_missing_files(tmp_path, monkeypatch):
    _video_probe(monkeypatch)
    with pytest.raises(RenderError, match="video file not found"):
        render_video(str(tmp_path / "nope.mp4"), str(tmp_path / "dub.wav"), str(tmp_path / "o.mp4"))
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    open(video, "wb").close()
    with pytest.raises(RenderError, match="audio file not found"):
        render_video(video, str(tmp_path / "nope.wav"), str(tmp_path / "o.mp4"))
    open(audio, "wb").close()
    with pytest.raises(RenderError, match="subtitles file not found"):
        render_video(video, audio, str(tmp_path / "o.mp4"), subtitles_path=str(tmp_path / "no.srt"))


def test_render_video_rejects_audio_only_input(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    for path in (video, audio):
        open(path, "wb").close()
    monkeypatch.setattr(
        "video_dubber.render.probe",
        lambda path: ProbeResult(duration=2.0, has_audio=True, has_video=False),
    )
    with pytest.raises(RenderError, match="no video stream"):
        render_video(video, audio, str(tmp_path / "out.mp4"))


def test_render_video_needs_subtitles_filter(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    srt = tmp_path / "subs.srt"
    srt.write_text(SRT, encoding="utf-8")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    monkeypatch.setattr("video_dubber.render.subtitles_filter_available", lambda: False)
    with pytest.raises(RenderError, match="no 'subtitles' filter"):
        render_video(video, audio, str(tmp_path / "out.mp4"), subtitles_path=str(srt))


def test_render_video_ffmpeg_failure(tmp_path, monkeypatch):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    for path in (video, audio):
        open(path, "wb").close()
    _video_probe(monkeypatch)
    monkeypatch.setattr(
        "video_dubber.render._run",
        lambda cmd, what: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RenderError, match="ffmpeg render failed"):
        render_video(video, audio, str(tmp_path / "out.mp4"))


# --- RenderReport round-trip ---------------------------------------------------


def test_render_report_round_trip():
    report = RenderReport(
        out_path="dub.mp4",
        duration=12.5,
        subtitle_cues=4,
        filter_used="subtitles",
        video_codec="h264",
        crf=23,
        is_preview=True,
    )
    restored = RenderReport.from_json(report.to_json())
    assert restored == report
    assert "preview slice" in report.to_markdown()
    assert "Subtitle cues burned in: 4" in report.to_markdown()


def test_render_report_malformed():
    with pytest.raises(ValueError):
        RenderReport.from_json("{not json")
    with pytest.raises(ValueError):
        RenderReport.from_dict({"out_path": "x.mp4"})
    with pytest.raises(ValueError):
        RenderReport.from_json(json.dumps(["not", "a", "dict"]))


# --- Live end-to-end (real ffmpeg) ---------------------------------------------


@ffmpeg_missing
@subtitles_missing
def test_render_video_live_burns_subtitles(tmp_path):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    srt = tmp_path / "subs.srt"
    srt.write_text(SRT, encoding="utf-8")
    _make_test_video(video, duration=2.0)
    _make_wav(audio, seconds=2.0)
    out = str(tmp_path / "rendered.mp4")

    report = render_video(video, audio, out, subtitles_path=str(srt))

    assert report.subtitle_cues == 2
    assert report.filter_used == "subtitles"
    assert report.duration == pytest.approx(2.0, abs=0.15)
    from video_dubber.extract import probe

    info = probe(out)
    assert info.has_video and info.has_audio
    assert info.video_codec == "h264"


@ffmpeg_missing
def test_render_preview_live_slices(tmp_path):
    video = str(tmp_path / "clip.mp4")
    audio = str(tmp_path / "dub.wav")
    _make_test_video(video, duration=4.0)
    _make_wav(audio, seconds=4.0)
    out = str(tmp_path / "preview.mp4")

    report = render_preview(video, audio, out, start=1.0, duration=1.0)

    assert report.is_preview is True
    assert report.duration == pytest.approx(1.0, abs=0.15)
