import json
import subprocess

import pytest

from video_dubber.audio import concat, make_silence, make_sine, save_wav
from video_dubber.cli import main
from video_dubber.extract import ffmpeg_available
from video_dubber.transcript import Transcript

ffmpeg_missing = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not on PATH")


def _make_test_video(path: str, duration: float = 1.0) -> None:
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


def _make_wav(path, with_speech: bool = True):
    audio = (
        concat(make_sine(440.0, 0.5), make_silence(0.5), make_sine(440.0, 0.5))
        if with_speech
        else make_silence(1.0)
    )
    save_wav(str(path), audio)


def test_help_and_version(capsys):
    assert main([]) == 0
    assert "video-dubber" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


@ffmpeg_missing
def test_cli_extract(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video))
    out = tmp_path / "audio.wav"
    assert main(["extract", str(video), "--out", str(out)]) == 0
    assert out.exists()
    # AAC priming pads the track slightly, so duration is ~1.02s, not 1.00s.
    assert "16000 Hz mono" in capsys.readouterr().out


def test_cli_extract_missing_video(tmp_path, capsys):
    out = tmp_path / "audio.wav"
    assert main(["extract", "nope.mp4", "--out", str(out)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_transcribe_scripted(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    _make_wav(wav)
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            [
                {"text": "hello", "start": 0.1, "end": 0.4},
                {"text": "world", "start": 0.6, "end": 0.9},
            ]
        )
    )
    out = tmp_path / "transcript.json"
    rc = main(
        [
            "transcribe",
            str(wav),
            "--backend",
            "scripted",
            "--script",
            str(script),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    transcript = Transcript.from_json(out.read_text())
    assert transcript.text == "hello world"
    assert "2 words" in capsys.readouterr().out


def test_cli_transcribe_scripted_needs_script(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    _make_wav(wav)
    assert main(["transcribe", str(wav), "--backend", "scripted"]) == 2
    assert "needs --script" in capsys.readouterr().err


def test_cli_transcribe_align_drops_silence_words(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    _make_wav(wav)  # speech 0-0.5s, silence 0.5-1.0s, speech 1.0-1.5s
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            [
                {"text": "first", "start": 0.1, "end": 0.4},
                {"text": "ghost", "start": 0.7, "end": 0.9},
                {"text": "second", "start": 1.1, "end": 1.4},
            ]
        )
    )
    assert (
        main(
            [
                "transcribe",
                str(wav),
                "--backend",
                "scripted",
                "--script",
                str(script),
                "--align",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "ghost" not in out
    assert "first" in out and "second" in out


def test_cli_transcribe_srt_out(tmp_path):
    wav = tmp_path / "clip.wav"
    _make_wav(wav)
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "hi", "start": 0.0, "end": 0.3}]))
    out = tmp_path / "subs.srt"
    assert (
        main(
            [
                "transcribe",
                str(wav),
                "--backend",
                "scripted",
                "--script",
                str(script),
                "--out",
                str(out),
                "--format",
                "srt",
            ]
        )
        == 0
    )
    assert out.read_text().startswith("1\n00:00:00,000 --> 00:00:00,300\nhi")


def test_cli_vad(tmp_path, capsys):
    wav = tmp_path / "clip.wav"
    _make_wav(wav)
    assert main(["vad", str(wav)]) == 0
    out = capsys.readouterr().out
    assert "2 speech segments" in out
    assert "speech ratio" in out


def test_cli_convert(tmp_path, capsys):
    transcript = tmp_path / "t.json"
    transcript.write_text(
        Transcript.from_json(
            json.dumps(
                {
                    "segments": [
                        {
                            "words": [{"text": "hi", "start": 0.0, "end": 0.3}],
                            "speaker": None,
                        }
                    ],
                    "language": "en",
                    "source": None,
                }
            )
        ).to_json()
    )
    assert main(["convert", str(transcript), "--format", "srt"]) == 0
    assert capsys.readouterr().out.startswith("1\n00:00:00,000 --> 00:00:00,300\nhi")
    out = tmp_path / "subs.vtt"
    assert main(["convert", str(transcript), "--format", "vtt", "--out", str(out)]) == 0
    assert out.read_text().startswith("WEBVTT")


def test_cli_convert_bad_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    assert main(["convert", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_unknown_command(capsys):
    with pytest.raises(SystemExit):
        main(["frobnicate"])
