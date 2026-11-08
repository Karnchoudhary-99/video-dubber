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


def _write_transcript(path, language="en"):
    path.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "Hello", "start": 0.0, "end": 0.5},
                        {"text": "world", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": None,
                }
            ],
            language=language,
        ).to_json()
    )


def test_cli_translate_to_hindi_json(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    out = tmp_path / "hi.json"
    assert main(["translate", str(transcript), "--target", "hi", "--out", str(out)]) == 0
    assert "coverage" in capsys.readouterr().out
    translated = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert translated.language == "hi"
    assert translated.segments[0].text == "नमस्ते दुनिया"
    assert translated.segments[0].start == 0.0
    assert translated.segments[0].end == 1.0


def test_cli_translate_srt_with_chunking(tmp_path):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    out = tmp_path / "hi.srt"
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--format",
                "srt",
                "--max-chars",
                "8",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    cues = out.read_text(encoding="utf-8").strip().split("\n\n")
    assert len(cues) == 2
    assert all(len(cue.split("\n", 2)[2].strip()) <= 8 for cue in cues)


def test_cli_translate_with_glossary(tmp_path):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    glossary = tmp_path / "glossary.json"
    glossary.write_text(json.dumps({"world": "दुनिया-जी"}), encoding="utf-8")
    out = tmp_path / "hi.json"
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--glossary",
                str(glossary),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert Transcript.from_json(out.read_text(encoding="utf-8")).segments[0].text == (
        "नमस्ते दुनिया-जी"
    )


def test_cli_translate_defaults_source_to_english(tmp_path):
    # A transcript with no language set translates as English by default.
    transcript = tmp_path / "nolanguage.json"
    transcript.write_text(
        Transcript(segments=[{"words": [{"text": "Hello", "start": 0.0, "end": 0.3}]}]).to_json()
    )
    out = tmp_path / "hi.json"
    assert main(["translate", str(transcript), "--target", "hi", "--out", str(out)]) == 0
    assert Transcript.from_json(out.read_text(encoding="utf-8")).segments[0].text == "नमस्ते"
