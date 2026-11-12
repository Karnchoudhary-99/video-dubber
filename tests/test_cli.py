import json
import subprocess

import pytest

from video_dubber.audio import concat, load_wav, make_silence, make_sine, save_wav
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


def test_cli_translate_round_trip_reports_fidelity(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    out = tmp_path / "hi.json"
    report_path = tmp_path / "fidelity.json"
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--out",
                str(out),
                "--round-trip",
                "--fidelity-out",
                str(report_path),
            ]
        )
        == 0
    )
    out_text = capsys.readouterr().out
    assert "fidelity: verdict good" in out_text
    assert "mean 1.00" in out_text
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "good"
    assert report["mean_score"] == 1.0
    assert report["segments"][0]["back_text"] == "hello world"


def test_cli_config_shows_defaults(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert '"target_language": null' in out
    assert '"backend": "dictionary"' in out
    assert "no config file found" in out


def test_cli_config_loads_file(tmp_path, capsys):
    config_path = tmp_path / "dubber.toml"
    config_path.write_text('[dubber]\ntarget_language = "hi"\nbed_level = 0.3\n', encoding="utf-8")
    assert main(["--config", str(config_path), "config"]) == 0
    out = capsys.readouterr().out
    assert '"target_language": "hi"' in out
    assert '"bed_level": 0.3' in out
    assert f"loaded from {config_path}" in out


def test_cli_config_toml_format(tmp_path, capsys):
    config_path = tmp_path / "dubber.toml"
    config_path.write_text('[dubber]\ntarget_language = "es"\n', encoding="utf-8")
    assert main(["--config", str(config_path), "config", "--format", "toml"]) == 0
    out = capsys.readouterr().out
    assert 'target_language = "es"' in out
    assert "[dubber.enhance]" in out
    assert "target = -16.0" in out


def test_cli_config_missing_file_errors(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.toml"), "config"]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_translate_target_from_config(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    config_path = tmp_path / "dubber.toml"
    config_path.write_text('[dubber]\ntarget_language = "hi"\n', encoding="utf-8")
    out = tmp_path / "hi.json"
    assert (
        main(["--config", str(config_path), "translate", str(transcript), "--out", str(out)]) == 0
    )
    assert Transcript.from_json(out.read_text(encoding="utf-8")).language == "hi"


def test_cli_translate_flag_beats_config(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    config_path = tmp_path / "dubber.toml"
    config_path.write_text('[dubber]\ntarget_language = "es"\n', encoding="utf-8")
    out = tmp_path / "out.json"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert Transcript.from_json(out.read_text(encoding="utf-8")).language == "hi"


def test_cli_translate_no_target_anywhere_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    assert main(["translate", str(transcript)]) == 2
    assert "needs --target" in capsys.readouterr().err


def test_cli_enhance_target_from_config(tmp_path):
    from video_dubber.enhance import measure_loudness

    src = tmp_path / "in.wav"
    save_wav(str(src), make_sine(440.0, 1.0, amplitude=0.5))
    config_path = tmp_path / "dubber.toml"
    config_path.write_text("[dubber.enhance]\ntarget = -30.0\n", encoding="utf-8")
    out = tmp_path / "out.wav"
    assert main(["--config", str(config_path), "enhance", str(src), "--out", str(out)]) == 0
    assert measure_loudness(load_wav(str(out))).rms_dbfs == pytest.approx(-30.0, abs=1.0)


def test_cli_enhance_flag_beats_config(tmp_path):
    from video_dubber.enhance import measure_loudness

    src = tmp_path / "in.wav"
    save_wav(str(src), make_sine(440.0, 1.0, amplitude=0.5))
    config_path = tmp_path / "dubber.toml"
    config_path.write_text("[dubber.enhance]\ntarget = -30.0\n", encoding="utf-8")
    out = tmp_path / "out.wav"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "enhance",
                str(src),
                "--out",
                str(out),
                "--target",
                "-10",
            ]
        )
        == 0
    )
    assert measure_loudness(load_wav(str(out))).rms_dbfs == pytest.approx(-10.0, abs=1.0)


def test_cli_dub_bed_level_from_config(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "orig.wav"
    save_wav(str(original), make_sine(97.3, 2.5, amplitude=0.5))
    config_path = tmp_path / "dubber.toml"
    config_path.write_text("[dubber]\nbed_level = 0.3\n", encoding="utf-8")
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--out",
                str(out),
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    assert "background bed level: 0.30" in capsys.readouterr().out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["bed_level"] == pytest.approx(0.3)


def test_cli_translate_round_trip_threshold_controls_flagging(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    transcript.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "Hello", "start": 0.0, "end": 0.5},
                        {"text": "world", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": None,
                },
                {
                    "words": [
                        {"text": "See", "start": 2.0, "end": 2.5},
                        {"text": "you", "start": 2.5, "end": 3.0},
                        {"text": "tomorrow", "start": 3.0, "end": 3.5},
                    ],
                    "speaker": None,
                },
            ],
            language="en",
        ).to_json()
    )
    # "tomorrow" round-trips at 0.5 (comes back "yesterday"): a 0.4
    # threshold passes it, the default 0.6 flags it.
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--round-trip",
                "--fidelity-threshold",
                "0.4",
            ]
        )
        == 0
    )
    assert "fidelity: verdict good" in capsys.readouterr().out
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--round-trip",
                "--fidelity-threshold",
                "0.9",
            ]
        )
        == 0
    )
    assert "fidelity: verdict review" in capsys.readouterr().out


def test_cli_translate_round_trip_bad_threshold_errors(tmp_path, capsys):
    transcript = tmp_path / "en.json"
    _write_transcript(transcript)
    assert (
        main(
            [
                "translate",
                str(transcript),
                "--target",
                "hi",
                "--round-trip",
                "--fidelity-threshold",
                "1.5",
            ]
        )
        == 2
    )
    assert "must be between 0 and 1" in capsys.readouterr().err


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


def _make_reference_wav(path, freq=220.0):
    save_wav(str(path), make_sine(freq, 0.5))


def test_cli_clone_and_synthesize(tmp_path, capsys):
    from video_dubber.voice import SpeakerProfile

    ref = tmp_path / "ref.wav"
    _make_reference_wav(ref)
    profile_path = tmp_path / "asha.json"
    assert main(["clone", str(ref), "--name", "Asha", "--out", str(profile_path)]) == 0
    assert "cloned Asha" in capsys.readouterr().out
    profile = SpeakerProfile.from_json(profile_path.read_text(encoding="utf-8"))
    assert profile.name == "Asha"
    assert profile.base_pitch_hz == pytest.approx(220.0, rel=0.1)

    out_wav = tmp_path / "hello.wav"
    assert (
        main(
            [
                "synthesize",
                "hello world",
                "--profile",
                str(profile_path),
                "--out",
                str(out_wav),
            ]
        )
        == 0
    )
    assert "synthesized 2 words" in capsys.readouterr().out
    assert load_wav(str(out_wav)).duration > 0.2


def test_cli_clone_silent_reference_fails(tmp_path, capsys):
    ref = tmp_path / "silent.wav"
    save_wav(str(ref), make_silence(0.5))
    profile_path = tmp_path / "nobody.json"
    assert main(["clone", str(ref), "--name", "Nobody", "--out", str(profile_path)]) == 1
    assert "silent" in capsys.readouterr().err


def test_cli_synthesize_with_pitch_shift(tmp_path):
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    profile_path = tmp_path / "low.json"
    profile_path.write_text(
        SpeakerProfile(
            name="Low",
            embedding=[0.25] * EMBEDDING_DIM,
            base_pitch_hz=110.0,
        ).to_json()
    )
    out = tmp_path / "shifted.wav"
    assert (
        main(
            [
                "synthesize",
                "hello",
                "--profile",
                str(profile_path),
                "--semitones",
                "12",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert load_wav(str(out)).duration > 0.1


def _write_dub_transcript(path):
    path.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "नमस्ते", "start": 0.0, "end": 0.5},
                        {"text": "दुनिया", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": None,
                },
                {
                    "words": [{"text": "शानदार", "start": 2.0, "end": 2.5}],
                    "speaker": None,
                },
            ],
            language="hi",
        ).to_json()
    )


def _write_dub_profile(path, pitch: float = 180.0):
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    path.write_text(
        SpeakerProfile(
            name="Dub",
            embedding=[0.25] * EMBEDDING_DIM,
            base_pitch_hz=pitch,
        ).to_json()
    )


@ffmpeg_missing
def test_cli_dub_renders_track_and_report(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    out_text = capsys.readouterr().out
    assert "dubbed 2 segments" in out_text
    assert "verdict: good" in out_text
    assert load_wav(str(out)).duration == pytest.approx(2.5)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["segments_dubbed"] == 2
    assert data["verdict"] == "good"


def test_cli_dub_html_report(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.html"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--report-out",
                str(report),
                "--report-format",
                "html",
            ]
        )
        == 0
    )
    assert "verdict: good" in capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert 'class="banner good"' in text
    assert "<h2>Segments</h2>" in text
    assert "Segments dubbed" in text


def _write_dub_report_json(tmp_path):
    """Dub the fixture transcript and return the saved JSON report path."""
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "dub.wav"),
                "--report-out",
                str(report),
            ]
        )
        == 0
    )
    return report


def test_cli_report_renders_saved_json_as_html(tmp_path, capsys):
    report = _write_dub_report_json(tmp_path)
    out = tmp_path / "report.html"
    assert main(["report", str(report), "--out", str(out)]) == 0
    assert f"wrote {out}" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert 'class="banner good"' in text
    assert "<h2>Segments</h2>" in text


def test_cli_report_prints_to_stdout_by_default(tmp_path, capsys):
    report = _write_dub_report_json(tmp_path)
    assert main(["report", str(report)]) == 0
    assert "<!DOCTYPE html>" in capsys.readouterr().out


def test_cli_report_markdown_format(tmp_path, capsys):
    report = _write_dub_report_json(tmp_path)
    out = tmp_path / "report.md"
    assert main(["report", str(report), "--format", "md", "--out", str(out)]) == 0
    assert "# Dub quality report" in out.read_text(encoding="utf-8")


def test_cli_report_bad_json_fails(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert main(["report", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_report_missing_file_fails(tmp_path, capsys):
    assert main(["report", str(tmp_path / "nope.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_dub_with_bed_level(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "orig.wav"
    save_wav(str(original), make_sine(97.3, 2.5, amplitude=0.5))
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--bed-level",
                "0.3",
                "--out",
                str(out),
                "--report-out",
                str(report),
            ]
        )
        == 0
    )
    assert "background bed level: 0.30" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8"))["bed_level"] == 0.3
    # The bed is audible in the gap between the two segments (1.0-2.0s).
    assert load_wav(str(out)).samples[int(1.5 * 16_000)] != pytest.approx(0.0)


def test_cli_enhance_trims_and_normalizes(tmp_path, capsys):
    from video_dubber.enhance import measure_loudness

    src = tmp_path / "raw.wav"
    save_wav(
        str(src),
        concat(make_silence(0.3), make_sine(440.0, 0.5, amplitude=0.05), make_silence(0.3)),
    )
    out = tmp_path / "clean.wav"
    report = tmp_path / "enhance.md"
    assert main(["enhance", str(src), "--out", str(out), "--report-out", str(report)]) == 0
    cleaned = load_wav(str(out))
    assert cleaned.duration == pytest.approx(0.6, abs=0.02)  # speech + margins
    assert measure_loudness(cleaned).rms_dbfs == pytest.approx(-16.0, abs=0.5)
    text = report.read_text(encoding="utf-8")
    assert "RMS:" in text and "-16.0 dBFS" in text
    assert "wrote" in capsys.readouterr().out


def test_cli_dub_with_enhance_keeps_timing(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--enhance",
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    out_text = capsys.readouterr().out
    assert "enhanced: RMS" in out_text
    # Timing preserved: the dub still spans the full 2.5 s transcript.
    assert load_wav(str(out)).duration == pytest.approx(2.5)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["enhancement"] is not None
    assert data["enhancement"]["gain_db"] != pytest.approx(0.0)


def test_cli_dub_bed_level_without_original_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--bed-level",
                "0.3",
                "--out",
                str(tmp_path / "dub.wav"),
            ]
        )
        == 2
    )
    assert "--bed-level needs --original" in capsys.readouterr().err


def test_cli_dub_bed_level_out_of_range_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "orig.wav"
    save_wav(str(original), make_sine(97.3, 2.5, amplitude=0.5))
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--bed-level",
                "1.5",
                "--out",
                str(tmp_path / "dub.wav"),
            ]
        )
        == 2
    )
    assert "--bed-level must be between 0 and 1" in capsys.readouterr().err


@ffmpeg_missing
def test_cli_dub_with_video_mux(tmp_path):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video), duration=2.5)
    out = tmp_path / "dub.wav"
    muxed = tmp_path / "dubbed.mp4"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--video",
                str(video),
                "--mux-out",
                str(muxed),
            ]
        )
        == 0
    )
    assert muxed.exists() and muxed.stat().st_size > 0


def test_cli_dub_video_without_mux_out_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "dub.wav"),
                "--video",
                str(tmp_path / "clip.mp4"),
            ]
        )
        == 2
    )
    assert "--video needs --mux-out" in capsys.readouterr().err


def _write_source_transcript(path, text="hello mars good morning"):
    words = [
        {"text": w, "start": i * 0.4, "end": (i + 1) * 0.4} for i, w in enumerate(text.split())
    ]
    path.write_text(
        Transcript(segments=[{"words": words, "speaker": None}], language="en").to_json()
    )


def _write_batch_manifest(path, jobs, defaults=None):
    manifest = {"jobs": jobs}
    if defaults is not None:
        manifest["defaults"] = defaults
    path.write_text(json.dumps(manifest))


def _batch_fixture(tmp_path):
    """Two transcript jobs (hi + es) sharing one profile; returns (manifest, out)."""
    src1 = tmp_path / "src1.json"
    _write_source_transcript(src1)
    src2 = tmp_path / "src2.json"
    _write_source_transcript(src2, "good morning mars")
    profile = tmp_path / "asha.json"
    _write_dub_profile(profile)
    manifest = tmp_path / "manifest.json"
    _write_batch_manifest(
        manifest,
        [
            {"name": "clip1", "transcript": str(src1), "target": "hi"},
            {"name": "clip2", "transcript": str(src2), "target": "es"},
        ],
        defaults={"profile": str(profile)},
    )
    return manifest, tmp_path / "out"


def test_cli_batch_two_jobs(tmp_path, capsys):
    manifest, out = _batch_fixture(tmp_path)
    assert main(["batch", str(manifest), "--out-dir", str(out)]) == 0
    assert (out / "clip1" / "dubbed.wav").exists()
    assert (out / "clip2" / "dubbed.wav").exists()
    report = json.loads((out / "batch-report.json").read_text(encoding="utf-8"))
    assert report["succeeded"] == 2
    assert report["failed"] == 0
    assert (out / "batch-report.md").exists()
    output = capsys.readouterr().out
    assert "ok clip1" in output and "ok clip2" in output
    assert "batch done: 2 ok, 0 failed, 0 skipped out of 2" in output


def test_cli_batch_second_run_skips(tmp_path, capsys):
    manifest, out = _batch_fixture(tmp_path)
    assert main(["batch", str(manifest), "--out-dir", str(out)]) == 0
    capsys.readouterr()
    assert main(["batch", str(manifest), "--out-dir", str(out)]) == 0
    output = capsys.readouterr().out
    assert "skipped clip1" in output and "skipped clip2" in output
    assert "0 ok, 0 failed, 2 skipped" in output


def test_cli_batch_failure_does_not_stop_others(tmp_path, capsys):
    manifest, out = _batch_fixture(tmp_path)
    # Give clip2 a corrupt profile: the manifest only checks files exist,
    # so the job starts and fails at runtime while clip1 still succeeds.
    bad_profile = tmp_path / "bad.json"
    bad_profile.write_text("{not json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["jobs"][1]["profile"] = str(bad_profile)
    manifest.write_text(json.dumps(data))
    assert main(["batch", str(manifest), "--out-dir", str(out)]) == 1
    assert (out / "clip1" / "dubbed.wav").exists()
    assert not (out / "clip2" / "dubbed.wav").exists()
    err = capsys.readouterr().err
    assert "failed clip2" in err
    report = json.loads((out / "batch-report.json").read_text(encoding="utf-8"))
    assert report["succeeded"] == 1 and report["failed"] == 1


def test_cli_batch_missing_manifest(tmp_path, capsys):
    assert main(["batch", "nope.json", "--out-dir", str(tmp_path / "out")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_dub_with_profiles_and_subtitles(tmp_path, capsys):
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    transcript = tmp_path / "hi.json"
    transcript.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "नमस्ते", "start": 0.0, "end": 0.5},
                        {"text": "दुनिया", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": "Asha",
                },
                {
                    "words": [{"text": "शानदार", "start": 2.0, "end": 2.5}],
                    "speaker": "Ravi",
                },
            ],
            language="hi",
        ).to_json()
    )
    default = tmp_path / "dub.json"
    _write_dub_profile(default)
    for name, pitch in (("asha", 220.0), ("ravi", 110.0)):
        (tmp_path / f"{name}.json").write_text(
            SpeakerProfile(
                name=name.capitalize(),
                embedding=[0.25] * EMBEDDING_DIM,
                base_pitch_hz=pitch,
            ).to_json()
        )
    profiles_map = tmp_path / "profiles.json"
    profiles_map.write_text(json.dumps({"Asha": "asha.json", "Ravi": "ravi.json"}))
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(default),
                "--profiles",
                str(profiles_map),
                "--out",
                str(out),
                "--subtitles",
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    assert "speakers dubbed: 2" in capsys.readouterr().out
    assert load_wav(str(out)).duration == pytest.approx(2.5)
    assert json.loads(report.read_text(encoding="utf-8"))["speakers_dubbed"] == 2
    srt = (tmp_path / "dub.srt").read_text(encoding="utf-8")
    assert "नमस्ते दुनिया" in srt and "-->" in srt
    assert (tmp_path / "dub.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


def test_cli_dub_subtitles_follow_placed_audio_not_window(tmp_path, capsys):
    """A clamped stretch ends the speech early; the cue must end with it."""
    transcript = tmp_path / "hi.json"
    transcript.write_text(
        Transcript(
            segments=[
                {
                    "words": [{"text": "नमस्ते", "start": 0.0, "end": 4.0}],
                    "speaker": "Asha",
                },
            ],
            language="hi",
        ).to_json()
    )
    default = tmp_path / "dub.json"
    _write_dub_profile(default)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(default),
                "--out",
                str(out),
                "--subtitles",
            ]
        )
        == 0
    )
    # One short word in a 4s window: the 1.6x clamp leaves the placed clip
    # ~0.85s long, so the cue must end there — not at the 4.0s window end.
    srt = (tmp_path / "dub.srt").read_text(encoding="utf-8")
    assert "00:00:04,000" not in srt
    cue_line = next(line for line in srt.splitlines() if "-->" in line)
    end = cue_line.split("-->")[1].strip()
    hours, minutes, rest = end.split(":")
    secs, millis = rest.split(",")
    end_s = int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000
    assert 0.80 < end_s < 0.90


def test_cli_dub_profiles_map_missing_file_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    bad_map = tmp_path / "profiles.json"
    bad_map.write_text(json.dumps({"Asha": "nope.json"}))
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--profiles",
                str(bad_map),
                "--out",
                str(tmp_path / "dub.wav"),
            ]
        )
        == 1
    )
    assert "cannot read profile for speaker 'Asha'" in capsys.readouterr().err


def test_cli_voices_list_and_add(tmp_path, capsys):
    library = tmp_path / "voices.json"
    # Missing library: friendly message, exit 0.
    assert main(["voices", str(library)]) == 0
    assert "no voice library" in capsys.readouterr().out
    profile = tmp_path / "asha.json"
    _write_dub_profile(profile)
    assert main(["voices", str(library), "--add", str(profile)]) == 0
    out = capsys.readouterr().out
    assert "added Dub to" in out
    assert "1 voice(s)" in out
    assert "Dub: 180 Hz" in out
    # Listing again shows the same voice.
    assert main(["voices", str(library)]) == 0
    assert "Dub: 180 Hz" in capsys.readouterr().out
    # Duplicate add without --overwrite fails.
    assert main(["voices", str(library), "--add", str(profile)]) == 1
    assert "already has a profile" in capsys.readouterr().err
    assert main(["voices", str(library), "--add", str(profile), "--overwrite"]) == 0


def test_cli_clone_adds_to_library(tmp_path, capsys):
    ref = tmp_path / "ref.wav"
    _make_reference_wav(ref)
    out = tmp_path / "asha.json"
    library = tmp_path / "voices.json"
    assert (
        main(
            [
                "clone",
                str(ref),
                "--name",
                "Asha",
                "--out",
                str(out),
                "--library",
                str(library),
            ]
        )
        == 0
    )
    assert "added Asha to library" in capsys.readouterr().out
    assert main(["voices", str(library)]) == 0
    assert "Asha:" in capsys.readouterr().out


def _make_diarize_fixtures(tmp_path):
    """Two-speaker audio + unlabelled transcript + a two-voice library."""
    from video_dubber.transcript import Segment, Transcript, Word
    from video_dubber.voice import VoiceLibrary, clone_from_references

    audio = tmp_path / "clip.wav"
    save_wav(str(audio), concat(make_sine(220.0, 1.0), make_sine(110.0, 1.0)))
    transcript = Transcript(
        segments=[
            Segment(words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)]),
            Segment(words=[Word("good", 1.0, 1.5), Word("morning", 1.5, 2.0)]),
        ],
        language="en",
    )
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    library = VoiceLibrary()
    library.add(clone_from_references("Asha", [make_sine(220.0, 1.0)]))
    library.add(clone_from_references("Ravi", [make_sine(110.0, 1.0)]))
    library_path = tmp_path / "voices.json"
    library.save(str(library_path))
    return audio, transcript_path, library_path


def test_cli_diarize_labels_segments(tmp_path, capsys):
    audio, transcript_path, library_path = _make_diarize_fixtures(tmp_path)
    out = tmp_path / "labelled.json"
    report_path = tmp_path / "report.json"
    assert (
        main(
            [
                "diarize",
                str(transcript_path),
                "--audio",
                str(audio),
                "--library",
                str(library_path),
                "--out",
                str(out),
                "--report-out",
                str(report_path),
            ]
        )
        == 0
    )
    labelled = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert [s.speaker for s in labelled.segments] == ["Asha", "Ravi"]
    stdout = capsys.readouterr().out
    assert "Asha" in stdout and "Ravi" in stdout
    assert "2 segments: 2 matched, 0 kept, 0 unknown" in stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [label["speaker"] for label in report["labels"]] == ["Asha", "Ravi"]


def test_cli_diarize_keeps_existing_label_without_overwrite(tmp_path, capsys):
    from video_dubber.transcript import Transcript

    audio, transcript_path, library_path = _make_diarize_fixtures(tmp_path)
    transcript = Transcript.from_json(transcript_path.read_text(encoding="utf-8"))
    transcript.segments[0].speaker = "Ravi"  # wrong label, must survive
    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    out = tmp_path / "labelled.json"
    assert (
        main(
            [
                "diarize",
                str(transcript_path),
                "--audio",
                str(audio),
                "--library",
                str(library_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    labelled = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert labelled.segments[0].speaker == "Ravi"
    assert "1 matched, 1 kept, 0 unknown" in capsys.readouterr().out
    assert (
        main(
            [
                "diarize",
                str(transcript_path),
                "--audio",
                str(audio),
                "--library",
                str(library_path),
                "--out",
                str(out),
                "--overwrite",
            ]
        )
        == 0
    )
    labelled = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert labelled.segments[0].speaker == "Asha"


def test_cli_diarize_library_from_config(tmp_path, capsys):
    audio, transcript_path, library_path = _make_diarize_fixtures(tmp_path)
    config_path = tmp_path / "dubber.toml"
    config_path.write_text(f'[dubber]\nvoice_library = "{library_path}"\n', encoding="utf-8")
    out = tmp_path / "labelled.json"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "diarize",
                str(transcript_path),
                "--audio",
                str(audio),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    labelled = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert [s.speaker for s in labelled.segments] == ["Asha", "Ravi"]


def test_cli_diarize_no_library_anywhere_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio, transcript_path, _ = _make_diarize_fixtures(tmp_path)
    assert main(["diarize", str(transcript_path), "--audio", str(audio)]) == 2
    assert "needs --library" in capsys.readouterr().err


def test_cli_diarize_empty_library_fails(tmp_path, capsys):
    from video_dubber.voice import VoiceLibrary

    audio, transcript_path, _ = _make_diarize_fixtures(tmp_path)
    empty = tmp_path / "empty.json"
    VoiceLibrary().save(str(empty))
    assert (
        main(
            [
                "diarize",
                str(transcript_path),
                "--audio",
                str(audio),
                "--library",
                str(empty),
            ]
        )
        == 1
    )
    assert "empty voice library" in capsys.readouterr().err


# --- patch -------------------------------------------------------------------


def _make_dubbed(tmp_path):
    """Run `dub` once; returns (dub wav, transcript, profile, report)."""
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    return out, transcript, profile, report


def test_cli_patch_replaces_one_segment(tmp_path, capsys):
    dub, transcript, profile, report = _make_dubbed(tmp_path)
    before = load_wav(str(dub))
    out = tmp_path / "patched.wav"
    patched_transcript = tmp_path / "patched.json"
    patched_report = tmp_path / "patched-report.json"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "2",
                "--text",
                "बदला हुआ",
                "--out",
                str(out),
                "--transcript-out",
                str(patched_transcript),
                "--report",
                str(report),
                "--report-out",
                str(patched_report),
            ]
        )
        == 0
    )
    assert "patched segment 2" in capsys.readouterr().out
    after = load_wav(str(out))
    assert after.duration == pytest.approx(before.duration)
    # Segment 1 (0-1s) is untouched; segment 2 changed.
    assert after.samples[:16_000] == before.samples[:16_000]
    assert after.samples[32_000:] != before.samples[32_000:]
    patched = Transcript.from_json(patched_transcript.read_text(encoding="utf-8"))
    assert patched.segments[1].text == "बदला हुआ"
    assert patched.segments[0].text == "नमस्ते दुनिया"
    data = json.loads(patched_report.read_text(encoding="utf-8"))
    assert data["segments"][1]["text"] == "बदला हुआ"
    assert "verdict" in data


def test_cli_patch_with_subtitles(tmp_path):
    dub, transcript, profile, report = _make_dubbed(tmp_path)
    out = tmp_path / "patched.wav"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "2",
                "--text",
                "बदला हुआ",
                "--out",
                str(out),
                "--report",
                str(report),
                "--subtitles",
            ]
        )
        == 0
    )
    srt = (tmp_path / "patched.srt").read_text(encoding="utf-8")
    assert "बदला हुआ" in srt
    assert "नमस्ते दुनिया" in srt
    assert (tmp_path / "patched.vtt").exists()


def test_cli_patch_bad_segment_fails(tmp_path, capsys):
    dub, transcript, profile, _report = _make_dubbed(tmp_path)
    out = tmp_path / "patched.wav"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "9",
                "--text",
                "x",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def test_cli_patch_missing_dub_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "patched.wav"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(tmp_path / "nope.wav"),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "1",
                "--text",
                "x",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def test_cli_patch_dry_run_writes_nothing(tmp_path, capsys):
    dub, transcript, profile, report = _make_dubbed(tmp_path)
    report_before = report.read_text(encoding="utf-8")
    out = tmp_path / "patched.wav"
    patched_transcript = tmp_path / "patched.json"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "2",
                "--text",
                "बदला हुआ",
                "--out",
                str(out),
                "--transcript-out",
                str(patched_transcript),
                "--report",
                str(report),
                "--dry-run",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "dry run: segment 2 would fit at" in stdout
    assert "dry run: report verdict would be" in stdout
    assert not out.exists()
    assert not patched_transcript.exists()
    assert report.read_text(encoding="utf-8") == report_before


def test_cli_patch_leaves_music_bed_untouched(tmp_path):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "orig.wav"
    save_wav(str(original), make_sine(97.3, 2.5, amplitude=0.5))
    dub = tmp_path / "dub.wav"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--bed-level",
                "0.3",
                "--out",
                str(dub),
                "--report-out",
                str(report),
                "--report-format",
                "json",
            ]
        )
        == 0
    )
    before = load_wav(str(dub))
    out = tmp_path / "patched.wav"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--segment",
                "2",
                "--text",
                "बदला हुआ",
                "--out",
                str(out),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    after = load_wav(str(out))
    # Everything before segment 2 (2.0s) — speech and music bed alike —
    # is bit-identical; only the patched segment's span changed.
    assert after.samples[: 2 * 16_000] == before.samples[: 2 * 16_000]
    assert after.samples[2 * 16_000 :] != before.samples[2 * 16_000 :]


# --- watch subcommand --------------------------------------------------------


def _watch_fixture(tmp_path):
    """A watch dir with one WAV file + scripted word list; returns paths."""
    watch = tmp_path / "watch"
    watch.mkdir()
    audio = watch / "morning clip.wav"
    save_wav(str(audio), make_sine(440.0, 2.0, amplitude=0.5))
    script = tmp_path / "words.json"
    script.write_text(
        json.dumps(
            [
                {"text": "Hello", "start": 0.0, "end": 0.4},
                {"text": "and", "start": 0.4, "end": 0.6},
                {"text": "welcome", "start": 0.6, "end": 1.1},
                {"text": "to", "start": 1.1, "end": 1.3},
                {"text": "Mars", "start": 1.3, "end": 1.7},
            ]
        )
    )
    profile = tmp_path / "asha.json"
    _write_dub_profile(profile)
    return watch, script, profile, tmp_path / "out"


def _watch_args(watch, script, profile, out, extra=None):
    args = [
        "watch",
        str(watch),
        "--out-dir",
        str(out),
        "--target",
        "hi",
        "--profile",
        str(profile),
        "--backend",
        "scripted",
        "--script",
        str(script),
        "--once",
        "--no-mux",
    ]
    return args + (extra or [])


def test_cli_watch_dubs_new_file_and_remembers_it(tmp_path, capsys):
    watch, script, profile, out = _watch_fixture(tmp_path)
    assert main(_watch_args(watch, script, profile, out)) == 0
    assert (out / "morning-clip" / "dubbed.wav").exists()
    assert (out / ".dubber-watch.json").exists()
    output = capsys.readouterr().out
    assert "ok morning-clip" in output
    assert "scan 1: 1 new, 1 ok, 0 failed, 0 skipped" in output
    # Second scan: the file is already dubbed, nothing to do.
    assert main(_watch_args(watch, script, profile, out)) == 0
    assert "scan 1: 0 new, 0 ok, 0 failed, 0 skipped" in capsys.readouterr().out


def test_cli_watch_empty_dir(tmp_path, capsys):
    watch = tmp_path / "watch"
    watch.mkdir()
    profile = tmp_path / "asha.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "watch",
                str(watch),
                "--out-dir",
                str(tmp_path / "out"),
                "--target",
                "hi",
                "--profile",
                str(profile),
                "--once",
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / ".dubber-watch.json").exists()
    assert "scan 1: 0 new" in capsys.readouterr().out


def test_cli_watch_missing_dir_fails(tmp_path, capsys):
    profile = tmp_path / "asha.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "watch",
                str(tmp_path / "missing"),
                "--out-dir",
                str(tmp_path / "out"),
                "--target",
                "hi",
                "--profile",
                str(profile),
                "--once",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def test_cli_watch_needs_target(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no dubber.toml here, so no config target
    watch, script, profile, out = _watch_fixture(tmp_path)
    args = _watch_args(watch, script, profile, out)
    args.remove("--target")
    args.remove("hi")
    assert main(args) == 2
    assert "needs --target" in capsys.readouterr().err


def test_cli_watch_once_and_scans_conflict(tmp_path, capsys):
    watch, script, profile, out = _watch_fixture(tmp_path)
    assert main(_watch_args(watch, script, profile, out, ["--scans", "2"])) == 2
    assert "conflict" in capsys.readouterr().err


def test_cli_watch_extensions_filter(tmp_path, capsys):
    watch, script, profile, out = _watch_fixture(tmp_path)
    (watch / "notes.txt").write_text("not media")
    assert main(_watch_args(watch, script, profile, out, ["--extensions", "wav"])) == 0
    assert (out / "morning-clip" / "dubbed.wav").exists()
    assert "scan 1: 1 new" in capsys.readouterr().out


def _write_clipped_wav(path):
    from video_dubber.audio import AudioData

    save_wav(str(path), AudioData([1.0] * 32_000, 16_000))


def _write_quiet_wav(path):
    save_wav(
        str(path),
        concat(
            make_sine(440.0, 1.0, amplitude=0.05),
            make_silence(1.0),
            make_sine(440.0, 1.0, amplitude=0.05),
            make_silence(1.0),
        ),
    )


def test_cli_preflight_clean_passes(tmp_path, capsys):
    wav = tmp_path / "clean.wav"
    _make_wav(wav)
    assert main(["preflight", str(wav)]) == 0
    assert "pass" in capsys.readouterr().out


def test_cli_preflight_clipped_fails(tmp_path, capsys):
    wav = tmp_path / "clipped.wav"
    _write_clipped_wav(wav)
    assert main(["preflight", str(wav)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_preflight_json_output(tmp_path, capsys):
    wav = tmp_path / "clean.wav"
    _make_wav(wav)
    assert main(["preflight", str(wav), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "pass"
    assert len(data["checks"]) == 8


def test_cli_preflight_fail_on_warn(tmp_path, capsys):
    wav = tmp_path / "quiet.wav"
    _write_quiet_wav(wav)
    assert main(["preflight", str(wav)]) == 0  # warnings alone pass
    assert main(["preflight", str(wav), "--fail-on-warn"]) == 1


def test_cli_preflight_missing_file(tmp_path, capsys):
    assert main(["preflight", str(tmp_path / "nope.wav")]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_dub_preflight_gate_aborts(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "clipped.wav"
    _write_clipped_wav(original)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--preflight",
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert "dub aborted" in capsys.readouterr().err
    assert not out.exists()


def test_cli_dub_preflight_gate_passes(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    original = tmp_path / "clean.wav"
    _make_wav(original)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--original",
                str(original),
                "--preflight",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists()


def _write_compare_report(path, **overrides):
    from video_dubber.assemble import DubReport

    base = {
        "segments_dubbed": 2,
        "total_duration": 2.5,
        "mean_sync_drift": 0.02,
        "max_sync_drift": 0.05,
        "segments_over_drift": 0,
        "mean_fit_factor": 1.02,
        "max_fit_factor": 1.1,
        "extreme_fits": 0,
        "loudness_error_db": 0.5,
        "translation_coverage": 0.9,
        "word_error_rate": None,
        "mos_proxy": 4.5,
        "verdict": "good",
    }
    base.update(overrides)
    path.write_text(DubReport(**base).to_json(), encoding="utf-8")


def test_cli_compare_names_the_winner(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_compare_report(a)
    _write_compare_report(b, mos_proxy=3.2, mean_sync_drift=0.15, verdict="review")
    assert main(["compare", str(a), str(b), "--label-a", "asha", "--label-b", "ravi"]) == 0
    out = capsys.readouterr().out
    assert "Verdict: **asha**" in out
    assert "| MOS proxy | 4.5 | 3.2 | asha |" in out


def test_cli_compare_json_output(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_compare_report(a)
    _write_compare_report(b)
    assert main(["compare", str(a), str(b), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "tie"
    assert data["label_a"] == "A"


def test_cli_compare_bad_report_fails(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{not json", encoding="utf-8")
    _write_compare_report(b)
    assert main(["compare", str(a), str(b)]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_preflight_report_out(tmp_path, capsys):
    wav = tmp_path / "clean.wav"
    _make_wav(wav)
    report = tmp_path / "preflight.md"
    assert main(["preflight", str(wav), "--report-out", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "Pre-flight report — pass" in text
    assert "[OK] clipping" in text


def test_cli_compare_report_out(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_compare_report(a)
    _write_compare_report(b, mos_proxy=3.2)
    report = tmp_path / "compare.md"
    assert main(["compare", str(a), str(b), "--report-out", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "Verdict: **A**" in text
    assert "| MOS proxy | 4.5 | 3.2 | A |" in text


@ffmpeg_missing
def test_cli_render_burns_subtitles(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video), duration=2.0)
    audio = tmp_path / "dub.wav"
    _make_wav(audio)
    srt = tmp_path / "subs.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nWelcome to Mars\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.mp4"
    report = tmp_path / "render.json"
    assert (
        main(
            [
                "render",
                str(video),
                str(audio),
                "--subtitles",
                str(srt),
                "--out",
                str(out),
                "--report-out",
                str(report),
            ]
        )
        == 0
    )
    assert out.exists() and out.stat().st_size > 0
    assert "burned in 2 subtitle cue(s)" in capsys.readouterr().out
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["subtitle_cues"] == 2
    assert data["filter_used"] == "subtitles"


@ffmpeg_missing
def test_cli_render_preview_slice(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video), duration=4.0)
    audio = tmp_path / "dub.wav"
    _make_wav(audio)
    out = tmp_path / "preview.mp4"
    assert (
        main(
            [
                "render",
                str(video),
                str(audio),
                "--out",
                str(out),
                "--preview-start",
                "1",
                "--preview-duration",
                "1",
            ]
        )
        == 0
    )
    from video_dubber.extract import probe

    assert probe(str(out)).duration == pytest.approx(1.0, abs=0.15)


def test_cli_render_missing_video_fails(tmp_path, capsys):
    audio = tmp_path / "dub.wav"
    _make_wav(audio)
    assert (
        main(["render", str(tmp_path / "nope.mp4"), str(audio), "--out", str(tmp_path / "o.mp4")])
        == 1
    )
    assert "error" in capsys.readouterr().err


def test_cli_render_bad_style_fails(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video), duration=1.0)
    audio = tmp_path / "dub.wav"
    _make_wav(audio)
    assert (
        main(
            [
                "render",
                str(video),
                str(audio),
                "--out",
                str(tmp_path / "o.mp4"),
                "--font-size",
                "200",
            ]
        )
        == 1
    )
    assert "error" in capsys.readouterr().err


@ffmpeg_missing
def test_cli_dub_burn_subtitles(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    video = tmp_path / "clip.mp4"
    _make_test_video(str(video), duration=2.5)
    out = tmp_path / "dub.wav"
    muxed = tmp_path / "dubbed.mp4"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--video",
                str(video),
                "--mux-out",
                str(muxed),
                "--burn-subtitles",
            ]
        )
        == 0
    )
    assert muxed.exists() and muxed.stat().st_size > 0
    assert (tmp_path / "dub.srt").exists()
    assert "subtitle cue(s) burned in" in capsys.readouterr().out


def test_cli_dub_burn_subtitles_without_video_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "dub.wav"),
                "--burn-subtitles",
            ]
        )
        == 2
    )
    assert "burn-subtitles needs --video" in capsys.readouterr().err


def _write_long_transcript(path):
    from video_dubber.transcript import Segment, Transcript, Word

    words = []
    cursor = 0.0
    for i in range(12):
        words.append(Word(f"w{i}", cursor, cursor + 0.6))
        cursor += 0.7
        if i == 5:
            cursor += 0.8  # a real pause after word 5
    path.write_text(
        Transcript(segments=[Segment(words=words, speaker="A")], language="en").to_json()
    )


def test_cli_split_splits_long_segment(tmp_path, capsys):
    transcript = tmp_path / "in.json"
    _write_long_transcript(transcript)
    out = tmp_path / "split.json"
    assert main(["split", str(transcript), "--out", str(out)]) == 0
    assert "1 -> 2" in capsys.readouterr().out
    split = Transcript.from_json(out.read_text())
    assert len(split.segments) == 2
    assert all(s.speaker == "A" for s in split.segments)
    assert [w.text for s in split.segments for w in s.words] == [f"w{i}" for i in range(12)]


def test_cli_split_json_report(tmp_path, capsys):
    transcript = tmp_path / "in.json"
    _write_long_transcript(transcript)
    out = tmp_path / "split.json"
    assert main(["split", str(transcript), "--out", str(out), "--json"]) == 0
    captured = capsys.readouterr().out
    data = json.loads(captured[: captured.index("wrote")].strip())
    assert data["segments_before"] == 1
    assert data["segments_after"] == 2
    assert data["splits"][0]["reason"] == "silence"


def test_cli_split_with_audio(tmp_path, capsys):
    from video_dubber.transcript import Segment, Transcript, Word

    # Words timed back-to-back, but the audio really pauses after word 5.
    words = [Word(f"w{i}", i * 0.6, i * 0.6 + 0.6) for i in range(12)]
    transcript = tmp_path / "in.json"
    transcript.write_text(Transcript(segments=[Segment(words=words)], language="en").to_json())
    audio = tmp_path / "source.wav"
    save_wav(
        str(audio),
        concat(make_sine(440.0, 3.6), make_silence(1.0), make_sine(440.0, 3.6)),
    )
    out = tmp_path / "split.json"
    assert (
        main(
            [
                "split",
                str(transcript),
                "--audio",
                str(audio),
                "--out",
                str(out),
                "--max-duration",
                "4",
            ]
        )
        == 0
    )
    assert "silence" in capsys.readouterr().out
    assert len(Transcript.from_json(out.read_text()).segments) == 2


def test_cli_split_bad_config(tmp_path, capsys):
    transcript = tmp_path / "in.json"
    _write_long_transcript(transcript)
    assert (
        main(
            [
                "split",
                str(transcript),
                "--out",
                str(tmp_path / "split.json"),
                "--max-duration",
                "0",
            ]
        )
        == 1
    )
    assert "error" in capsys.readouterr().err


def test_cli_dub_split_segments(tmp_path, capsys):
    transcript = tmp_path / "in.json"
    _write_long_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--split-segments",
                "--max-segment-duration",
                "5",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    captured = capsys.readouterr().out
    assert "Segment split" in captured
    assert "dubbed 2 segments" in captured
    assert out.exists()


def test_cli_dub_max_segment_duration_needs_split_flag(tmp_path, capsys):
    transcript = tmp_path / "in.json"
    _write_long_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--max-segment-duration",
                "4",
                "--out",
                str(tmp_path / "dub.wav"),
            ]
        )
        == 2
    )
    assert "needs --split-segments" in capsys.readouterr().err


# --- ASS subtitles ------------------------------------------------------------


def test_cli_convert_ass(tmp_path):
    transcript = tmp_path / "t.json"
    transcript.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "Hello", "start": 0.0, "end": 0.5},
                        {"text": "world", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": "Asha",
                },
                {
                    "words": [{"text": "Hi", "start": 2.0, "end": 2.5}],
                    "speaker": "Ravi",
                },
            ],
            language="en",
        ).to_json()
    )
    out = tmp_path / "subs.ass"
    assert main(["convert", str(transcript), "--format", "ass", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "[V4+ Styles]" in text
    assert "Style: Asha," in text and "Style: Ravi," in text
    # Asha gets the first palette colour (white), Ravi the second (yellow).
    assert "&H00FFFFFF" in text  # Asha / Default white
    assert "&H0000FFFF" in text  # Ravi yellow (#FFFF00 -> &H0000FFFF)
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Asha,Asha,0,0,0,,Hello world" in text
    assert "Dialogue: 0,0:00:02.00,0:00:02.50,Ravi,Ravi,0,0,0,,Hi" in text


def test_cli_dub_writes_ass_subtitles(tmp_path):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--subtitles",
            ]
        )
        == 0
    )
    ass = (tmp_path / "dub.ass").read_text(encoding="utf-8")
    assert "[V4+ Styles]" in ass
    assert ass.count("Dialogue:") == 2
    assert "नमस्ते दुनिया" in ass
    assert (tmp_path / "dub.srt").exists()
    assert (tmp_path / "dub.vtt").exists()


def test_cli_patch_writes_ass_subtitles(tmp_path):
    dub, transcript, profile, report = _make_dubbed(tmp_path)
    out = tmp_path / "patched.wav"
    assert (
        main(
            [
                "patch",
                "--dub",
                str(dub),
                "--transcript",
                str(transcript),
                "--profile",
                str(profile),
                "--segment",
                "2",
                "--text",
                "बदला हुआ",
                "--out",
                str(out),
                "--report",
                str(report),
                "--subtitles",
            ]
        )
        == 0
    )
    ass = (tmp_path / "patched.ass").read_text(encoding="utf-8")
    assert "बदला हुआ" in ass
    assert "नमस्ते दुनिया" in ass
    assert ass.count("Dialogue:") == 2


# --- merge --------------------------------------------------------------------


def test_cli_merge_joins_short_gaps(tmp_path, capsys):
    from video_dubber.transcript import Segment, Transcript, Word

    transcript = tmp_path / "fragments.json"
    transcript.write_text(
        Transcript(
            segments=[
                Segment(words=[Word("hello", 0.0, 0.4)]),
                Segment(words=[Word("world", 0.6, 1.0)]),
                Segment(words=[Word("later", 3.0, 3.4)], speaker="Ravi"),
                Segment(words=[Word("again", 3.6, 4.0)], speaker="Ravi"),
            ],
            language="en",
        ).to_json()
    )
    out = tmp_path / "merged.json"
    report_path = tmp_path / "merge.md"
    assert (
        main(
            [
                "merge",
                str(transcript),
                "--out",
                str(out),
                "--report-out",
                str(report_path),
            ]
        )
        == 0
    )
    merged = Transcript.from_json(out.read_text(encoding="utf-8"))
    assert [s.text for s in merged.segments] == ["hello world", "later again"]
    assert merged.segments[1].speaker == "Ravi"
    assert "Segments: 4 -> 2" in capsys.readouterr().out
    assert "# Segment merge" in report_path.read_text(encoding="utf-8")


def test_cli_merge_bad_config_fails(tmp_path, capsys):
    from video_dubber.transcript import Segment, Transcript, Word

    transcript = tmp_path / "t.json"
    transcript.write_text(
        Transcript(
            segments=[Segment(words=[Word("hi", 0.0, 0.3)])],
        ).to_json()
    )
    assert (
        main(["merge", str(transcript), "--out", str(tmp_path / "out.json"), "--max-gap", "-1"])
        == 2
    )
    assert "max_gap" in capsys.readouterr().err


# --- doctor -------------------------------------------------------------------


def test_cli_doctor_reports_environment(tmp_path, capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "# Doctor: " in out
    assert "python (required)" in out
    assert "ffmpeg (required)" in out
    assert "ffprobe (required)" in out
    assert "ffmpeg subtitles filter (required)" in out
    assert "whisper (optional)" in out
    assert "disk space (optional)" in out


def test_cli_doctor_json_and_report_out(tmp_path):
    out = tmp_path / "doctor.json"
    assert main(["doctor", "--json", "--report-out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] in ("ok", "ok_with_warnings", "fail")
    assert {c["name"] for c in data["checks"]} == {
        "python",
        "ffmpeg",
        "ffprobe",
        "ffmpeg subtitles filter",
        "whisper",
        "argos-translate",
        "disk space",
    }


def test_cli_doctor_fail_on_warn_exits_1(tmp_path, capsys):
    # This machine has no whisper/argos, so the report carries warnings.
    assert main(["doctor", "--fail-on-warn"]) == 1
    assert "ok_with_warnings" in capsys.readouterr().out


def _write_level_transcript(path):
    from video_dubber.transcript import Segment, Word

    path.write_text(
        Transcript(
            segments=[
                Segment(
                    words=[
                        Word("quiet", 0.0, 0.25),
                        Word("line", 0.25, 0.5),
                    ]
                ),
                Segment(
                    words=[
                        Word("loud", 1.0, 1.25),
                        Word("line", 1.25, 1.5),
                    ]
                ),
            ],
            language="en",
        ).to_json(),
        encoding="utf-8",
    )


def test_cli_level_evens_segments(tmp_path, capsys):
    from video_dubber.level import measure_segment_levels

    src = tmp_path / "dub.wav"
    save_wav(
        str(src),
        concat(
            make_sine(440.0, 0.5, amplitude=0.05),
            make_silence(0.5),
            make_sine(440.0, 0.5, amplitude=0.5),
        ),
    )
    transcript = tmp_path / "dub.json"
    _write_level_transcript(transcript)
    out = tmp_path / "level.wav"
    report = tmp_path / "level.md"
    assert (
        main(
            [
                "level",
                str(src),
                str(transcript),
                "--out",
                str(out),
                "--max-gain",
                "20",
                "--report-out",
                str(report),
            ]
        )
        == 0
    )
    levelled = load_wav(str(out))
    assert levelled.duration == pytest.approx(1.5)
    after = measure_segment_levels(levelled, Transcript.from_json(transcript.read_text()))
    assert after[0].rms_dbfs == pytest.approx(-16.0, abs=0.5)
    assert after[1].rms_dbfs == pytest.approx(-16.0, abs=0.5)
    text = report.read_text(encoding="utf-8")
    assert "Max deviation" in text and "Segments levelled: 2" in text
    assert "wrote" in capsys.readouterr().out


def test_cli_level_json_report(tmp_path):
    src = tmp_path / "dub.wav"
    save_wav(str(src), make_sine(440.0, 1.5, amplitude=0.1))
    transcript = tmp_path / "dub.json"
    _write_level_transcript(transcript)
    out = tmp_path / "level.wav"
    report = tmp_path / "level.json"
    assert (
        main(
            [
                "level",
                str(src),
                str(transcript),
                "--out",
                str(out),
                "--json",
                "--report-out",
                str(report),
            ]
        )
        == 0
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["segments_levelled"] == 2
    assert data["target_dbfs"] == -16.0


def test_cli_level_bad_target_is_a_clean_error(tmp_path, capsys):
    src = tmp_path / "dub.wav"
    save_wav(str(src), make_sine(440.0, 1.5, amplitude=0.1))
    transcript = tmp_path / "dub.json"
    _write_level_transcript(transcript)
    out = tmp_path / "level.wav"
    assert main(["level", str(src), str(transcript), "--out", str(out), "--target", "3.0"]) == 1
    assert "must be <= 0" in capsys.readouterr().err


def test_cli_dub_with_level_keeps_timing(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--level",
            ]
        )
        == 0
    )
    out_text = capsys.readouterr().out
    assert "levelled: max deviation" in out_text
    # Timing preserved: the dub still spans the full 2.5 s transcript.
    assert load_wav(str(out)).duration == pytest.approx(2.5)


def test_cli_dub_level_target_without_level_fails(tmp_path, capsys):
    transcript = tmp_path / "hi.json"
    _write_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "dub.wav"),
                "--level-target",
                "-20",
            ]
        )
        == 2
    )
    assert "need --level" in capsys.readouterr().err


class TestCliStyles:
    def test_styles_lists_presets(self, tmp_path, capsys):
        assert main(["styles"]) == 0
        out = capsys.readouterr().out
        assert "5 dubbing style(s)" in out
        for name in ("documentary", "conversational", "energetic", "narration", "dramatic"):
            assert name in out
        assert "energetic: 1.15x rate, +2 semitones" in out

    def test_styles_json_format(self, tmp_path, capsys):
        assert main(["styles", "--format", "json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert [item["name"] for item in data] == [
            "documentary",
            "conversational",
            "energetic",
            "narration",
            "dramatic",
        ]
        assert data[2]["rate_factor"] == pytest.approx(1.15)

    def test_dub_with_style_records_it(self, tmp_path, capsys):
        transcript = tmp_path / "hi.json"
        _write_dub_transcript(transcript)
        profile = tmp_path / "dub.json"
        _write_dub_profile(profile)
        out = tmp_path / "dub.wav"
        report = tmp_path / "report.json"
        assert (
            main(
                [
                    "dub",
                    str(transcript),
                    "--profile",
                    str(profile),
                    "--style",
                    "energetic",
                    "--out",
                    str(out),
                    "--report-out",
                    str(report),
                ]
            )
            == 0
        )
        assert "delivery style: energetic" in capsys.readouterr().out
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["style"] == "energetic"

    def test_dub_unknown_style_fails(self, tmp_path, capsys):
        transcript = tmp_path / "hi.json"
        _write_dub_transcript(transcript)
        profile = tmp_path / "dub.json"
        _write_dub_profile(profile)
        assert (
            main(
                [
                    "dub",
                    str(transcript),
                    "--profile",
                    str(profile),
                    "--style",
                    "shouty",
                    "--out",
                    str(tmp_path / "dub.wav"),
                ]
            )
            == 2
        )
        assert "unknown dub style 'shouty'" in capsys.readouterr().err

    def test_dub_style_from_config_and_flag_beats_it(self, tmp_path, capsys):
        transcript = tmp_path / "hi.json"
        _write_dub_transcript(transcript)
        profile = tmp_path / "dub.json"
        _write_dub_profile(profile)
        config_path = tmp_path / "dubber.toml"
        config_path.write_text('[dubber]\nstyle = "narration"\n', encoding="utf-8")
        out = tmp_path / "dub.wav"
        report = tmp_path / "report.json"
        assert (
            main(
                [
                    "--config",
                    str(config_path),
                    "dub",
                    str(transcript),
                    "--profile",
                    str(profile),
                    "--out",
                    str(out),
                    "--report-out",
                    str(report),
                ]
            )
            == 0
        )
        assert json.loads(report.read_text(encoding="utf-8"))["style"] == "narration"
        # An explicit --style wins over the file.
        assert (
            main(
                [
                    "--config",
                    str(config_path),
                    "dub",
                    str(transcript),
                    "--profile",
                    str(profile),
                    "--style",
                    "energetic",
                    "--out",
                    str(out),
                    "--report-out",
                    str(report),
                ]
            )
            == 0
        )
        assert json.loads(report.read_text(encoding="utf-8"))["style"] == "energetic"

    def test_estimate_with_style_is_shorter(self, tmp_path, capsys):
        import re

        transcript = tmp_path / "hi.json"
        _write_dub_transcript(transcript)
        profile = tmp_path / "dub.json"
        _write_dub_profile(profile)
        numbers = {}
        for label, extra in (("plain", []), ("styled", ["--style", "energetic"])):
            assert main(["estimate", str(transcript), "--profile", str(profile), *extra]) == 0
            out = capsys.readouterr().out
            match = re.search(r"Predicted speech: ([0-9.]+)s", out)
            assert match is not None
            numbers[label] = float(match.group(1))
        assert numbers["styled"] < numbers["plain"]
        assert "Delivery style: energetic" in out

    def test_patch_with_style(self, tmp_path, capsys):
        dub, transcript, profile, _report = _make_dubbed(tmp_path)
        out = tmp_path / "patched.wav"
        assert (
            main(
                [
                    "patch",
                    "--dub",
                    str(dub),
                    "--transcript",
                    str(transcript),
                    "--profile",
                    str(profile),
                    "--segment",
                    "2",
                    "--text",
                    "बदला हुआ",
                    "--style",
                    "energetic",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert "patched segment 2" in capsys.readouterr().out
        assert load_wav(str(out)).duration == pytest.approx(load_wav(str(dub)).duration)


def _make_verify_fixtures(tmp_path, audio_hz=(220.0, 110.0)):
    """Labelled two-speaker dub + library; audio_hz picks the voice per segment."""
    from video_dubber.transcript import Segment, Word
    from video_dubber.voice import VoiceLibrary, clone_from_references

    audio = tmp_path / "dub.wav"
    save_wav(
        str(audio),
        concat(make_sine(audio_hz[0], 1.0), make_sine(audio_hz[1], 1.0)),
    )
    transcript = Transcript(
        segments=[
            Segment(
                words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)],
                speaker="Asha",
            ),
            Segment(
                words=[Word("good", 1.0, 1.5), Word("morning", 1.5, 2.0)],
                speaker="Ravi",
            ),
        ],
        language="en",
    )
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    library = VoiceLibrary()
    library.add(clone_from_references("Asha", [make_sine(220.0, 1.0)]))
    library.add(clone_from_references("Ravi", [make_sine(110.0, 1.0)]))
    library_path = tmp_path / "voices.json"
    library.save(str(library_path))
    return audio, transcript_path, library_path


def test_cli_verify_passes_a_correct_dub(tmp_path, capsys):
    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path)
    assert main(["verify", str(audio), str(transcript_path), str(library_path)]) == 0
    assert "Verdict: **pass**" in capsys.readouterr().out


def test_cli_verify_fails_a_wrong_voice_dub(tmp_path, capsys):
    # First segment claims Asha but is 440 Hz: a mismatch, not a match.
    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path, (440.0, 110.0))
    assert main(["verify", str(audio), str(transcript_path), str(library_path)]) == 1
    stdout = capsys.readouterr().out
    assert "Verdict: **review**" in stdout
    assert "[mismatch]" in stdout


def test_cli_verify_fail_on_uncertain(tmp_path, capsys):
    # 330 Hz for Asha is uncertain: passes by default, fails with the flag.
    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path, (330.0, 110.0))
    assert main(["verify", str(audio), str(transcript_path), str(library_path)]) == 0
    assert (
        main(
            [
                "verify",
                str(audio),
                str(transcript_path),
                str(library_path),
                "--fail-on-uncertain",
            ]
        )
        == 1
    )


def test_cli_verify_json_and_report_out(tmp_path, capsys):
    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path)
    report_path = tmp_path / "verify.json"
    assert (
        main(
            [
                "verify",
                str(audio),
                str(transcript_path),
                str(library_path),
                "--json",
                "--report-out",
                str(report_path),
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert '"verdict": "pass"' in stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "pass"
    assert report["matches"] == 2


def test_cli_verify_library_from_config(tmp_path, capsys):
    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path)
    config_path = tmp_path / "dubber.toml"
    config_path.write_text(f'[dubber]\nvoice_library = "{library_path}"\n', encoding="utf-8")
    assert main(["--config", str(config_path), "verify", str(audio), str(transcript_path)]) == 0
    assert "Verdict: **pass**" in capsys.readouterr().out


def test_cli_verify_no_library_anywhere_errors(tmp_path, capsys):
    audio, transcript_path, _library_path = _make_verify_fixtures(tmp_path)
    assert main(["verify", str(audio), str(transcript_path)]) == 2
    assert "needs a library path" in capsys.readouterr().err


def _write_labelled_dub_transcript(path):
    path.write_text(
        Transcript(
            segments=[
                {
                    "words": [
                        {"text": "नमस्ते", "start": 0.0, "end": 0.5},
                        {"text": "दुनिया", "start": 0.5, "end": 1.0},
                    ],
                    "speaker": "Dub",
                },
                {
                    "words": [{"text": "शानदार", "start": 2.0, "end": 2.5}],
                    "speaker": "Dub",
                },
            ],
            language="hi",
        ).to_json()
    )


def test_cli_dub_verify_passes_a_matching_voice(tmp_path, capsys):
    from video_dubber.voice import VoiceLibrary, clone_from_references

    transcript = tmp_path / "hi.json"
    _write_labelled_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    library = VoiceLibrary()
    library.add(clone_from_references("Dub", [make_sine(180.0, 1.0)]))
    library_path = tmp_path / "voices.json"
    library.save(str(library_path))
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--verify",
                "--verify-library",
                str(library_path),
            ]
        )
        == 0
    )
    assert "Verdict: **pass**" in capsys.readouterr().out
    assert out.exists()


def test_cli_dub_verify_fails_a_wrong_voice(tmp_path, capsys):
    from video_dubber.voice import VoiceLibrary, clone_from_references

    transcript = tmp_path / "hi.json"
    _write_labelled_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    # The dub renders at 440 Hz, but the library says Dub sounds like 180 Hz.
    _write_dub_profile(profile, pitch=440.0)
    library = VoiceLibrary()
    library.add(clone_from_references("Dub", [make_sine(180.0, 1.0)]))
    library_path = tmp_path / "voices.json"
    library.save(str(library_path))
    out = tmp_path / "dub.wav"
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(out),
                "--verify",
                "--verify-library",
                str(library_path),
            ]
        )
        == 1
    )
    stdout = capsys.readouterr().out
    assert "Verdict: **review**" in stdout
    # The dub is still written: verification gates, it does not delete.
    assert out.exists()


def test_cli_dub_verify_without_library_errors(tmp_path):
    transcript = tmp_path / "hi.json"
    _write_labelled_dub_transcript(transcript)
    profile = tmp_path / "dub.json"
    _write_dub_profile(profile)
    assert (
        main(
            [
                "dub",
                str(transcript),
                "--profile",
                str(profile),
                "--out",
                str(tmp_path / "dub.wav"),
                "--verify",
            ]
        )
        == 2
    )


def test_cli_verify_unknown_speaker_errors(tmp_path, capsys):
    from video_dubber.transcript import Segment, Word

    audio, transcript_path, library_path = _make_verify_fixtures(tmp_path)
    bad = Transcript(
        segments=[
            Segment(
                words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)],
                speaker="Nobody",
            )
        ],
        language="en",
    )
    transcript_path.write_text(bad.to_json(), encoding="utf-8")
    assert main(["verify", str(audio), str(transcript_path), str(library_path)]) == 1
    assert "no such profile" in capsys.readouterr().err
