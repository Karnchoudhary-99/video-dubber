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


def _write_dub_profile(path):
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    path.write_text(
        SpeakerProfile(
            name="Dub",
            embedding=[0.25] * EMBEDDING_DIM,
            base_pitch_hz=180.0,
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
