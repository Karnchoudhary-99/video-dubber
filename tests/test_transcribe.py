import json

import pytest

from video_dubber.audio import AudioData, concat, make_silence, make_sine, save_wav
from video_dubber.transcribe import (
    ScriptedTranscriber,
    TranscriptionError,
    WhisperCliTranscriber,
    _transcript_from_whisper_json,
    align_to_speech,
    transcribe_file,
)
from video_dubber.transcript import Word


def _whisper_json(with_words: bool = True) -> dict:
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": " hello world",
            "words": (
                [
                    {"word": " hello", "start": 0.0, "end": 0.4, "probability": 0.95},
                    {"word": " world", "start": 0.5, "end": 0.9, "probability": 0.9},
                ]
                if with_words
                else []
            ),
        }
    ]
    return {"language": "en", "segments": segments}


def test_whisper_json_parses_word_timestamps():
    t = _transcript_from_whisper_json(_whisper_json(), language="en")
    assert t.language == "en"
    assert t.text == "hello world"
    assert t.words[0].confidence == pytest.approx(0.95)
    assert len(t.segments) == 1


def test_whisper_json_falls_back_to_segment_text():
    t = _transcript_from_whisper_json(_whisper_json(with_words=False), language=None)
    assert t.text == "hello world"
    assert t.words[0].start == 0.0
    assert t.words[0].end == 1.0
    assert len(t.segments) == 1


def test_whisper_cli_missing_binary(monkeypatch):
    monkeypatch.setattr(
        "video_dubber.transcribe.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("whisper")),
    )
    transcriber = WhisperCliTranscriber(command="whisper")
    with pytest.raises(TranscriptionError, match="whisper CLI not found"):
        transcriber.transcribe(AudioData([0.0] * 1600, 16_000))


def test_whisper_cli_nonzero_exit(monkeypatch):
    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "model download failed"

    monkeypatch.setattr("video_dubber.transcribe.subprocess.run", lambda *a, **k: FakeResult())
    with pytest.raises(TranscriptionError, match="model download failed"):
        WhisperCliTranscriber().transcribe(AudioData([0.0] * 1600, 16_000))


def test_whisper_cli_parses_output(monkeypatch, tmp_path):
    """Fake whisper writes the JSON it would write; adapter must parse it."""
    calls = {}

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        out_dir = cmd[cmd.index("--output_dir") + 1]
        (tmp_path / "input.json").write_text(json.dumps(_whisper_json()))
        import shutil

        shutil.copy(tmp_path / "input.json", f"{out_dir}/input.json")
        return FakeResult()

    monkeypatch.setattr("video_dubber.transcribe.subprocess.run", fake_run)
    t = WhisperCliTranscriber(model="tiny").transcribe(
        AudioData([0.0] * 1600, 16_000), language="en"
    )
    assert t.text == "hello world"
    assert "--language" in calls["cmd"] and "en" in calls["cmd"]
    assert "tiny" in calls["cmd"]


def test_whisper_cli_missing_json_output(monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("video_dubber.transcribe.subprocess.run", lambda *a, **k: FakeResult())
    with pytest.raises(TranscriptionError, match="no usable JSON"):
        WhisperCliTranscriber().transcribe(AudioData([0.0] * 1600, 16_000))


def test_whisper_available_false_when_missing(monkeypatch):
    monkeypatch.setattr("video_dubber.transcribe.shutil.which", lambda _: None)
    assert WhisperCliTranscriber.available() is False


def test_scripted_transcriber_returns_words():
    words = [Word("alpha", 0.0, 0.3), Word("beta", 0.4, 0.8)]
    t = ScriptedTranscriber(words).transcribe(AudioData([], 16_000), language="hi")
    assert t.text == "alpha beta"
    assert t.language == "hi"


def test_scripted_transcriber_from_json_file(tmp_path):
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"text": "namaste", "start": 0.0, "end": 0.5}]))
    t = ScriptedTranscriber.from_json_file(str(script)).transcribe(AudioData([], 16_000))
    assert t.text == "namaste"
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(TranscriptionError, match="invalid script file"):
        ScriptedTranscriber.from_json_file(str(bad))


def test_transcribe_file_loads_and_tags_source(tmp_path):
    wav = tmp_path / "clip.wav"
    save_wav(str(wav), make_sine(440.0, 0.5))
    words = [Word("test", 0.0, 0.4)]
    t = transcribe_file(str(wav), ScriptedTranscriber(words))
    assert t.text == "test"
    assert t.source == str(wav)


def test_align_to_speech_drops_words_in_silence():
    audio = concat(make_sine(440.0, 1.0), make_silence(1.0), make_sine(440.0, 1.0))
    words = [
        Word("first", 0.1, 0.5),  # inside speech burst 1
        Word("ghost", 1.2, 1.6),  # in the silence gap — hallucination
        Word("second", 2.1, 2.5),  # inside speech burst 2
    ]
    aligned = align_to_speech(ScriptedTranscriber(words).transcribe(audio), audio)
    assert aligned.text == "first second"
    assert len(aligned.segments) == 2


def test_align_to_speech_with_no_speech_returns_empty():
    audio = make_silence(1.0)
    words = [Word("ghost", 0.1, 0.5)]
    aligned = align_to_speech(ScriptedTranscriber(words).transcribe(audio), audio)
    assert aligned.segments == []
