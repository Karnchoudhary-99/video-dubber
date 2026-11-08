"""Speech-to-text backends.

Transcription is deliberately pluggable: the pipeline only depends on the
:class:`Transcriber` protocol (audio in, :class:`Transcript` out).

* :class:`WhisperCliTranscriber` — shells out to the ``whisper`` CLI
  (openai-whisper) when it is installed and parses its word-timestamped
  JSON output. This is the real STT path; nothing here pretends to
  recognise speech on its own.
* :class:`ScriptedTranscriber` — a clearly-labelled demo/test double that
  returns a caller-supplied word list. Used by the demo and the test
  suite to exercise everything downstream of STT without a model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from video_dubber.audio import DEFAULT_SAMPLE_RATE, AudioData, load_wav, resample, save_wav
from video_dubber.transcript import Segment, Transcript, Word, from_words
from video_dubber.vad import SpeechSegment, VadConfig, detect_speech


class TranscriptionError(RuntimeError):
    """Raised when a transcription backend fails or is unavailable."""


class Transcriber(Protocol):
    """Anything that turns audio into a timed transcript."""

    def transcribe(self, audio: AudioData, language: str | None = None) -> Transcript: ...


class WhisperCliTranscriber:
    """Transcribe via the ``whisper`` CLI (openai-whisper, any model size).

    Runs ``whisper <wav> --model <model> --output_format json
    --word_timestamps True`` in a temp dir and parses the ``segments``
    (with per-word timings) out of the JSON it writes. The CLI must be on
    PATH; if it is missing or exits non-zero, :class:`TranscriptionError`
    is raised with the underlying message.
    """

    def __init__(self, model: str = "tiny", command: str = "whisper") -> None:
        self.model = model
        self.command = command

    def transcribe(self, audio: AudioData, language: str | None = None) -> Transcript:
        with tempfile.TemporaryDirectory(prefix="video-dubber-whisper-") as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"
            save_wav(str(wav_path), resample(audio, DEFAULT_SAMPLE_RATE))
            cmd = [
                self.command,
                str(wav_path),
                "--model",
                self.model,
                "--output_format",
                "json",
                "--word_timestamps",
                "True",
                "--output_dir",
                tmpdir,
            ]
            if language:
                cmd += ["--language", language]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise TranscriptionError(
                    f"whisper CLI not found (tried '{self.command}'); "
                    "install openai-whisper or use --scripted for a demo transcript"
                ) from exc
            if result.returncode != 0:
                raise TranscriptionError(
                    f"whisper failed (exit {result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            json_path = Path(tmpdir) / "input.json"
            try:
                raw = json.loads(json_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptionError(
                    f"whisper wrote no usable JSON output in {tmpdir}"
                ) from exc
        return _transcript_from_whisper_json(raw, language=language)

    @classmethod
    def available(cls, command: str = "whisper") -> bool:
        """True when the whisper CLI runs (``--help`` exits 0)."""
        if shutil.which(command) is None:
            return False
        try:
            result = subprocess.run(
                [command, "--help"], capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            return False
        return result.returncode == 0


def _transcript_from_whisper_json(raw: dict, language: str | None) -> Transcript:
    words: list[Word] = []
    for segment in raw.get("segments", []):
        for w in segment.get("words", []):
            words.append(
                Word(
                    text=str(w["word"]).strip(),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w.get("probability", 1.0)),
                )
            )
    if not words:  # whisper without word timestamps: fall back to segment text
        for segment in raw.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if text:
                words.append(
                    Word(
                        text=text,
                        start=float(segment.get("start", 0.0)),
                        end=float(segment.get("end", 0.0)),
                    )
                )
    return from_words(words, language=language or raw.get("language"))


class ScriptedTranscriber:
    """Demo/test double: returns a fixed word list as a transcript.

    NOT speech recognition — the words come from the caller (a script file
    or a test fixture). Its only job is to let the rest of the pipeline
    (VAD alignment, subtitle export, dub assembly) run without a model.
    """

    def __init__(self, words: list[Word]) -> None:
        self._words = list(words)

    def transcribe(self, audio: AudioData, language: str | None = None) -> Transcript:
        return from_words(self._words, language=language)

    @classmethod
    def from_json_file(cls, path: str) -> ScriptedTranscriber:
        """Load a scripted word list: JSON list of {text, start, end}."""
        try:
            raw = json.loads(Path(path).read_text())
            words = [Word(**entry) for entry in raw]
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
            raise TranscriptionError(f"invalid script file {path}: {exc}") from exc
        return cls(words)


def transcribe_file(
    path: str,
    transcriber: Transcriber,
    language: str | None = None,
) -> Transcript:
    """Load a WAV file (resampled to 16 kHz) and transcribe it."""
    audio = resample(load_wav(path), DEFAULT_SAMPLE_RATE)
    transcript = transcriber.transcribe(audio, language=language)
    transcript.source = path
    return transcript


def align_to_speech(
    transcript: Transcript,
    audio: AudioData,
    config: VadConfig | None = None,
) -> Transcript:
    """Re-segment a transcript's words at VAD speech boundaries.

    Words falling in silence gaps are dropped (they are usually
    hallucinations); words inside one speech stretch become one segment.
    Returns a new transcript; the input is unchanged.
    """
    speech = detect_speech(audio, config)
    if not speech:
        return Transcript(segments=[], language=transcript.language, source=transcript.source)
    segments: list[Segment] = []
    for span in speech:
        words = _words_in_span(transcript.words, span)
        if words:
            segments.append(Segment(words))
    return Transcript(segments=segments, language=transcript.language, source=transcript.source)


def _words_in_span(words: list[Word], span: SpeechSegment) -> list[Word]:
    return [w for w in words if w.start >= span.start and w.end <= span.end]
