"""Tests for multi-target dubbing: one source, several languages, one summary."""

import json

import pytest

from video_dubber.audio import load_wav
from video_dubber.cli import main
from video_dubber.multilang import (
    MultiTargetError,
    dub_all_targets,
    parse_targets,
)
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, TranslationError
from video_dubber.voice import EMBEDDING_DIM, ScriptedVoiceCloner, SpeakerProfile


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(words=_words(["Hello", "and", "welcome"], start=0.0)),
            Segment(words=_words(["to", "Mars"], start=1.5)),
        ],
        language="en",
    )


def _profile() -> SpeakerProfile:
    return SpeakerProfile(name="Test", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)


def _factory(source="en"):
    def make(target: str) -> DictionaryTranslator:
        return DictionaryTranslator(source=source, target=target)

    return make


def _run(tmp_path, targets=("hi", "es"), **kwargs):
    kwargs.setdefault("original", None)
    return dub_all_targets(
        _transcript(),
        list(targets),
        _factory(),
        ScriptedVoiceCloner(),
        _profile(),
        str(tmp_path / "out"),
        **kwargs,
    )


class TestParseTargets:
    def test_dedupes_and_normalises(self):
        assert parse_targets("hi, es,HI, fr ") == ["hi", "es", "fr"]

    def test_empty_raises(self):
        with pytest.raises(MultiTargetError, match="no target languages"):
            parse_targets(" , ,")

    def test_single_target(self):
        assert parse_targets("hi") == ["hi"]


class TestDubAllTargets:
    def test_two_targets_write_bundles_and_summary(self, tmp_path):
        summary = _run(tmp_path)
        assert summary.succeeded == 2
        assert summary.failed == 0
        assert summary.targets == ["hi", "es"]
        assert summary.source_language == "en"
        assert summary.mean_mos is not None
        assert set(summary.verdicts) == {"hi", "es"}
        out = tmp_path / "out"
        for target in ("hi", "es"):
            assert (out / f"dub.{target}.wav").exists()
            assert (out / f"dub.{target}.report.json").exists()
            assert (out / f"dub.{target}.srt").exists()
            assert (out / f"dub.{target}.vtt").exists()
            report = json.loads((out / f"dub.{target}.report.json").read_text())
            assert report["segments_dubbed"] == 2
            assert report["translation_coverage"] is not None
        for result in summary.results:
            assert load_wav(result.dub_path).duration == pytest.approx(2.5, abs=0.05)

    def test_summary_json_and_markdown_round_trip(self, tmp_path):
        summary = _run(tmp_path)
        parsed = json.loads(summary.to_json())
        assert parsed["succeeded"] == 2
        assert parsed["results"][0]["target"] == "hi"
        assert parsed["failures"] == {}
        md = summary.to_markdown()
        assert "| hi |" in md and "| es |" in md
        assert "Source language: `en`" in md

    def test_duplicate_targets_dubbed_once(self, tmp_path):
        summary = _run(tmp_path, targets=("hi", "hi", "es"))
        assert summary.targets == ["hi", "es"]
        assert summary.succeeded == 2

    def test_target_equal_to_source_is_an_error(self, tmp_path):
        with pytest.raises(MultiTargetError, match="source language"):
            _run(tmp_path, targets=("en",))

    def test_empty_target_list_is_an_error(self, tmp_path):
        with pytest.raises(MultiTargetError, match="no target languages"):
            _run(tmp_path, targets=())

    def test_unknown_source_language_is_an_error(self, tmp_path):
        transcript = _transcript()
        transcript.language = ""
        with pytest.raises(MultiTargetError, match="source language unknown"):
            dub_all_targets(
                transcript,
                ["hi"],
                _factory(),
                ScriptedVoiceCloner(),
                _profile(),
                str(tmp_path / "out"),
            )

    def test_one_failing_target_does_not_kill_the_others(self, tmp_path):
        def flaky(target: str):
            if target == "xx":
                raise TranslationError("no dictionary for en->xx")
            return DictionaryTranslator(source="en", target=target)

        summary = dub_all_targets(
            _transcript(),
            ["xx", "hi"],
            flaky,
            ScriptedVoiceCloner(),
            _profile(),
            str(tmp_path / "out"),
        )
        assert summary.succeeded == 1
        assert summary.failed == 1
        assert summary.results[0].target == "hi"
        assert "xx" in summary.failures
        assert "no dictionary" in summary.failures["xx"]
        assert "Failures" in summary.to_markdown()
        out = tmp_path / "out"
        assert (out / "dub.hi.wav").exists()
        assert not (out / "dub.xx.wav").exists()

    def test_stop_on_error_aborts_before_later_targets(self, tmp_path):
        def flaky(target: str):
            if target == "xx":
                raise TranslationError("no dictionary for en->xx")
            return DictionaryTranslator(source="en", target=target)

        with pytest.raises(MultiTargetError, match="target 'xx' failed"):
            dub_all_targets(
                _transcript(),
                ["xx", "hi"],
                flaky,
                ScriptedVoiceCloner(),
                _profile(),
                str(tmp_path / "out"),
                stop_on_error=True,
            )
        assert not (tmp_path / "out" / "dub.hi.wav").exists()

    def test_no_subtitles_writes_no_subtitle_files(self, tmp_path):
        summary = _run(tmp_path, targets=("hi",), subtitles=False)
        assert summary.results[0].subtitle_paths == []
        assert not (tmp_path / "out" / "dub.hi.srt").exists()

    def test_enhance_keeps_timing_and_records_it(self, tmp_path):
        summary = _run(tmp_path, targets=("hi",), enhance=True)
        report = summary.results[0].report
        assert report.enhancement is not None
        assert report.total_duration == pytest.approx(2.5, abs=0.05)


class TestDubMultiCli:
    def _write_inputs(self, tmp_path):
        transcript = tmp_path / "source.json"
        transcript.write_text(_transcript().to_json(), encoding="utf-8")
        profile = tmp_path / "voice.json"
        profile.write_text(_profile().to_json(), encoding="utf-8")
        return transcript, profile

    def test_cli_dubs_two_targets_and_writes_summary(self, tmp_path, capsys):
        transcript, profile = self._write_inputs(tmp_path)
        out_dir = tmp_path / "bundles"
        assert (
            main(
                [
                    "dub-multi",
                    str(transcript),
                    "--targets",
                    "hi,es",
                    "--profile",
                    str(profile),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            == 0
        )
        assert (out_dir / "dub.hi.wav").exists()
        assert (out_dir / "dub.es.wav").exists()
        assert (out_dir / "dub.multilang.json").exists()
        assert (out_dir / "dub.multilang.md").exists()
        parsed = json.loads((out_dir / "dub.multilang.json").read_text())
        assert parsed["succeeded"] == 2
        assert "dubbed 2 of 2 targets" in capsys.readouterr().out

    def test_cli_dedupes_targets(self, tmp_path):
        transcript, profile = self._write_inputs(tmp_path)
        out_dir = tmp_path / "bundles"
        assert (
            main(
                [
                    "dub-multi",
                    str(transcript),
                    "--targets",
                    "hi,hi,es",
                    "--profile",
                    str(profile),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            == 0
        )
        parsed = json.loads((out_dir / "dub.multilang.json").read_text())
        assert parsed["targets"] == ["hi", "es"]

    def test_cli_target_equal_to_source_fails(self, tmp_path):
        transcript, profile = self._write_inputs(tmp_path)
        assert (
            main(
                [
                    "dub-multi",
                    str(transcript),
                    "--targets",
                    "en",
                    "--profile",
                    str(profile),
                    "--out-dir",
                    str(tmp_path / "bundles"),
                ]
            )
            == 1
        )

    def test_cli_empty_targets_is_usage_error(self, tmp_path):
        transcript, profile = self._write_inputs(tmp_path)
        assert (
            main(
                [
                    "dub-multi",
                    str(transcript),
                    "--targets",
                    " , ",
                    "--profile",
                    str(profile),
                    "--out-dir",
                    str(tmp_path / "bundles"),
                ]
            )
            == 2
        )
