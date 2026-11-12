"""Tests for the pronunciation lexicon: validation, matching, application."""

import json

import pytest

from video_dubber.assemble import DubReport, build_report, dub_transcript
from video_dubber.cli import main
from video_dubber.lexicon import (
    LexiconApplication,
    LexiconError,
    PronunciationEntry,
    PronunciationLexicon,
    apply_lexicon,
    load_lexicon,
)
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import EMBEDDING_DIM, ScriptedVoiceCloner, SpeakerProfile


def _lexicon(**entries: str) -> PronunciationLexicon:
    return PronunciationLexicon(entries=dict(entries))


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def _profile() -> SpeakerProfile:
    return SpeakerProfile(name="Test", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)


class TestEntryValidation:
    def test_entry_normalises_whitespace(self):
        entry = PronunciationEntry(word="  New   Delhi ", pronunciation="  nyoo delhi ")
        assert entry.word == "New Delhi"
        assert entry.pronunciation == "nyoo delhi"
        assert entry.phrase_words == 2

    def test_empty_word_rejected(self):
        with pytest.raises(LexiconError, match="must not be empty"):
            PronunciationEntry(word="   ", pronunciation="something")

    def test_empty_pronunciation_rejected(self):
        with pytest.raises(LexiconError, match="must not be empty"):
            PronunciationEntry(word="NASA", pronunciation="")

    def test_word_needs_a_letter(self):
        with pytest.raises(LexiconError, match="at least one letter"):
            PronunciationEntry(word="123", pronunciation="one two three")

    def test_pronunciation_needs_a_letter(self):
        with pytest.raises(LexiconError, match="at least one letter"):
            PronunciationEntry(word="NASA", pronunciation="123")

    def test_control_characters_rejected(self):
        with pytest.raises(LexiconError, match="control characters"):
            PronunciationEntry(word="NA\x01SA", pronunciation="nasa")

    def test_unsupported_word_characters_rejected(self):
        with pytest.raises(LexiconError, match="unsupported characters"):
            PronunciationEntry(word="NASA!", pronunciation="nasa")


class TestLexiconLoading:
    def test_from_dict_with_entries_key(self):
        lexicon = PronunciationLexicon.from_dict({"entries": {"NASA": "N A S A"}})
        assert lexicon.get("nasa") == "N A S A"
        assert len(lexicon) == 1

    def test_from_dict_bare_mapping(self):
        assert _lexicon(NASA="N A S A").get("NASA") == "N A S A"

    def test_duplicate_words_case_insensitively_rejected(self):
        with pytest.raises(LexiconError, match="duplicate lexicon word"):
            PronunciationLexicon(entries={"NASA": "nasa", "nasa": "N A S A"})

    def test_non_object_rejected(self):
        with pytest.raises(LexiconError, match="must be an object"):
            PronunciationLexicon.from_dict({"entries": ["NASA"]})  # type: ignore[arg-type]

    def test_invalid_json_rejected(self):
        with pytest.raises(LexiconError, match="invalid lexicon JSON"):
            PronunciationLexicon.from_json("{not json")

    def test_json_round_trip(self):
        original = _lexicon(NASA="N A S A", Karn="kar-n")
        rebuilt = PronunciationLexicon.from_json(original.to_json())
        assert rebuilt.to_dict() == original.to_dict()

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(LexiconError, match="cannot read lexicon"):
            load_lexicon(str(tmp_path / "nope.json"))

    def test_load_file(self, tmp_path):
        path = tmp_path / "lex.json"
        path.write_text(json.dumps({"entries": {"Nguyen": "win"}}), encoding="utf-8")
        assert load_lexicon(str(path)).get("nguyen") == "win"

    def test_contains_and_get(self):
        lexicon = _lexicon(NASA="N A S A")
        assert "nasa" in lexicon
        assert "NASA" in lexicon
        assert "ISRO" not in lexicon
        assert lexicon.get("ISRO") is None

    def test_with_entry_adds_and_overwrites(self):
        lexicon = _lexicon(NASA="nasa").with_entry(
            PronunciationEntry(word="NASA", pronunciation="N A S A")
        )
        assert lexicon.get("nasa") == "N A S A"
        assert len(lexicon) == 1


class TestApplyLexicon:
    def test_single_word_case_insensitive(self):
        applied = apply_lexicon("NASA launched", _lexicon(NASA="N A S A"))
        assert applied.text == "N A S A launched"
        assert applied.applied_count == 1
        assert applied.replacements == [("NASA", "N A S A")]

    def test_lowercase_match_keeps_lexicon_spelling(self):
        applied = apply_lexicon("nasa launched", _lexicon(NASA="N A S A"))
        assert applied.text == "N A S A launched"

    def test_capitalised_match_capitalises_pronunciation(self):
        applied = apply_lexicon("Karn speaks", _lexicon(Karn="kar-n"))
        assert applied.text == "Kar-n speaks"
        assert applied.replacements == [("Karn", "Kar-n")]

    def test_all_caps_match_uppercases_pronunciation(self):
        applied = apply_lexicon("KARN speaks", _lexicon(Karn="kar-n"))
        assert applied.text == "KAR-N speaks"

    def test_word_boundaries_respected(self):
        applied = apply_lexicon("NASAL spray", _lexicon(NASA="N A S A"))
        assert applied.text == "NASAL spray"
        assert applied.applied_count == 0

    def test_punctuation_boundaries(self):
        applied = apply_lexicon("(NASA), NASA. NASA!", _lexicon(NASA="N A S A"))
        assert applied.text == "(N A S A), N A S A. N A S A!"
        assert applied.applied_count == 3

    def test_longest_phrase_wins(self):
        lexicon = _lexicon(**{"New Delhi": "nyoo delhi", "Delhi": "deli"})
        applied = apply_lexicon("New Delhi is hot", lexicon)
        # "New Delhi" is capitalised, so the pronunciation is capitalised too.
        assert applied.text == "Nyoo delhi is hot"
        assert applied.replacements == [("New Delhi", "Nyoo delhi")]

    def test_phrase_needs_single_spaces(self):
        # Two spaces between the words is not the phrase — only "Delhi" matches.
        lexicon = _lexicon(**{"New Delhi": "nyoo delhi", "Delhi": "deli"})
        applied = apply_lexicon("New  Delhi", lexicon)
        assert applied.text == "New  Deli"
        assert applied.replacements == [("Delhi", "Deli")]

    def test_replacement_is_never_re_scanned(self):
        # "A" is an entry too, but the "N A S A" it expands to is output, not input.
        lexicon = _lexicon(NASA="N A S A", A="ay")
        applied = apply_lexicon("NASA", lexicon)
        assert applied.text == "N A S A"
        assert applied.applied_count == 1

    def test_devanagari_words_stay_whole(self):
        # The grapheme-aware tokenizer keeps combining marks with their
        # base letter — the old \\b regex shattered "नमस्ते" and never matched.
        applied = apply_lexicon("नमस्ते दुनिया", _lexicon(नमस्ते="namaste"))
        assert applied.text == "namaste दुनिया"
        assert applied.applied_count == 1

    def test_possessive_is_a_different_word(self):
        # "Karn's" tokenises as one word, so the "Karn" entry does not match
        # it — entries match whole words, never a prefix of one.
        applied = apply_lexicon("Karn's rocket", _lexicon(Karn="kar-n"))
        assert applied.text == "Karn's rocket"
        assert applied.applied_count == 0

    def test_empty_lexicon_is_identity(self):
        assert apply_lexicon("NASA", _lexicon()).text == "NASA"

    def test_empty_text(self):
        assert apply_lexicon("", _lexicon(NASA="N A S A")).applied_count == 0


class TestLexiconApplicationReport:
    def test_json_round_trip(self):
        applied = apply_lexicon("NASA and Karn", _lexicon(NASA="N A S A", Karn="kar-n"))
        rebuilt = LexiconApplication.from_json(applied.to_json())
        assert rebuilt.text == applied.text
        assert rebuilt.replacements == applied.replacements

    def test_markdown_lists_replacements(self):
        applied = apply_lexicon("NASA", _lexicon(NASA="N A S A"))
        md = applied.to_markdown()
        assert "Replacements: 1" in md
        assert "| NASA | N A S A |" in md

    def test_invalid_application_json_rejected(self):
        with pytest.raises(LexiconError, match="needs a 'text' string"):
            LexiconApplication.from_json('{"replacements": []}')


class TestDubWithLexicon:
    def test_dub_speaks_the_lexicon_pronunciation(self):
        # "NASA" (1 word) becomes "N A S A" (4 words): the raw synthesis
        # is longer, so the dub needs a stronger time-stretch to fit the
        # same window — the fit factor is the honest signal (the placed
        # dub is clamped to the window either way, so durations match).
        from video_dubber.voice import synthesize_text

        transcript = Transcript(segments=[Segment(words=_words(["NASA", "launched"], start=0.0))])
        profile = _profile()
        lexicon = _lexicon(NASA="N A S A")
        plain_text = "NASA launched"
        spoken_text = apply_lexicon(plain_text, lexicon).text
        assert spoken_text == "N A S A launched"
        assert (
            synthesize_text(spoken_text, profile).duration
            > synthesize_text(plain_text, profile).duration
        )
        _, plain_fit, _ = dub_transcript(transcript, ScriptedVoiceCloner(), profile)
        _, lex_fit, _ = dub_transcript(transcript, ScriptedVoiceCloner(), profile, lexicon=lexicon)
        assert lex_fit[0] < plain_fit[0]

    def test_report_records_lexicon_count(self):
        transcript = Transcript(segments=[Segment(words=_words(["NASA", "launched"], start=0.0))])
        lexicon = _lexicon(NASA="N A S A")
        audio, fit, placed = dub_transcript(
            transcript, ScriptedVoiceCloner(), _profile(), lexicon=lexicon
        )
        # The caller counts with the same pure function dub_transcript uses.
        applied = sum(
            apply_lexicon(segment.text, lexicon).applied_count for segment in transcript.segments
        )
        assert applied == 1
        report = build_report(transcript, placed, fit, audio, lexicon_applied=applied)
        assert report.lexicon_applied == 1
        assert "Lexicon pronunciations applied: 1" in report.to_markdown()
        assert json.loads(report.to_json())["lexicon_applied"] == 1
        assert DubReport.from_json(report.to_json()).lexicon_applied == 1

    def test_old_report_without_lexicon_loads_as_zero(self):
        report_json = json.dumps(
            {
                "segments_dubbed": 1,
                "total_duration": 1.0,
                "mean_sync_drift_s": 0.0,
                "max_sync_drift_s": 0.0,
                "segments_over_drift": 0,
                "mean_fit_factor": 1.0,
                "max_fit_factor": 1.0,
                "extreme_fits": 0,
                "loudness_error_db": None,
                "translation_coverage": None,
                "word_error_rate": None,
                "speakers_dubbed": 0,
                "bed_level": None,
                "style": None,
                "enhancement": None,
                "mos_proxy": 5.0,
                "verdict": "good",
                "segments": [],
            }
        )
        assert DubReport.from_json(report_json).lexicon_applied == 0


class TestLexiconCli:
    def test_lexicon_lists_entries(self, tmp_path, capsys):
        path = tmp_path / "lex.json"
        path.write_text(
            json.dumps({"entries": {"NASA": "N A S A", "Karn": "kar-n"}}),
            encoding="utf-8",
        )
        assert main(["lexicon", str(path)]) == 0
        out = capsys.readouterr().out
        assert "2 pronunciation(s)" in out
        # Keys are stored normalised (lowercase) — lookup is case-insensitive.
        assert "nasa -> N A S A" in out
        assert "karn -> kar-n" in out

    def test_lexicon_apply_text(self, tmp_path, capsys):
        path = tmp_path / "lex.json"
        path.write_text(json.dumps({"entries": {"NASA": "N A S A"}}), encoding="utf-8")
        assert main(["lexicon", str(path), "--apply", "NASA launched"]) == 0
        out = capsys.readouterr().out
        assert "N A S A launched" in out
        assert "NASA -> N A S A" in out

    def test_lexicon_apply_json(self, tmp_path, capsys):
        path = tmp_path / "lex.json"
        path.write_text(json.dumps({"entries": {"NASA": "N A S A"}}), encoding="utf-8")
        assert main(["lexicon", str(path), "--apply", "NASA", "--format", "json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["applied_count"] == 1
        assert data["text"] == "N A S A"

    def test_lexicon_missing_file_is_exit_2(self, tmp_path, capsys):
        assert main(["lexicon", str(tmp_path / "nope.json")]) == 2
        assert "error:" in capsys.readouterr().err

    def test_dub_with_lexicon_flag(self, tmp_path, capsys):
        lexicon_path = tmp_path / "lex.json"
        lexicon_path.write_text(json.dumps({"entries": {"NASA": "N A S A"}}), encoding="utf-8")
        transcript_path = tmp_path / "t.json"
        transcript = Transcript(segments=[Segment(words=_words(["NASA", "launched"], start=0.0))])
        transcript_path.write_text(transcript.to_json(), encoding="utf-8")
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(_profile().to_json(), encoding="utf-8")
        out = tmp_path / "dub.wav"
        assert (
            main(
                [
                    "dub",
                    str(transcript_path),
                    "--profile",
                    str(profile_path),
                    "--lexicon",
                    str(lexicon_path),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        assert "lexicon pronunciations applied: 1" in capsys.readouterr().out
        assert out.exists()

    def test_dub_with_bad_lexicon_is_exit_2(self, tmp_path, capsys):
        transcript_path = tmp_path / "t.json"
        transcript = Transcript(segments=[Segment(words=_words(["hi"], start=0.0))])
        transcript_path.write_text(transcript.to_json(), encoding="utf-8")
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(_profile().to_json(), encoding="utf-8")
        assert (
            main(
                [
                    "dub",
                    str(transcript_path),
                    "--profile",
                    str(profile_path),
                    "--lexicon",
                    str(tmp_path / "nope.json"),
                    "--out",
                    str(tmp_path / "dub.wav"),
                ]
            )
            == 2
        )
        assert "error:" in capsys.readouterr().err
