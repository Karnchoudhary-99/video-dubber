"""Tests for translation fidelity: round-trip scoring and reporting."""

import json

import pytest

from video_dubber.fidelity import (
    FidelityError,
    FidelityReport,
    check_fidelity,
    reverse_translator,
    round_trip_text,
    word_jaccard,
)
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import ArgosTranslator, DictionaryTranslator, translate_transcript


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def _transcript():
    return Transcript(
        segments=[
            Segment(words=_words(["Hello", "world"], 0.0), speaker="A"),
            Segment(words=_words(["tomorrow"], 2.0), speaker="A"),
        ],
        language="en",
    )


# --- word_jaccard -----------------------------------------------------------


class TestWordJaccard:
    def test_identical_scores_one(self):
        assert word_jaccard("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert word_jaccard("Hello World", "hello world") == 1.0

    def test_disjoint_scores_zero(self):
        assert word_jaccard("hello world", "mars game") == 0.0

    def test_partial_overlap(self):
        assert word_jaccard("hello world", "hello mars") == pytest.approx(1 / 3)

    def test_both_empty_scores_one(self):
        assert word_jaccard("", "") == 1.0

    def test_one_empty_scores_zero(self):
        assert word_jaccard("hello", "") == 0.0


# --- reverse_translator -----------------------------------------------------


class TestReverseTranslator:
    def test_dictionary_reverses_to_hindi_source(self):
        forward = DictionaryTranslator(source="en", target="hi")
        back = reverse_translator(forward)
        assert back.translate("नमस्ते दुनिया", "hi", "en") == "hello world"

    def test_dictionary_reversed_method(self):
        back = DictionaryTranslator(source="en", target="hi").reversed()
        assert back.source == "hi"
        assert back.target == "en"
        assert back.translate("सुप्रभात", "hi", "en") == "good morning"

    def test_argos_reverses_to_fresh_adapter(self):
        back = reverse_translator(ArgosTranslator(timeout=5.0))
        assert isinstance(back, ArgosTranslator)
        assert back.timeout == 5.0

    def test_unknown_translator_raises(self):
        class CustomTranslator:
            name = "custom"

            def translate(self, text, source, target):
                return text

        with pytest.raises(FidelityError, match="cannot reverse"):
            reverse_translator(CustomTranslator())


# --- round_trip_text --------------------------------------------------------


class TestRoundTripText:
    def test_known_sentence_round_trips_cleanly(self):
        translator = DictionaryTranslator(source="en", target="hi")
        translated, back, score = round_trip_text(
            "hello world good morning", translator, "en", "hi"
        )
        assert translated == "नमस्ते दुनिया सुप्रभात"
        assert back == "hello world good morning"
        assert score == 1.0

    def test_empty_text_scores_one(self):
        translator = DictionaryTranslator(source="en", target="hi")
        assert round_trip_text("   ", translator, "en", "hi") == ("   ", "   ", 1.0)

    def test_dictionary_collision_scores_zero(self):
        # "tomorrow" and "yesterday" both become "कल"; the inversion can
        # only recover one of them, so "tomorrow" comes back as "yesterday".
        # The check surfacing this is the feature working, not a bug.
        translator = DictionaryTranslator(source="en", target="hi")
        translated, back, score = round_trip_text("tomorrow", translator, "en", "hi")
        assert translated == "कल"
        assert back == "yesterday"
        assert score == 0.0

    def test_unknown_word_passes_through_both_ways(self):
        # Honest limit: a word the dictionary never knew scores 1.0
        # because it survives untouched — stability, not correctness.
        translator = DictionaryTranslator(source="en", target="hi")
        _, back, score = round_trip_text("zqxj", translator, "en", "hi")
        assert back == "zqxj"
        assert score == 1.0


# --- check_fidelity ---------------------------------------------------------


class TestCheckFidelity:
    def test_flags_collision_segment_only(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        report = check_fidelity(source, translated, translator, target="hi")
        assert report.source == "en"
        assert report.target == "hi"
        assert report.mean_score == pytest.approx(0.5)
        assert report.min_score == 0.0
        assert report.verdict == "review"
        assert len(report.flagged) == 1
        assert report.flagged[0].index == 1
        assert report.flagged[0].source_text == "tomorrow"
        assert report.flagged[0].back_text == "yesterday"
        assert report.segments[0].score == 1.0
        assert not report.segments[0].flagged

    def test_clean_transcript_verdict_good(self):
        source = Transcript(
            segments=[Segment(words=_words(["Hello", "world"], 0.0))],
            language="en",
        )
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        report = check_fidelity(source, translated, translator, target="hi")
        assert report.verdict == "good"
        assert report.mean_score == 1.0
        assert report.flagged == []

    def test_threshold_controls_flagging(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        # With threshold 0.0 nothing is flagged, even the 0.0 segment.
        report = check_fidelity(source, translated, translator, target="hi", threshold=0.0)
        assert report.verdict == "good"

    def test_threshold_out_of_range_raises(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        with pytest.raises(ValueError, match="threshold"):
            check_fidelity(source, translated, translator, target="hi", threshold=1.5)

    def test_segment_count_mismatch_raises(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        translated.segments.pop()
        with pytest.raises(FidelityError, match="segment count mismatch"):
            check_fidelity(source, translated, translator, target="hi")

    def test_empty_segments_score_one(self):
        source = Transcript(segments=[Segment(words=[])], language="en")
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        report = check_fidelity(source, translated, translator, target="hi")
        assert report.segments[0].score == 1.0
        assert report.verdict == "good"


# --- FidelityReport rendering -----------------------------------------------


class TestFidelityReportRendering:
    def test_json_round_trip(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        report = check_fidelity(source, translated, translator, target="hi")
        data = json.loads(report.to_json())
        assert data["verdict"] == "review"
        assert data["mean_score"] == pytest.approx(0.5)
        assert data["flagged_count"] == 1
        assert data["segments"][1]["back_text"] == "yesterday"

    def test_markdown_names_flagged_segments(self):
        source = _transcript()
        translator = DictionaryTranslator(source="en", target="hi")
        translated = translate_transcript(source, translator, target="hi")
        report = check_fidelity(source, translated, translator, target="hi")
        text = report.to_markdown()
        assert "## Needs review" in text
        assert "Segment 1" in text
        assert "tomorrow" in text
        assert "yesterday" in text

    def test_empty_report_is_good(self):
        report = FidelityReport(source="en", target="hi", backend="dictionary", threshold=0.6)
        assert report.verdict == "good"
        assert report.mean_score == 1.0
        assert "Flagged segments: 0 of 0" in report.to_markdown()
