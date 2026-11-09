"""Tests for the translation pipeline: dictionary, glossary, timing, chunking."""

import json

import pytest

from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import (
    ArgosTranslator,
    DictionaryTranslator,
    Glossary,
    TranslationError,
    chunk_segment,
    translate_text,
    translate_transcript,
)


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def _transcript():
    return Transcript(
        segments=[
            Segment(words=_words(["Hello", "world"], 0.0), speaker="A"),
            Segment(words=_words(["Good", "morning", "team"], 2.0), speaker="B"),
        ],
        language="en",
    )


# --- DictionaryTranslator ---------------------------------------------------


class TestDictionaryTranslator:
    def test_word_translation_en_hi(self):
        t = DictionaryTranslator(source="en", target="hi")
        assert t.translate("hello", "en", "hi") == "नमस्ते"
        assert t.translate("Hello", "en", "hi") == "नमस्ते"

    def test_phrase_beats_single_word(self):
        t = DictionaryTranslator(source="en", target="hi")
        assert t.translate("good morning", "en", "hi") == "सुप्रभात"
        assert t.translate("Good morning team", "en", "hi") == "सुप्रभात टीम"

    def test_unknown_words_pass_through_and_are_reported(self):
        t = DictionaryTranslator(source="en", target="hi")
        out = t.translate("hello zqxj", "en", "hi")
        assert out == "नमस्ते zqxj"
        assert t.last_unknown == ["zqxj"]
        assert t.last_coverage == pytest.approx(0.5)

    def test_full_coverage_for_known_sentence(self):
        t = DictionaryTranslator(source="en", target="es")
        t.translate("hello world", "en", "es")
        assert t.last_coverage == 1.0
        assert t.last_unknown == []

    def test_case_preserved_for_uppercase(self):
        t = DictionaryTranslator(source="en", target="es")
        assert t.translate("HELLO", "en", "es") == "HOLA"
        assert t.translate("Hello", "en", "es") == "Hola"

    def test_punctuation_and_spacing_survive(self):
        t = DictionaryTranslator(source="en", target="es")
        assert t.translate("Hello, world!", "en", "es") == "Hola, mundo!"

    def test_empty_text_is_a_noop(self):
        t = DictionaryTranslator(source="en", target="hi")
        assert t.translate("   ", "en", "hi") == "   "

    def test_unsupported_pair_raises(self):
        t = DictionaryTranslator(source="en", target="hi")
        with pytest.raises(TranslationError, match="no en->fr table"):
            t.translate("hello", "en", "fr")

    def test_custom_dictionary(self):
        t = DictionaryTranslator(source="en", target="xx", dictionary={"hello": "xyzzy"})
        assert t.translate("hello", "en", "xx") == "xyzzy"


# --- Glossary ----------------------------------------------------------------


class TestGlossary:
    def test_term_override_applied_before_dictionary(self):
        g = Glossary(terms={"Mars Rovers": "मार्स रोवर्स"})
        out = translate_text(
            "Mars Rovers win the match",
            DictionaryTranslator(source="en", target="hi"),
            "en",
            "hi",
            glossary=g,
        )
        assert "मार्स रोवर्स" in out
        assert "जीत" in out  # win still translated by the dictionary
        assert g.last_applied == ["Mars Rovers"]

    def test_glossary_can_pin_english_spelling(self):
        g = Glossary(terms={"NASA": "NASA"})
        assert g.apply("NASA wins") == "NASA wins"
        assert g.last_applied == ["NASA"]

    def test_from_json_and_file(self, tmp_path):
        path = tmp_path / "glossary.json"
        path.write_text(json.dumps({"Mars": "मंगल ग्रह"}), encoding="utf-8")
        g = Glossary.from_json_file(str(path))
        assert g.apply("Mars") == "मंगल ग्रह"

    def test_invalid_glossary_json_raises(self):
        with pytest.raises(TranslationError, match="invalid glossary JSON"):
            Glossary.from_json("{not json")
        with pytest.raises(TranslationError, match="string->string"):
            Glossary.from_json(json.dumps({"a": 1}))


# --- translate_transcript: timing preservation -------------------------------


class TestTranslateTranscript:
    def test_segments_keep_windows_and_speakers(self):
        out = translate_transcript(
            _transcript(), DictionaryTranslator(source="en", target="hi"), "hi"
        )
        assert out.language == "hi"
        assert out.source == "dictionary:en->hi"
        assert len(out.segments) == 2
        assert (out.segments[0].start, out.segments[0].end) == (0.0, 1.0)
        assert (out.segments[1].start, out.segments[1].end) == (2.0, 3.5)
        assert out.segments[0].speaker == "A"
        assert out.segments[1].speaker == "B"
        assert out.segments[0].text == "नमस्ते दुनिया"
        assert out.segments[1].text == "सुप्रभात टीम"

    def test_words_cover_window_exactly_and_monotonically(self):
        out = translate_transcript(
            _transcript(), DictionaryTranslator(source="en", target="es"), "es"
        )
        for source_seg, target_seg in zip(_transcript().segments, out.segments):
            assert target_seg.start == source_seg.start
            assert target_seg.end == source_seg.end
            starts = [w.start for w in target_seg.words]
            ends = [w.end for w in target_seg.words]
            assert starts == sorted(starts)
            assert all(s < e for s, e in zip(starts, ends))

    def test_longer_words_get_more_time(self):
        seg = Segment(words=_words(["hi"], 0.0, 1.0))
        tr = Transcript(segments=[seg], language="en")
        out = translate_transcript(
            tr,
            DictionaryTranslator(source="en", target="es", dictionary={"hi": "hola amigo largo"}),
            "es",
        )
        durations = [w.end - w.start for w in out.segments[0].words]
        assert durations[2] > durations[0]  # "largo" outlasts "hola"
        assert sum(durations) == pytest.approx(1.0)

    def test_empty_segment_stays_empty(self):
        tr = Transcript(
            segments=[Segment(words=[]), Segment(words=_words(["hello"]))], language="en"
        )
        out = translate_transcript(tr, DictionaryTranslator(source="en", target="hi"), "hi")
        assert out.segments[0].words == []
        assert out.segments[1].text == "नमस्ते"

    def test_unknown_source_language_raises(self):
        tr = Transcript(segments=[Segment(words=_words(["hello"]))])
        with pytest.raises(TranslationError, match="source language unknown"):
            translate_transcript(tr, DictionaryTranslator(source="en", target="hi"), "hi")

    def test_explicit_source_overrides_transcript_language(self):
        tr = Transcript(segments=[Segment(words=_words(["hello"]))], language="fr")
        out = translate_transcript(
            tr, DictionaryTranslator(source="en", target="hi"), "hi", source="en"
        )
        assert out.segments[0].text == "नमस्ते"

    def test_coverage_aggregates_across_segments(self):
        t = DictionaryTranslator(source="en", target="hi")
        out = translate_transcript(_transcript(), t, "hi")
        # Segment 1: hello+world known (2/2). Segment 2: good+morning (phrase)
        # +team known (3/3). Whole transcript: 5/5 = 100%.
        assert t.last_coverage == 1.0
        assert t.last_unknown == []
        assert out.segments[1].text == "सुप्रभात टीम"

    def test_coverage_counts_only_source_words(self):
        t = DictionaryTranslator(source="en", target="hi")
        g = Glossary(terms={"Mars": "मंगल"})
        tr = Transcript(segments=[Segment(words=_words(["Hello", "Mars", "and"]))], language="en")
        translate_transcript(tr, t, "hi", glossary=g)
        # "Mars" was pinned by the glossary (not an unknown source word);
        # only "and" is untranslated: 1 known of 2 counted words.
        assert t.last_coverage == pytest.approx(0.5)
        assert t.last_unknown == ["and"]
        assert g.all_applied == ["Mars"]
        out = translate_transcript(
            _transcript(), DictionaryTranslator(source="en", target="hi"), "hi"
        )
        again = Transcript.from_json(out.to_json())
        assert again.language == "hi"
        assert again.segments[0].text == out.segments[0].text

    def test_max_chars_chunks_long_segments(self):
        out = translate_transcript(
            _transcript(),
            DictionaryTranslator(source="en", target="hi"),
            "hi",
            max_chars=10,
        )
        assert len(out.segments) >= 2
        assert all(len(s.text) <= 10 for s in out.segments)
        assert out.text.replace(" ", "") == "नमस्तेदुनियासुप्रभातटीम"


# --- chunk_segment -----------------------------------------------------------


class TestChunkSegment:
    def test_short_segment_unchanged(self):
        seg = Segment(words=_words(["hello", "world"]))
        assert chunk_segment(seg, max_chars=84) == [seg]

    def test_long_segment_splits_at_word_boundaries(self):
        words = _words([f"w{i:02d}" for i in range(30)])
        seg = Segment(words=words)
        chunks = chunk_segment(seg, max_chars=20)
        assert len(chunks) > 1
        assert all(len(c.text) <= 20 for c in chunks)
        assert "".join(c.text for c in chunks).replace(" ", "") == seg.text.replace(" ", "")
        # Timing windows are preserved across the split.
        assert chunks[0].start == seg.start
        assert chunks[-1].end == seg.end
        assert all(chunks[i].end <= chunks[i + 1].start for i in range(len(chunks) - 1))

    def test_prefers_break_after_sentence_punctuation(self):
        words = _words(["aaa", "bbb.", "ccc", "ddd", "eee", "fff", "ggg", "hhh"])
        seg = Segment(words=words)
        chunks = chunk_segment(seg, max_chars=10)
        assert chunks[0].text == "aaa bbb."

    def test_invalid_max_chars_raises(self):
        seg = Segment(words=_words(["hello"]))
        with pytest.raises(TranslationError, match="max_chars must be positive"):
            chunk_segment(seg, max_chars=0)


# --- ArgosTranslator ---------------------------------------------------------


class TestArgosTranslator:
    def test_missing_cli_raises_clear_error(self, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("no argos-translate")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(TranslationError, match="argos-translate CLI not found"):
            ArgosTranslator().translate("hello", "en", "hi")

    def test_failed_cli_raises_with_stderr(self, monkeypatch):
        import subprocess

        class Result:
            returncode = 1
            stdout = ""
            stderr = "package not installed"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        with pytest.raises(TranslationError, match="package not installed"):
            ArgosTranslator().translate("hello", "en", "hi")


class TestDevanagariTokenization:
    """Regression tests (2026-11-09): the old ``\\w`` tokenizer shattered
    Devanagari words at combining marks, so Hindi input never translated
    at all — ``\\b`` never matched around a word ending in a vowel sign."""

    def test_hindi_word_with_vowel_sign_translates_back(self):
        back = DictionaryTranslator(source="en", target="hi").reversed()
        assert back.translate("नमस्ते दुनिया", "hi", "en") == "hello world"

    def test_hindi_phrase_with_vowel_signs_translates_back(self):
        back = DictionaryTranslator(source="en", target="hi").reversed()
        assert back.translate("सुप्रभात टीम", "hi", "en") == "good morning team"

    def test_reverse_coverage_counts_hindi_words(self):
        back = DictionaryTranslator(source="en", target="hi").reversed()
        back.translate("नमस्ते zqxj", "hi", "en")
        assert back.last_coverage == pytest.approx(0.5)
        assert back.last_unknown == ["zqxj"]
