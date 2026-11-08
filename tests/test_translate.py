"""Tests for the translation pipeline: dictionary, glossary, timing, chunking."""

import json

import pytest

from video_dubber.transcript import Segment, Transcript, Word, from_words
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
        t = DictionaryTranslator(
            source="en", target="xx", dictionary={"hello": "xyzzy"}
        )
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
