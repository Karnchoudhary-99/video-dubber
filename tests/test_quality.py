import pytest

from video_dubber.quality import (
    WerResult,
    mos_proxy,
    normalize_words,
    wer_from_transcripts,
    word_error_rate,
)
from video_dubber.transcript import Segment, Transcript, Word


def _words(text: str, start: float = 0.0) -> list[Word]:
    out = []
    t = start
    for w in text.split():
        out.append(Word(text=w, start=t, end=t + 0.4))
        t += 0.5
    return out


def _transcript(text: str, language: str = "en") -> Transcript:
    return Transcript(segments=[Segment(words=_words(text))], language=language)


def test_normalize_words_lowercases_and_strips_punctuation():
    assert normalize_words("Hello, Mars! It's 2026.") == ["hello", "mars", "it's", "2026"]


def test_wer_identical_is_zero():
    result = word_error_rate(["hello", "mars"], ["hello", "mars"])
    assert result == WerResult(0.0, 0, 0, 0, 2, 2)


def test_wer_substitution_deletion_insertion():
    # "hello brave mars" -> "hello red mars rover": 1 sub + 1 ins
    result = word_error_rate(["hello", "brave", "mars"], ["hello", "red", "mars", "rover"])
    assert result.substitutions == 1
    assert result.insertions == 1
    assert result.deletions == 0
    assert result.wer == pytest.approx(2 / 3)


def test_wer_deletion_only():
    result = word_error_rate(["one", "two", "three"], ["one", "three"])
    assert (result.deletions, result.substitutions, result.insertions) == (1, 0, 0)
    assert result.wer == pytest.approx(1 / 3)


def test_wer_can_exceed_one():
    # Hypothesis twice as long as the reference.
    result = word_error_rate(["hi"], ["hi", "hi", "hi"])
    assert result.wer == pytest.approx(2.0)
    assert result.insertions == 2


def test_wer_empty_reference():
    assert word_error_rate([], []).wer == 0.0
    assert word_error_rate([], ["hi"]).wer == 1.0


def test_wer_from_transcripts_normalizes_case_and_punctuation():
    ref = _transcript("Hello, Mars!")
    hyp = _transcript("hello mars")
    assert wer_from_transcripts(ref, hyp).wer == 0.0


def test_wer_from_transcripts_counts_a_missed_word():
    ref = _transcript("welcome to mars base alpha")
    hyp = _transcript("welcome to mars base")
    result = wer_from_transcripts(ref, hyp)
    assert result.deletions == 1
    assert result.wer == pytest.approx(0.2)


def test_mos_proxy_perfect_dub_scores_five():
    assert mos_proxy(0.0, 0.0, 1.0, 0, 4) == 5.0


def test_mos_proxy_penalises_drift_and_extreme_fits():
    # Mean drift 0.15s over the 0.05s grace + one bad segment + 2 extreme fits.
    score = mos_proxy(0.15, 0.30, 1.0, 2, 4)
    assert score == pytest.approx(5.0 - 0.5 - 1.0 - 0.5)


def test_mos_proxy_penalises_loudness_and_wer():
    score = mos_proxy(0.0, 0.0, 1.0, 0, 4, loudness_error_db=5.0, wer=0.25)
    assert score == pytest.approx(5.0 - 1.0 - 0.5)


def test_mos_proxy_clamps_to_one():
    assert mos_proxy(1.0, 2.0, 0.6, 10, 4, loudness_error_db=20.0, wer=1.0) == 1.0


def test_mos_proxy_ignores_missing_loudness_and_wer():
    assert mos_proxy(0.0, 0.0, 1.0, 0, 4, None, None) == 5.0
