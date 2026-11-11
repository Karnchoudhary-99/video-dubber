"""Tests for the dub cost/time estimator (estimate.py)."""

import json

import pytest

from video_dubber.cache import CachedVoiceCloner, SynthesisCache, cache_key
from video_dubber.estimate import (
    BACKEND_RTF,
    DubEstimate,
    EstimateError,
    estimate_dub,
    predict_speech_duration,
)
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner, SpeakerProfile


def _profile(name: str = "asha", wpm: float = 150.0) -> SpeakerProfile:
    return SpeakerProfile(
        name=name,
        embedding=[0.25] * 16,
        base_pitch_hz=180.0,
        speaking_rate_wpm=wpm,
    )


def _sample() -> Transcript:
    # Two segments: a roomy one and a tight one (short window, long text).
    return Transcript(
        segments=[
            Segment(
                words=[
                    Word("hello", 0.0, 0.4),
                    Word("world", 0.5, 0.9),
                ],
                speaker="A",
            ),
            Segment(
                words=[
                    Word("welcome", 2.0, 2.3),
                    Word("to", 2.4, 2.5),
                    Word("the", 2.6, 2.8),
                    Word("show", 2.9, 3.1),
                ],
                speaker="A",
            ),
        ],
        language="hi",
        source="clip.wav",
    )


def test_predict_matches_scripted_synthesis_exactly():
    """The scripted model's prediction must equal the real rendered length."""
    profile = _profile(wpm=150.0)
    cloner = ScriptedVoiceCloner()
    for text in ["hello world", "a much longer sentence with several words in it", "x"]:
        audio = cloner.synthesize(text, profile)
        assert audio.duration == pytest.approx(predict_speech_duration(text, 150.0), abs=1e-3)


def test_predict_empty_text_is_silence():
    assert predict_speech_duration("", 150.0) == 0.0
    assert predict_speech_duration("   ", 150.0) == 0.0


def test_predict_rejects_non_positive_wpm():
    with pytest.raises(EstimateError, match="positive"):
        predict_speech_duration("hello", 0)


def test_predict_slower_rate_is_longer():
    # Long words only: short words hit the 0.12 s floor and break the ratio.
    # The 0.05 s word gap is fixed, so only the word-time part doubles.
    text = "elephant giraffe rhinoceros hippopotamus"
    fast = predict_speech_duration(text, 200.0)
    slow = predict_speech_duration(text, 100.0)
    assert slow > fast
    gaps = len(text.split()) * 0.05
    assert slow - gaps == pytest.approx(2 * (fast - gaps))


def test_estimate_basics():
    est = estimate_dub(_sample(), backend="scripted", target="hi")
    assert est.segments == 2
    assert est.total_words == 6
    assert est.source_duration_s == pytest.approx(3.1)
    assert est.duration_exact is True
    assert est.real_time_factor == BACKEND_RTF["scripted"]
    # Segment 1: "hello world" at 150 wpm -> 2 words * (5/5*0.4 + 0.05) = 0.9s
    row1 = est.rows[0]
    assert row1.words == 2
    assert row1.predicted_s == pytest.approx(0.9)
    assert row1.fit_factor == pytest.approx(0.9 / 0.9)
    assert row1.extreme is False
    assert row1.cached is False
    # Segment 2: window 1.1s, predicted well over it -> extreme stretch.
    row2 = est.rows[1]
    assert row2.fit_factor > 1.25
    assert row2.extreme is True
    assert est.extreme_fits == 1
    assert est.max_fit_factor == pytest.approx(row2.fit_factor)
    assert est.est_cost is None  # unpriced


def test_estimate_wall_time_uses_rtf():
    est = estimate_dub(_sample(), backend="scripted")
    # All uncached: synth wall = predicted / 20, plus 0.05s per segment.
    assert est.est_synth_wall_s == pytest.approx(est.predicted_speech_s / 20.0)
    assert est.est_total_wall_s == pytest.approx(est.est_synth_wall_s + 0.10)


def test_estimate_unknown_backend_needs_rtf():
    with pytest.raises(EstimateError, match="unknown backend"):
        estimate_dub(_sample(), backend="mystery-tts")
    est = estimate_dub(_sample(), backend="mystery-tts", real_time_factor=3.0)
    assert est.real_time_factor == 3.0
    assert est.duration_exact is False


def test_estimate_rejects_bad_inputs(tmp_path):
    with pytest.raises(EstimateError, match="empty transcript"):
        estimate_dub(Transcript(segments=[]))
    with pytest.raises(EstimateError, match="positive"):
        estimate_dub(_sample(), words_per_minute=0)
    with pytest.raises(EstimateError, match="positive"):
        estimate_dub(_sample(), real_time_factor=0)
    with pytest.raises(EstimateError, match="at most one"):
        estimate_dub(_sample(), price_per_char=0.01, price_per_minute=1.0)
    with pytest.raises(EstimateError, match=">= 0"):
        estimate_dub(_sample(), price_per_char=-0.01)
    with pytest.raises(EstimateError, match="needs a speaker profile"):
        estimate_dub(_sample(), cache=SynthesisCache(tmp_path / "synth"))


def test_estimate_price_per_char():
    est = estimate_dub(_sample(), backend="scripted", price_per_char=0.001)
    assert est.est_cost == pytest.approx(est.total_chars * 0.001)


def test_estimate_price_per_minute():
    est = estimate_dub(_sample(), backend="scripted", price_per_minute=0.60)
    assert est.est_cost == pytest.approx(est.predicted_speech_s / 60.0 * 0.60)


def test_estimate_cache_hits_excluded_from_cost_and_wall_time(tmp_path):
    cache = SynthesisCache(tmp_path / "synth")
    profile = _profile()
    cloner = CachedVoiceCloner(ScriptedVoiceCloner(), cache)
    transcript = _sample()
    # Synthesise only the first segment, so it lands in the cache.
    cloner.synthesize(transcript.segments[0].text, profile)
    before = cache.stats()

    est = estimate_dub(
        transcript,
        backend="scripted",
        cache=cache,
        profile=profile,
        price_per_char=0.01,
    )
    assert est.rows[0].cached is True
    assert est.rows[1].cached is False
    assert est.cached_segments == 1
    assert est.chars_to_synthesize == len(transcript.segments[1].text)
    assert est.est_cost == pytest.approx(len(transcript.segments[1].text) * 0.01)
    # Only the uncached segment's speech counts towards wall time.
    assert est.est_synth_wall_s == pytest.approx(est.rows[1].predicted_s / BACKEND_RTF["scripted"])
    # Probing must not disturb the cache: no new hits/misses recorded.
    assert cache.stats().hits == before.hits
    assert cache.stats().misses == before.misses


def test_cache_contains_is_read_only(tmp_path):
    cache = SynthesisCache(tmp_path / "synth")
    profile = _profile()
    key = cache_key("scripted", "hello world", profile)
    assert cache.contains(key) is False
    cache.put(key, ScriptedVoiceCloner().synthesize("hello world", profile))
    assert cache.contains(key) is True
    stats = cache.stats()
    assert cache.contains(key) is True
    assert cache.stats().hits == stats.hits  # no counter moved
    assert cache.contains("deadbeef") is False


def test_estimate_json_round_trip():
    est = estimate_dub(_sample(), backend="scripted", target="hi", price_per_minute=0.5)
    restored = DubEstimate.from_json(est.to_json())
    # Rows round-trip exactly once serialised (to_dict rounds to 3 decimals);
    # aggregates are re-derived from the rounded rows, so compare with tolerance.
    assert [row.to_dict() for row in restored.rows] == [row.to_dict() for row in est.rows]
    assert restored.backend == est.backend
    assert restored.target == est.target
    assert restored.mean_fit_factor == pytest.approx(est.mean_fit_factor, abs=0.001)
    assert restored.max_fit_factor == pytest.approx(est.max_fit_factor, abs=0.001)
    assert restored.predicted_speech_s == pytest.approx(est.predicted_speech_s, abs=0.001)
    assert restored.est_cost == pytest.approx(est.est_cost)


def test_estimate_json_rejects_bad_input():
    with pytest.raises(EstimateError, match="invalid dub estimate JSON"):
        DubEstimate.from_json("{not json")


def test_estimate_markdown_names_the_numbers():
    est = estimate_dub(_sample(), backend="scripted", target="hi", price_per_char=0.002)
    md = est.to_markdown()
    assert "# Dub estimate" in md
    assert "scripted" in md
    assert "hi" in md
    assert "150 wpm" in md
    assert "Estimated cost:" in md
    assert "⚠" in md  # the extreme-fit row is flagged
    assert "| 2 | 2.00–3.10s" in md


def test_estimate_markdown_without_price_or_cache():
    est = estimate_dub(_sample(), backend="coqui")
    md = est.to_markdown()
    assert "approximate durations" in md
    assert "Estimated cost:" not in md
    assert "Cache:" not in md


def test_estimate_to_dict_is_json_serialisable():
    est = estimate_dub(_sample(), backend="scripted")
    json.dumps(est.to_dict())  # must not raise


# --- CLI ---------------------------------------------------------------------


def test_cli_estimate_markdown(tmp_path, capsys):
    from video_dubber.cli import main

    transcript_path = tmp_path / "translated.json"
    transcript_path.write_text(_sample().to_json(), encoding="utf-8")
    assert main(["estimate", str(transcript_path), "--target", "hi"]) == 0
    out = capsys.readouterr().out
    assert "# Dub estimate" in out
    assert "hi" in out
    assert "⚠" in out  # the tight second segment is flagged


def test_cli_estimate_json_and_out_file(tmp_path, capsys):
    from video_dubber.cli import main

    transcript_path = tmp_path / "translated.json"
    transcript_path.write_text(_sample().to_json(), encoding="utf-8")
    out_path = tmp_path / "estimate.json"
    assert (
        main(
            [
                "estimate",
                str(transcript_path),
                "--format",
                "json",
                "--price-per-char",
                "0.002",
                "--out",
                str(out_path),
            ]
        )
        == 0
    )
    saved = DubEstimate.from_json(out_path.read_text(encoding="utf-8"))
    assert saved.est_cost == pytest.approx(saved.total_chars * 0.002)
    assert "wrote" in capsys.readouterr().out


def test_cli_estimate_with_profile_and_cache(tmp_path, capsys):
    from video_dubber.cache import CachedVoiceCloner
    from video_dubber.cli import main

    transcript = _sample()
    transcript_path = tmp_path / "translated.json"
    transcript_path.write_text(transcript.to_json(), encoding="utf-8")
    profile = _profile()
    profile_path = tmp_path / "asha.json"
    profile_path.write_text(profile.to_json(), encoding="utf-8")
    cache_dir = tmp_path / "synth"
    CachedVoiceCloner(ScriptedVoiceCloner(), SynthesisCache(cache_dir)).synthesize(
        transcript.segments[0].text, profile
    )
    assert (
        main(
            [
                "estimate",
                str(transcript_path),
                "--profile",
                str(profile_path),
                "--cache-dir",
                str(cache_dir),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Cache: 1/2 segments already cached" in out


def test_cli_estimate_errors_are_exit_1(tmp_path, capsys):
    from video_dubber.cli import main

    transcript_path = tmp_path / "translated.json"
    transcript_path.write_text(_sample().to_json(), encoding="utf-8")
    # Unknown backend without --rtf.
    assert main(["estimate", str(transcript_path), "--backend", "mystery"]) == 1
    assert "unknown backend" in capsys.readouterr().err
    # Cache dir without a profile.
    assert main(["estimate", str(transcript_path), "--cache-dir", str(tmp_path)]) == 1
    assert "speaker profile" in capsys.readouterr().err
