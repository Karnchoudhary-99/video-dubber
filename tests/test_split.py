"""Tests for segment splitting at silence gaps."""

import pytest

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.split import (
    SplitConfig,
    SplitError,
    SplitReport,
    split_transcript,
)
from video_dubber.transcript import Segment, Transcript, Word


def _words(texts, start=0.0, step=1.0, word_len=0.6):
    """Words back-to-back-ish: each ``word_len`` long, ``step`` apart."""
    words = []
    cursor = start
    for text in texts:
        words.append(Word(text, cursor, cursor + word_len))
        cursor += step
    return words


def _long_segment(n_words=12, step=0.7, gap_at=None, gap=0.8, speaker="A"):
    """One 12-word segment (~9.1s); an optional extra pause after word ``gap_at``.

    The regular inter-word gap is 0.1s (below the default min_gap), so
    only the deliberate pause is a silence candidate.
    """
    words = _words([f"w{i}" for i in range(n_words)], step=step)
    if gap_at is not None:
        words = [
            Word(w.text, w.start + (gap if i > gap_at else 0.0),
                 w.end + (gap if i > gap_at else 0.0))
            for i, w in enumerate(words)
        ]
    return Segment(words=words, speaker=speaker)


def test_config_validation():
    with pytest.raises(SplitError):
        SplitConfig(max_duration=0)
    with pytest.raises(SplitError):
        SplitConfig(min_gap=-0.1)
    with pytest.raises(SplitError):
        SplitConfig(max_duration=2.0, min_segment=2.0)


def test_short_segment_passes_through():
    transcript = Transcript(
        segments=[Segment(words=_words(["hello", "world"]), speaker="A")],
        language="en",
    )
    out, report = split_transcript(transcript, config=SplitConfig(max_duration=8.0))
    assert len(out.segments) == 1
    assert out.segments[0].speaker == "A"
    assert report.total_splits == 0
    assert report.segments_before == report.segments_after == 1


def test_splits_at_silence_gap():
    # Pause of 0.8s after word 5; segment is ~12.4s, max 8s.
    segment = _long_segment(gap_at=5, gap=0.8)
    transcript = Transcript(segments=[segment], language="en")
    out, report = split_transcript(transcript, config=SplitConfig(max_duration=8.0))
    assert len(out.segments) == 2
    assert report.total_splits == 1
    assert report.splits[0].reason == "silence"
    assert report.splits[0].segment_index == 0
    # Words and timings survive the cut, in order, speaker kept.
    assert [w.text for s in out.segments for w in s.words] == [
        f"w{i}" for i in range(12)
    ]
    assert all(s.speaker == "A" for s in out.segments)
    assert all(s.duration <= 8.0 for s in out.segments)
    assert report.max_duration_after <= 8.0
    assert report.max_duration_before == pytest.approx(segment.duration)
    # The cut is at the end of word 5 (the last word before the pause).
    assert report.splits[0].at == pytest.approx(segment.words[5].end)


def test_input_transcript_is_untouched():
    segment = _long_segment(gap_at=5, gap=0.8)
    transcript = Transcript(segments=[segment], language="en")
    split_transcript(transcript, config=SplitConfig(max_duration=8.0))
    assert len(transcript.segments) == 1
    assert len(transcript.segments[0].words) == 12


def test_continuous_speech_forces_a_cut():
    # No pauses at all: words 1s apart, 0.6s long — gaps of 0.4s only
    # between word end and next start... make them truly continuous:
    # word_len == step, so gap is 0.0.
    words = _words([f"w{i}" for i in range(12)], step=0.6, word_len=0.6)
    transcript = Transcript(segments=[Segment(words=words)], language="en")
    out, report = split_transcript(
        transcript, config=SplitConfig(max_duration=4.0, min_gap=0.4)
    )
    assert len(out.segments) == 2
    assert report.total_splits == 1
    assert report.splits[0].reason == "forced"
    assert report.forced_splits == 1
    assert [w.text for s in out.segments for w in s.words] == [
        f"w{i}" for i in range(12)
    ]


def test_falls_back_to_longest_gap():
    # Gaps of 0.2s only — below min_gap 0.4, so no silence candidate;
    # one gap is 0.35s, the longest, and the split must use it.
    words = _words([f"w{i}" for i in range(12)], step=0.8, word_len=0.6)
    shifted = []
    for i, w in enumerate(words):
        shift = 0.15 if i > 6 else 0.0
        shifted.append(Word(w.text, w.start + shift, w.end + shift))
    transcript = Transcript(segments=[Segment(words=shifted)], language="en")
    out, report = split_transcript(
        transcript, config=SplitConfig(max_duration=6.0, min_gap=0.4)
    )
    assert report.total_splits == 1
    assert report.splits[0].reason == "longest-gap"
    assert report.splits[0].at == pytest.approx(shifted[6].end)
    assert [w.text for s in out.segments for w in s.words] == [
        f"w{i}" for i in range(12)
    ]


def test_min_segment_skips_too_early_pause():
    # A pause after word 0 would leave a 0.6s piece (< min_segment 1.0),
    # so the split must skip it and use the later pause after word 5.
    # word0 0.0-0.6, 0.8s pause, then words 1.4s apart with a 0.8s pause after word 5.
    rebuilt = [Word("w0", 0.0, 0.6)]
    cursor = 1.4
    for i in range(1, 12):
        rebuilt.append(Word(f"w{i}", cursor, cursor + 0.6))
        cursor += 1.0
        if i == 5:
            cursor += 0.8
    transcript = Transcript(segments=[Segment(words=rebuilt)], language="en")
    out, report = split_transcript(
        transcript, config=SplitConfig(max_duration=8.0, min_segment=1.0)
    )
    assert report.total_splits == 1
    assert report.splits[0].reason == "silence"
    assert report.splits[0].at == pytest.approx(rebuilt[5].end)
    assert out.segments[0].duration >= 1.0


def test_audio_confirms_boundary_hidden_by_coarse_timings():
    # Words timed back-to-back (no timing gap), but the audio really
    # pauses for 1s after word 5: without audio no split is possible at
    # a silence point; with audio the VAD finds it.
    words = _words([f"w{i}" for i in range(12)], step=0.6, word_len=0.6)
    transcript = Transcript(segments=[Segment(words=words)], language="en")
    audio = concat(
        make_sine(440.0, 3.6),  # words 0-5
        make_silence(1.0),  # the real pause
        make_sine(440.0, 3.6),  # words 6-11
    )
    _, no_audio = split_transcript(transcript, config=SplitConfig(max_duration=4.0))
    assert no_audio.splits[0].reason == "forced"
    out, report = split_transcript(
        transcript, audio=audio, config=SplitConfig(max_duration=4.0)
    )
    assert report.total_splits == 1
    assert report.splits[0].reason == "silence"
    assert report.splits[0].at == pytest.approx(words[5].end)


def test_single_long_word_is_never_cut():
    words = [Word("supercalifragilistic", 0.0, 10.0)]
    transcript = Transcript(segments=[Segment(words=words)], language="en")
    out, report = split_transcript(transcript, config=SplitConfig(max_duration=8.0))
    assert len(out.segments) == 1
    assert report.total_splits == 0


def test_empty_transcript():
    out, report = split_transcript(Transcript(segments=[]))
    assert out.segments == []
    assert report.segments_before == 0
    assert report.max_duration_before == 0.0


def test_multiple_segments_split_independently():
    first = _long_segment(gap_at=5, gap=0.8)
    second = Segment(words=_words(["short", "line"]))
    transcript = Transcript(segments=[first, second], language="en")
    out, report = split_transcript(transcript, config=SplitConfig(max_duration=8.0))
    assert len(out.segments) == 3
    assert report.segments_before == 2
    assert report.segments_after == 3
    assert report.splits[0].segment_index == 0
    assert out.segments[2].text == "short line"


def test_report_json_round_trip_and_markdown():
    segment = _long_segment(gap_at=5, gap=0.8)
    _, report = split_transcript(
        Transcript(segments=[segment]), config=SplitConfig(max_duration=8.0)
    )
    parsed = SplitReport.from_json(report.to_json())
    assert parsed.segments_after == report.segments_after
    assert parsed.splits[0].reason == "silence"
    markdown = report.to_markdown()
    assert "1 -> 2" in markdown
    assert "silence" in markdown
    with pytest.raises(SplitError):
        SplitReport.from_json("{not json")


def test_forced_split_report_names_the_problem():
    words = _words([f"w{i}" for i in range(12)], step=0.6, word_len=0.6)
    _, report = split_transcript(
        Transcript(segments=[Segment(words=words)]),
        config=SplitConfig(max_duration=4.0),
    )
    assert "Forced cuts" in report.to_markdown()
