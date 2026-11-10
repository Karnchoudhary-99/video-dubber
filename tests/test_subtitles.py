"""Tests for dub-synced subtitles (subtitles.retime_transcript)."""

import pytest

from video_dubber.subtitles import SubtitleError, retime_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("world", 0.4, 1.0),
                ],
                speaker="asha",
            ),
            Segment(words=[]),  # wordless: no placed entry, copied as-is
            Segment(
                words=[
                    Word("What", 2.0, 2.4),
                    Word("a", 2.4, 2.5),
                    Word("goal", 2.5, 3.0),
                ],
                speaker="ravi",
            ),
        ],
        language="hi",
        source="demo",
    )


def test_words_rescale_proportionally_into_placed_span():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    first, _, third = retimed.segments
    # Segment 1 was 1.0s, placed 0.5s: every word halves.
    assert (first.start, first.end) == pytest.approx((0.0, 0.5))
    assert first.words[0].start == pytest.approx(0.0)
    assert first.words[0].end == pytest.approx(0.2)
    assert first.words[1].start == pytest.approx(0.2)
    assert first.words[1].end == pytest.approx(0.5)
    # Segment 3 was 1.0s at 2.0-3.0, placed 2.0-3.0: unchanged.
    assert (third.start, third.end) == pytest.approx((2.0, 3.0))
    assert third.words[1].text == "a"
    assert third.words[1].start == pytest.approx(2.4)


def test_placed_start_moves_the_cue():
    retimed = retime_transcript(_transcript(), [(0.5, 0.5), (2.5, 1.0)])
    first = retimed.segments[0]
    assert (first.start, first.end) == pytest.approx((0.5, 1.0))


def test_srt_uses_retimed_timings():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    srt = retimed.to_srt()
    assert "00:00:00,000 --> 00:00:00,500" in srt
    assert "00:00:02,000 --> 00:00:03,000" in srt
    # Only the two timed segments become cues.
    assert srt.count("-->") == 2


def test_wordless_segments_need_no_placed_entry_and_survive():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    assert retimed.segments[1].words == []
    assert retimed.segments[1].speaker is None


def test_speaker_language_and_source_preserved():
    retimed = retime_transcript(_transcript(), [(0.0, 0.5), (2.0, 1.0)])
    assert retimed.segments[0].speaker == "asha"
    assert retimed.segments[2].speaker == "ravi"
    assert retimed.language == "hi"
    assert retimed.source == "demo"


def test_original_transcript_not_mutated():
    original = _transcript()
    retime_transcript(original, [(0.0, 0.5), (2.0, 1.0)])
    assert (original.segments[0].start, original.segments[0].end) == pytest.approx((0.0, 1.0))


def test_short_cue_extends_to_minimum():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.2)])
    assert (retimed.segments[0].start, retimed.segments[0].end) == pytest.approx((0.0, 0.5))


def test_short_cue_never_overlaps_next_cue():
    transcript = Transcript(
        segments=[
            Segment(words=[Word("Hi", 0.0, 0.4)]),
            Segment(words=[Word("Yo", 0.6, 1.0)]),
        ],
    )
    # First cue would extend to 0.5s, but the next cue starts at 0.45s.
    retimed = retime_transcript(transcript, [(0.0, 0.2), (0.45, 0.8)])
    assert retimed.segments[0].end == pytest.approx(0.45)
    assert retimed.segments[1].start == pytest.approx(0.45)


def test_zero_duration_segment_collapses_then_gets_min_cue():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 1.0, 1.0), Word("there", 1.0, 1.0)])],
    )
    retimed = retime_transcript(transcript, [(1.0, 0.3)])
    first = retimed.segments[0]
    assert first.words[0].start == pytest.approx(1.0)
    assert (first.start, first.end) == pytest.approx((1.0, 1.5))


def test_min_cue_can_be_disabled():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.2)], min_cue_s=0.0)
    assert retimed.segments[0].end == pytest.approx(0.2)


def test_placed_count_mismatch_raises():
    with pytest.raises(SubtitleError, match="2 spans for 1 timed"):
        retime_transcript(
            Transcript(segments=[Segment(words=[Word("Hi", 0.0, 0.4)])]),
            [(0.0, 0.5), (1.0, 0.5)],
        )


def test_bad_spans_raise():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    with pytest.raises(SubtitleError, match="starts before 0"):
        retime_transcript(transcript, [(-0.1, 0.5)])
    with pytest.raises(SubtitleError, match="non-positive duration"):
        retime_transcript(transcript, [(0.0, 0.0)])


def test_negative_min_cue_raises():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hi", 0.0, 0.4)])],
    )
    with pytest.raises(SubtitleError, match="min_cue_s"):
        retime_transcript(transcript, [(0.0, 0.5)], min_cue_s=-0.1)


def test_word_confidences_survive():
    transcript = Transcript(
        segments=[Segment(words=[Word("Hello", 0.0, 0.4, 0.7), Word("world", 0.4, 1.0, 0.9)])],
    )
    retimed = retime_transcript(transcript, [(0.0, 0.5)])
    assert [w.confidence for w in retimed.segments[0].words] == pytest.approx([0.7, 0.9])
