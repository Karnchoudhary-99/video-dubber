"""Tests for segment merging (merge.merge_transcript)."""

import pytest

from video_dubber.merge import MergeConfig, MergeError, MergeReport, merge_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float, speaker: str | None = None) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)],
        speaker=speaker,
    )


def test_config_validation():
    with pytest.raises(MergeError, match="max_gap"):
        MergeConfig(max_gap=-0.1)
    with pytest.raises(MergeError, match="max_duration"):
        MergeConfig(max_duration=0)


def test_short_gap_merges_neighbours():
    transcript = Transcript(
        segments=[
            _seg("hello", 0.0, 0.4),
            _seg("world", 0.6, 1.0),  # 0.2s gap: merges
            _seg("later", 3.0, 3.4),  # 2.0s gap: stays separate
        ]
    )
    merged, report = merge_transcript(transcript)
    assert len(merged.segments) == 2
    assert merged.segments[0].text == "hello world"
    assert merged.segments[1].text == "later"
    assert report.segments_before == 3
    assert report.segments_after == 2
    assert report.total_merges == 1
    assert report.segments_merged == 1
    assert report.merges[0].segment_indices == [0, 1]
    assert report.merges[0].reason == "short-gap"


def test_merge_keeps_word_timings():
    transcript = Transcript(
        segments=[_seg("hello", 0.0, 0.4), _seg("world", 0.6, 1.0)],
    )
    merged, _report = merge_transcript(transcript)
    words = merged.segments[0].words
    assert [(w.text, w.start, w.end) for w in words] == [
        ("hello", 0.0, 0.4),
        ("world", 0.6, 1.0),
    ]


def test_speaker_change_blocks_merge():
    transcript = Transcript(
        segments=[
            _seg("hello", 0.0, 0.4, speaker="Asha"),
            _seg("world", 0.6, 1.0, speaker="Ravi"),
        ]
    )
    merged, report = merge_transcript(transcript)
    assert len(merged.segments) == 2
    assert report.total_merges == 0


def test_unlabelled_segments_merge_together():
    transcript = Transcript(
        segments=[_seg("hello", 0.0, 0.4), _seg("world", 0.6, 1.0)],
    )
    merged, report = merge_transcript(transcript)
    assert len(merged.segments) == 1
    assert report.total_merges == 1


def test_max_duration_blocks_merge():
    transcript = Transcript(
        segments=[_seg("hello world", 0.0, 3.0), _seg("again", 3.2, 6.0)],
    )
    merged, report = merge_transcript(transcript, MergeConfig(max_duration=5.0))
    # Combined would be 6.0s > 5.0s: no merge.
    assert len(merged.segments) == 2
    assert report.total_merges == 0


def test_chain_merges_across_several_segments():
    transcript = Transcript(
        segments=[
            _seg("one", 0.0, 0.4),
            _seg("two", 0.6, 1.0),
            _seg("three", 1.2, 1.6),
        ]
    )
    merged, report = merge_transcript(transcript)
    assert len(merged.segments) == 1
    assert merged.segments[0].text == "one two three"
    assert report.merges[0].segment_indices == [0, 1, 2]
    assert report.segments_merged == 2


def test_gap_exactly_at_max_gap_merges():
    transcript = Transcript(
        segments=[_seg("hello", 0.0, 0.4), _seg("world", 0.8, 1.2)],
    )
    merged, report = merge_transcript(transcript, MergeConfig(max_gap=0.4))
    assert len(merged.segments) == 1
    assert report.total_merges == 1


def test_wordless_segment_breaks_the_group():
    transcript = Transcript(
        segments=[
            _seg("hello", 0.0, 0.4),
            Segment(words=[]),
            _seg("world", 0.6, 1.0),
        ]
    )
    merged, report = merge_transcript(transcript)
    assert len(merged.segments) == 3
    assert merged.segments[1].words == []
    assert report.total_merges == 0


def test_report_round_trip_and_markdown():
    transcript = Transcript(
        segments=[_seg("hello", 0.0, 0.4), _seg("world", 0.6, 1.0)],
    )
    _merged, report = merge_transcript(transcript)
    restored = MergeReport.from_json(report.to_json())
    assert restored.segments_before == 2
    assert restored.segments_after == 1
    assert restored.merges[0].segment_indices == [0, 1]
    markdown = report.to_markdown()
    assert "# Segment merge" in markdown
    assert "Segments: 2 -> 1 (1 merge, 1 segment joined)" in markdown
    assert "| 1+2 | short-gap |" in markdown
    with pytest.raises(MergeError, match="invalid merge report"):
        MergeReport.from_json("{nope")


def test_empty_transcript_merges_to_empty():
    merged, report = merge_transcript(Transcript(segments=[]))
    assert merged.segments == []
    assert report.segments_before == 0
    assert report.segments_after == 0
    assert report.max_duration_after == 0.0


def test_language_and_source_survive():
    transcript = Transcript(
        segments=[_seg("hello", 0.0, 0.4), _seg("world", 0.6, 1.0)],
        language="hi",
        source="demo",
    )
    merged, _report = merge_transcript(transcript)
    assert merged.language == "hi"
    assert merged.source == "demo"
