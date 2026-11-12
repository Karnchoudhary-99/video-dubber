"""Tests for subtitle alignment (align.align_transcript)."""

import pytest

from video_dubber.align import (
    AlignConfig,
    AlignError,
    AlignmentReport,
    align_transcript,
)
from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def _two_line_track():
    """Silence 0.3s, tone 0.6s, silence 0.6s, tone 0.6s, silence 0.3s.

    The VAD (30 ms frames + 60 ms hangover) reports the tones as
    [0.24, 0.96] and [1.44, 2.16].
    """
    audio = concat(
        make_silence(0.3),
        make_sine(440.0, 0.6, amplitude=0.3),
        make_silence(0.6),
        make_sine(440.0, 0.6, amplitude=0.3),
        make_silence(0.3),
    )
    transcript = Transcript(
        segments=[_seg("first line here", 0.0, 0.9), _seg("second line here", 1.2, 2.1)]
    )
    return audio, transcript


def test_align_moves_cues_to_detected_onsets():
    audio, transcript = _two_line_track()
    aligned, report = align_transcript(audio, transcript)
    # Both cues start at the detected onsets, not the window edges.
    assert aligned.segments[0].start == pytest.approx(0.24, abs=0.01)
    assert aligned.segments[0].end == pytest.approx(0.9, abs=0.01)
    assert aligned.segments[1].start == pytest.approx(1.44, abs=0.01)
    assert aligned.segments[1].end == pytest.approx(2.1, abs=0.01)
    assert report.segments_aligned == 2
    assert report.segments_clamped == 0
    assert report.segments_no_speech == 0
    assert report.mean_abs_shift_ms == pytest.approx(240.0, abs=10.0)
    assert report.max_abs_shift_ms == pytest.approx(240.0, abs=10.0)
    # The SRT a player sees uses the aligned timings.
    assert "00:00:00,240 --> 00:00:00,900" in aligned.to_srt()
    assert "00:00:01,440 --> 00:00:02,100" in aligned.to_srt()
    # Words stay in order and inside their cue.
    assert aligned.segments[0].text == "first line here"
    assert aligned.segments[0].words[0].start >= aligned.segments[0].start


def test_align_no_speech_keeps_planned_timing():
    # Tone, then a full second of silence; the second segment is
    # planned entirely inside the silence.
    audio = concat(make_sine(440.0, 0.5, amplitude=0.3), make_silence(1.0))
    transcript = Transcript(
        segments=[_seg("spoken line here", 0.0, 0.5), _seg("silent pause", 0.6, 1.4)]
    )
    aligned, report = align_transcript(audio, transcript)
    assert report.segments_no_speech == 1
    assert report.rows[1].status == "no-speech"
    assert aligned.segments[1].start == pytest.approx(0.6)
    assert aligned.segments[1].end == pytest.approx(1.4)
    # The spoken cue starts at the window edge already, so it is
    # unchanged — only the silent one is reported.
    assert report.segments_unchanged == 1
    assert report.segments_aligned == 0


def test_align_clamps_large_shift():
    audio, transcript = _two_line_track()
    _aligned, report = align_transcript(audio, transcript, AlignConfig(max_shift_s=0.1))
    assert report.segments_clamped == 2
    assert report.rows[0].aligned_start == pytest.approx(0.1)
    assert report.rows[0].shift_ms == pytest.approx(100.0)


def test_align_clamps_early_end():
    # One long planned window; the voice stops a second early.
    audio = concat(make_sine(440.0, 0.5, amplitude=0.3), make_silence(1.0))
    transcript = Transcript(segments=[_seg("short line here", 0.0, 1.5)])
    _aligned, report = align_transcript(audio, transcript, AlignConfig(max_shift_s=0.2))
    assert report.rows[0].status == "clamped"
    assert report.rows[0].aligned_end == pytest.approx(1.3)
    assert report.rows[0].aligned_start == pytest.approx(0.0)


def test_align_unchanged_when_speech_at_window_edge():
    # The VAD needs some silence in the track to calibrate its noise
    # floor, so the tone is followed by trailing silence.
    audio = concat(make_sine(440.0, 0.5, amplitude=0.3), make_silence(0.5))
    transcript = Transcript(segments=[_seg("line starts now", 0.0, 0.5)])
    _aligned, report = align_transcript(audio, transcript)
    assert report.segments_unchanged == 1
    assert report.rows[0].shift_ms == pytest.approx(0.0)


def test_align_never_overlaps_previous_cue():
    # Overlapping planned windows with continuous speech: the second
    # cue's detected onset sits inside the first cue, so it is pushed
    # to the first cue's end instead of overlapping it.
    audio = concat(make_silence(0.3), make_sine(440.0, 1.2, amplitude=0.3))
    transcript = Transcript(
        segments=[_seg("first line here", 0.0, 1.0), _seg("second line here", 0.9, 1.5)]
    )
    aligned, report = align_transcript(audio, transcript)
    assert aligned.segments[1].start >= aligned.segments[0].end
    assert report.rows[1].status == "clamped"


def test_align_skips_wordless_segments():
    audio, transcript = _two_line_track()
    transcript.segments.insert(1, Segment(words=[], speaker="narrator"))
    aligned, report = align_transcript(audio, transcript)
    assert len(report.rows) == 2  # only the timed segments
    assert len(aligned.segments) == 3  # the wordless one is kept
    assert aligned.segments[1].words == []


def test_align_rejects_bad_params():
    audio, transcript = _two_line_track()
    with pytest.raises(AlignError, match="max_shift_s"):
        align_transcript(audio, transcript, AlignConfig(max_shift_s=-0.5))
    with pytest.raises(AlignError, match="min_cue_s"):
        align_transcript(audio, transcript, AlignConfig(min_cue_s=-0.5))
    with pytest.raises(AlignError, match="no timed segments"):
        align_transcript(audio, Transcript(segments=[Segment(words=[])]))


def test_alignment_report_json_round_trip():
    audio, transcript = _two_line_track()
    _aligned, report = align_transcript(audio, transcript)
    restored = AlignmentReport.from_json(report.to_json())
    assert restored.segments_aligned == report.segments_aligned
    assert restored.mean_abs_shift_ms == pytest.approx(report.mean_abs_shift_ms)
    assert restored.rows[0].to_dict() == report.rows[0].to_dict()
    assert restored.max_shift_s == report.max_shift_s
    with pytest.raises(ValueError, match="not valid JSON"):
        AlignmentReport.from_json("{not json")
    with pytest.raises(ValueError, match="not a valid alignment report"):
        AlignmentReport.from_json('{"segments_aligned": 1}')


def test_alignment_report_markdown():
    audio, transcript = _two_line_track()
    transcript.segments.insert(1, _seg("silent pause", 1.0, 1.35))
    _aligned, report = align_transcript(audio, transcript)
    text = report.to_markdown()
    assert "Segments aligned: 2" in text
    assert "no speech detected" in text
    assert "Mean |start shift|" in text
