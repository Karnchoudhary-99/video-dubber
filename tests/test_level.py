"""Tests for per-segment loudness levelling (level.level_segments)."""

import math

import pytest

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.level import (
    LevelError,
    LevelReport,
    SegmentLevel,
    level_segments,
    measure_segment_levels,
)
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def _two_tone_track() -> tuple[AudioData, Transcript]:
    """Quiet 0.5s line, 0.5s gap, loud 0.5s line (1.5s total)."""
    quiet = make_sine(440.0, 0.5, amplitude=0.05)
    gap = make_silence(0.5)
    loud = make_sine(440.0, 0.5, amplitude=0.5)
    audio = concat(quiet, gap, loud)
    transcript = Transcript(
        segments=[_seg("quiet line here", 0.0, 0.5), _seg("loud line here", 1.0, 1.5)]
    )
    return audio, transcript


def test_measure_segment_levels():
    audio, transcript = _two_tone_track()
    levels = measure_segment_levels(audio, transcript)
    assert len(levels) == 2
    # Sine RMS is amplitude / sqrt(2): 0.05 -> -29.0 dBFS, 0.5 -> -9.0 dBFS.
    assert levels[0].rms_dbfs == pytest.approx(-29.0, abs=0.2)
    assert levels[1].rms_dbfs == pytest.approx(-9.0, abs=0.2)
    assert levels[0].peak_dbfs == pytest.approx(-26.0, abs=0.2)
    assert levels[0].gain_db == 0.0
    assert not levels[0].silent and not levels[0].at_cap


def test_level_evens_segments_to_target():
    audio, transcript = _two_tone_track()
    # The quiet line needs +13 dB, past the default 12 dB cap, so lift
    # the cap for this test — the cap itself is covered below.
    levelled, report = level_segments(audio, transcript, max_gain_db=20.0)
    after = measure_segment_levels(levelled, transcript)
    # Both segments land within 0.5 dB of the -16 dBFS target.
    assert after[0].rms_dbfs == pytest.approx(-16.0, abs=0.5)
    assert after[1].rms_dbfs == pytest.approx(-16.0, abs=0.5)
    assert report.max_deviation_before_db == pytest.approx(13.0, abs=0.5)
    assert report.max_deviation_after_db < 0.5
    assert report.segments_levelled == 2
    assert report.segments_at_cap == 0
    assert report.segments_silent == 0
    assert levelled.duration == audio.duration  # timing never moves
    assert levelled.sample_rate == audio.sample_rate


def test_level_applies_reported_gains():
    audio, transcript = _two_tone_track()
    _levelled, report = level_segments(audio, transcript, target_dbfs=-16.0, max_gain_db=20.0)
    # Quiet line needs +13 dB, loud line needs -7 dB.
    assert report.levels[0].gain_db == pytest.approx(13.0, abs=0.5)
    assert report.levels[1].gain_db == pytest.approx(-7.0, abs=0.5)


def test_gain_cap_is_reported_not_hidden():
    audio, transcript = _two_tone_track()
    # A 2 dB cap cannot lift the quiet line 13 dB to the target, and
    # it also clips the loud line's -7 dB cut — both segments hit it.
    _levelled, report = level_segments(audio, transcript, max_gain_db=2.0)
    assert report.levels[0].gain_db == pytest.approx(2.0, abs=0.1)
    assert report.levels[0].at_cap
    assert report.segments_at_cap == 2
    assert report.max_deviation_after_db > 10.0  # still off-target, honestly


def test_ceiling_caps_gain_for_spiky_segments():
    # Mostly quiet line with one full-scale spike: the RMS gain would
    # push the spike past the ceiling, so the ceiling wins.
    samples = [0.01 * math.sin(2 * math.pi * 440 * i / 16_000) for i in range(8000)]
    samples[4000] = 1.0
    audio = AudioData(samples, 16_000)
    transcript = Transcript(segments=[_seg("spiky line here", 0.0, 0.5)])
    levelled, report = level_segments(audio, transcript)
    assert levelled.peak <= 10.0 ** (-1.0 / 20.0) + 1e-9
    assert report.levels[0].at_cap


def test_silent_segment_is_reported_not_gained():
    audio, transcript = _two_tone_track()
    # Add a third segment over the silent gap (0.5-1.0s).
    transcript.segments.append(_seg("nothing said here", 0.5, 1.0))
    levelled, report = level_segments(audio, transcript)
    assert report.levels[2].silent
    assert report.levels[2].gain_db == 0.0
    assert report.segments_silent == 1
    assert report.segments_levelled == 2
    # The gap stays silent: levelling never invents sound.
    gap = levelled.samples[8000:16000]
    assert max(abs(s) for s in gap) == 0.0


def test_wordless_segments_are_skipped():
    audio, transcript = _two_tone_track()
    transcript.segments.insert(1, Segment(words=[]))
    levels = measure_segment_levels(audio, transcript)
    assert len(levels) == 2


def test_level_rejects_bad_params_and_empty_input():
    audio, transcript = _two_tone_track()
    with pytest.raises(LevelError, match="must be <= 0"):
        level_segments(audio, transcript, target_dbfs=3.0)
    with pytest.raises(LevelError, match="must sit below"):
        level_segments(audio, transcript, target_dbfs=-0.5, ceiling_dbfs=-1.0)
    with pytest.raises(LevelError, match="must be positive"):
        level_segments(audio, transcript, max_gain_db=0.0)
    with pytest.raises(LevelError, match="no timed segments"):
        level_segments(audio, Transcript(segments=[]))


def test_segment_past_audio_end_is_a_caller_bug():
    audio, transcript = _two_tone_track()
    transcript.segments.append(_seg("beyond the track", 5.0, 5.5))
    with pytest.raises(LevelError, match="past the"):
        measure_segment_levels(audio, transcript)


def test_report_json_round_trip():
    audio, transcript = _two_tone_track()
    _levelled, report = level_segments(audio, transcript)
    rebuilt = LevelReport.from_json(report.to_json())
    assert rebuilt.target_dbfs == report.target_dbfs
    assert rebuilt.segments_levelled == report.segments_levelled
    assert rebuilt.segments_at_cap == report.segments_at_cap
    assert [row.gain_db for row in rebuilt.levels] == [
        pytest.approx(row.gain_db, abs=0.01) for row in report.levels
    ]
    assert rebuilt.levels[0].index == 1


def test_report_markdown_names_cap_and_silence():
    audio, transcript = _two_tone_track()
    transcript.segments.append(_seg("nothing said here", 0.5, 1.0))
    _levelled, report = level_segments(audio, transcript, max_gain_db=2.0)
    text = report.to_markdown()
    assert "Target RMS: -16.0 dBFS" in text
    assert "(at cap)" in text
    assert "silent" in text
    assert "Max deviation" in text


def test_segment_level_json_round_trip_with_silence():
    level = SegmentLevel(
        index=1, start=0.0, end=0.5, rms_dbfs=float("-inf"), peak_dbfs=float("-inf"), silent=True
    )
    rebuilt = SegmentLevel.from_dict(level.to_dict())
    assert rebuilt.rms_dbfs == float("-inf")
    assert rebuilt.silent
    with pytest.raises(ValueError, match="not a valid segment level"):
        SegmentLevel.from_dict({"index": 1})
    with pytest.raises(ValueError, match="not valid JSON"):
        LevelReport.from_json("{not json")
    with pytest.raises(ValueError, match="not a valid level report"):
        LevelReport.from_dict({"target_dbfs": -16.0})


def test_gain_ramps_avoid_clicks_at_edges():
    # A hard gain step would put a discontinuity at the segment edge;
    # the 5 ms ramp keeps the first/last samples near unity gain.
    audio, transcript = _two_tone_track()
    levelled, _report = level_segments(audio, transcript, max_gain_db=20.0)
    # First sample of the quiet segment is ~unchanged (ramp starts at 1.0).
    assert abs(levelled.samples[0] - audio.samples[0]) < 1e-6
    # Mid-segment the gain is fully applied: sample 0.25s in is ~13 dB up.
    mid = int(0.25 * 16_000)
    ratio = levelled.samples[mid] / audio.samples[mid]
    assert ratio == pytest.approx(10.0 ** (13.0 / 20.0), abs=0.3)
