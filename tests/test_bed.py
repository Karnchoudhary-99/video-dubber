"""Tests for the background-music bed (extract / duck / mix)."""

import pytest

from video_dubber.audio import AudioData, make_silence, make_sine
from video_dubber.bed import BedError, duck_bed, extract_bed, mix_bed


def _flat(value: float, duration: float, rate: int = 16_000) -> AudioData:
    return AudioData([value] * int(duration * rate), rate)


def test_extract_bed_keeps_gaps_and_mutes_speech():
    original = _flat(0.5, 4.0)
    bed = extract_bed(original, [(1.0, 2.0)])
    assert bed.sample_rate == 16_000
    assert bed.samples[int(0.5 * 16_000)] == pytest.approx(0.5)
    assert bed.samples[int(1.5 * 16_000)] == pytest.approx(0.0)
    assert bed.samples[int(2.5 * 16_000)] == pytest.approx(0.5)


def test_extract_bed_fades_at_window_edges():
    original = _flat(0.5, 4.0)
    bed = extract_bed(original, [(1.0, 2.0)])
    # 30 ms fade: halfway into the fade the gain is ~0.5.
    assert bed.samples[int(1.015 * 16_000)] == pytest.approx(0.25, abs=0.01)
    assert bed.samples[int(1.985 * 16_000)] == pytest.approx(0.25, abs=0.01)
    # Just outside the window the bed is back at full level.
    assert bed.samples[int(0.999 * 16_000)] == pytest.approx(0.5)
    assert bed.samples[int(2.001 * 16_000)] == pytest.approx(0.5)


def test_extract_bed_multiple_windows_and_out_of_range():
    original = _flat(0.5, 4.0)
    bed = extract_bed(original, [(0.5, 1.0), (3.0, 3.5), (9.0, 10.0), (-2.0, -1.0)])
    assert bed.samples[int(0.75 * 16_000)] == pytest.approx(0.0)
    assert bed.samples[int(2.0 * 16_000)] == pytest.approx(0.5)
    assert bed.samples[int(3.25 * 16_000)] == pytest.approx(0.0)
    # Out-of-range windows are ignored, not errors.
    assert bed.samples[int(3.9 * 16_000)] == pytest.approx(0.5)


def test_extract_bed_empty_audio_and_bad_fade():
    assert extract_bed(AudioData([], 16_000), [(0.0, 1.0)]).samples == []
    with pytest.raises(BedError):
        extract_bed(_flat(0.5, 1.0), [(0.0, 0.5)], fade_ms=0)


def test_duck_bed_attenuates_under_speech_only():
    bed = _flat(0.5, 4.0)
    ducked = duck_bed(bed, [(1.0, 1.0)], duck_level=0.2)
    assert ducked.samples[int(0.5 * 16_000)] == pytest.approx(0.5)
    assert ducked.samples[int(1.5 * 16_000)] == pytest.approx(0.1)
    assert ducked.samples[int(2.5 * 16_000)] == pytest.approx(0.5)


def test_duck_bed_fades_at_clip_boundaries():
    bed = _flat(0.5, 4.0)
    ducked = duck_bed(bed, [(1.0, 1.0)], duck_level=0.2)
    # 50 ms fade: halfway in, gain is ~0.6 (midway between 1.0 and 0.2).
    assert ducked.samples[int(1.025 * 16_000)] == pytest.approx(0.3, abs=0.01)
    assert ducked.samples[int(1.975 * 16_000)] == pytest.approx(0.3, abs=0.01)


def test_duck_bed_level_bounds():
    bed = _flat(0.5, 2.0)
    assert duck_bed(bed, [(0.5, 1.0)], duck_level=0.0).samples[int(1.0 * 16_000)] == pytest.approx(
        0.0
    )
    assert duck_bed(bed, [(0.5, 1.0)], duck_level=1.0).samples[int(1.0 * 16_000)] == pytest.approx(
        0.5
    )
    with pytest.raises(BedError):
        duck_bed(bed, [(0.5, 1.0)], duck_level=1.5)
    with pytest.raises(BedError):
        duck_bed(bed, [(0.5, 1.0)], duck_level=-0.1)
    assert duck_bed(AudioData([], 16_000), [(0.0, 1.0)]).samples == []


def test_mix_bed_adds_at_level_and_clamps():
    dub = _flat(0.4, 2.0)
    bed = _flat(0.5, 2.0)
    mixed = mix_bed(dub, bed, bed_level=0.3)
    assert mixed.samples[100] == pytest.approx(0.55)
    loud = mix_bed(_flat(0.9, 1.0), _flat(0.5, 1.0), bed_level=0.5)
    assert loud.samples[10] == pytest.approx(1.0)  # 0.9 + 0.25 would clip
    assert mix_bed(dub, bed, bed_level=0.0).samples == dub.samples


def test_mix_bed_length_and_rate_mismatch():
    dub = _flat(0.2, 2.0)
    short_bed = _flat(0.5, 1.0)  # shorter: zero-padded past 1s
    mixed = mix_bed(dub, short_bed, bed_level=0.5)
    assert mixed.samples[int(0.5 * 16_000)] == pytest.approx(0.45)
    assert mixed.samples[int(1.5 * 16_000)] == pytest.approx(0.2)
    long_bed = _flat(0.5, 4.0)  # longer: truncated to the dub's length
    assert len(mix_bed(dub, long_bed, bed_level=0.5).samples) == len(dub.samples)
    slow_bed = AudioData([0.5] * 8000, 8000)  # 1s at 8 kHz -> resampled to 16 kHz
    mixed = mix_bed(dub, slow_bed, bed_level=0.5)
    assert mixed.sample_rate == 16_000
    assert mixed.samples[int(0.5 * 16_000)] == pytest.approx(0.45)
    with pytest.raises(BedError):
        mix_bed(dub, bed=_flat(0.5, 1.0), bed_level=2.0)


def test_mix_bed_empty_inputs():
    assert mix_bed(AudioData([], 16_000), _flat(0.5, 1.0)).samples == []
    assert mix_bed(_flat(0.4, 1.0), AudioData([], 16_000)).samples == _flat(0.4, 1.0).samples


def test_extract_bed_preserves_real_tone_in_gaps():
    tone = make_sine(440.0, 2.0)
    bed = extract_bed(tone, [(0.5, 1.0)])
    assert bed.samples[int(0.25 * 16_000)] == pytest.approx(tone.samples[int(0.25 * 16_000)])
    assert bed.samples[int(0.75 * 16_000)] == pytest.approx(0.0)
    assert bed.samples[int(1.5 * 16_000)] == pytest.approx(tone.samples[int(1.5 * 16_000)])
    assert make_silence(1.0).samples == [0.0] * 16_000
