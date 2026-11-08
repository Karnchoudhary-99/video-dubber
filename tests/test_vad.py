import pytest

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.vad import VadConfig, detect_speech, speech_ratio


def _speech_burst() -> object:
    """1s tone, 1s silence, 1s tone, 1s silence — two clear speech bursts."""
    return concat(
        make_sine(440.0, 1.0),
        make_silence(1.0),
        make_sine(440.0, 1.0),
        make_silence(1.0),
    )


def test_detects_two_bursts():
    segments = detect_speech(_speech_burst())
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.0, abs=0.1)
    assert segments[0].end == pytest.approx(1.0, abs=0.1)
    assert segments[1].start == pytest.approx(2.0, abs=0.1)
    assert segments[1].end == pytest.approx(3.0, abs=0.1)


def test_silence_has_no_speech():
    assert detect_speech(make_silence(2.0)) == []
    assert speech_ratio(make_silence(2.0)) == 0.0


def test_empty_audio_has_no_speech():
    from video_dubber.audio import AudioData

    assert detect_speech(AudioData([], 16_000)) == []
    assert speech_ratio(AudioData([], 16_000)) == 0.0


def test_short_click_is_dropped():
    # A 50 ms burst is below the 120 ms minimum — dropped as a click.
    audio = concat(make_sine(440.0, 0.05), make_silence(1.0))
    assert detect_speech(audio) == []


def test_brief_pause_inside_speech_is_bridged():
    # Two 0.5 s bursts separated by a 0.15 s gap (< 300 ms) merge into one.
    audio = concat(make_sine(440.0, 0.5), make_silence(0.15), make_sine(440.0, 0.5))
    segments = detect_speech(audio)
    assert len(segments) == 1
    assert segments[0].duration == pytest.approx(1.15, abs=0.15)


def test_long_pause_splits_segments():
    audio = concat(make_sine(440.0, 0.5), make_silence(0.6), make_sine(440.0, 0.5))
    assert len(detect_speech(audio)) == 2


def test_explicit_threshold_overrides_adaptive():
    audio = _speech_burst()
    # Absurdly high threshold: nothing counts as speech.
    assert detect_speech(audio, VadConfig(energy_threshold=10.0)) == []
    # Zero threshold: everything counts, one segment spans the file.
    segments = detect_speech(audio, VadConfig(energy_threshold=0.0))
    assert len(segments) == 1
    assert segments[0].duration == pytest.approx(4.0, abs=0.1)


def test_speech_ratio_matches_segments():
    audio = _speech_burst()
    assert speech_ratio(audio) == pytest.approx(0.5, abs=0.1)


def test_quiet_speech_below_noise_floor_is_missed_honestly():
    # A burst 100x quieter than the surrounding tone floor is not speech
    # to an energy VAD — document the limit instead of faking a pass.
    loud = make_sine(440.0, 1.0, amplitude=0.5)
    quiet = make_sine(440.0, 1.0, amplitude=0.005)
    audio = concat(loud, quiet)
    segments = detect_speech(audio)
    assert len(segments) == 1
    assert segments[0].end == pytest.approx(1.0, abs=0.15)
