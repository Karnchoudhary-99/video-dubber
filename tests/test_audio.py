import pytest

from video_dubber.audio import (
    concat,
    load_wav,
    make_silence,
    make_sine,
    resample,
    save_wav,
)


def test_make_sine_duration_and_peak():
    audio = make_sine(440.0, 0.5, amplitude=0.5)
    assert audio.sample_rate == 16_000
    assert len(audio.samples) == 8_000
    assert audio.duration == pytest.approx(0.5)
    assert audio.peak == pytest.approx(0.5, abs=0.01)


def test_slice_clamps_to_bounds():
    audio = make_sine(440.0, 1.0)
    assert audio.slice(0.25, 0.75).duration == pytest.approx(0.5)
    assert audio.slice(-1.0, 99.0).duration == pytest.approx(1.0)
    assert audio.slice(0.8, 0.2).duration == pytest.approx(0.0)


def test_normalized_scales_peak_and_keeps_silence():
    audio = make_sine(440.0, 0.1, amplitude=0.25)
    assert audio.normalized(0.9).peak == pytest.approx(0.9, abs=0.01)
    assert make_silence(0.1).normalized().peak == 0.0


def test_save_load_wav_round_trip(tmp_path):
    audio = make_sine(440.0, 0.25, amplitude=0.8)
    path = tmp_path / "tone.wav"
    save_wav(str(path), audio)
    loaded = load_wav(str(path))
    assert loaded.sample_rate == 16_000
    assert loaded.duration == pytest.approx(0.25)
    assert loaded.peak == pytest.approx(0.8, abs=0.002)
    # Sample-for-sample close (16-bit quantisation only).
    for got, want in zip(loaded.samples, audio.samples):
        assert got == pytest.approx(want, abs=1 / 32767 + 1e-9)


def test_load_wav_rejects_non_wav(tmp_path):
    path = tmp_path / "not-audio.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="not a readable WAV"):
        load_wav(str(path))


def test_load_wav_missing_file():
    with pytest.raises(FileNotFoundError):
        load_wav("/definitely/not/here.wav")


def test_resample_down_and_up():
    audio = make_sine(440.0, 1.0)
    down = resample(audio, 8000)
    assert down.sample_rate == 8000
    assert down.duration == pytest.approx(1.0)
    assert down.peak == pytest.approx(0.5, abs=0.02)
    up = resample(down, 16_000)
    assert up.sample_rate == 16_000
    assert up.duration == pytest.approx(1.0)
    with pytest.raises(ValueError):
        resample(audio, 0)


def test_resample_identity_returns_copy():
    audio = make_sine(440.0, 0.1)
    copy = resample(audio, 16_000)
    assert copy.samples == audio.samples
    assert copy.samples is not audio.samples


def test_concat_mixes_rates():
    a = make_sine(440.0, 0.2)
    b = make_sine(880.0, 0.3, sample_rate=8000)
    joined = concat(a, b)
    assert joined.sample_rate == 16_000
    assert joined.duration == pytest.approx(0.5)
    assert concat().duration == 0.0


def test_stereo_wav_mixes_to_mono(tmp_path):
    import struct
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        frames = b"".join(
            struct.pack("<hh", int(0.5 * 32767), int(-0.5 * 32767)) for _ in range(800)
        )
        wav.writeframes(frames)
    loaded = load_wav(str(path))
    assert loaded.sample_rate == 8000
    assert loaded.duration == pytest.approx(0.1)
    # Left +0.5 and right -0.5 average to ~0.
    assert loaded.peak == pytest.approx(0.0, abs=1e-3)


def test_24bit_wav_decodes(tmp_path):
    import wave

    path = tmp_path / "wide.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(8000)
        value = int(0.5 * 8388608)
        wav.writeframes(value.to_bytes(3, "little", signed=True) * 800)
    loaded = load_wav(str(path))
    assert loaded.peak == pytest.approx(0.5, abs=1e-3)
