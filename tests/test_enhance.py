import math

import pytest

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.enhance import (
    EnhanceError,
    enhance_audio,
    high_pass,
    measure_loudness,
    noise_gate,
    normalize_loudness,
    trim_silence,
)


def _rms(audio: AudioData) -> float:
    if not audio.samples:
        return 0.0
    return math.sqrt(sum(s * s for s in audio.samples) / len(audio.samples))


def test_measure_loudness_sine_and_silence():
    audio = make_sine(440.0, 0.5, amplitude=0.5)
    stats = measure_loudness(audio)
    # RMS of a sine is amplitude / sqrt(2).
    assert stats.rms_dbfs == pytest.approx(-9.03, abs=0.1)
    assert stats.peak_dbfs == pytest.approx(-6.02, abs=0.1)
    assert stats.crest_db == pytest.approx(3.01, abs=0.1)
    assert stats.dc_offset == pytest.approx(0.0, abs=1e-3)
    assert stats.duration == pytest.approx(0.5)

    silent = measure_loudness(make_silence(0.25))
    assert silent.rms_dbfs == float("-inf")
    assert silent.peak_dbfs == float("-inf")
    assert silent.crest_db == 0.0
    assert measure_loudness(AudioData([], 16_000)).duration == 0.0


def test_normalize_hits_target_rms_and_respects_ceiling():
    audio = make_sine(440.0, 1.0, amplitude=0.1)
    out, gain_db = normalize_loudness(audio, target_dbfs=-16.0, ceiling_dbfs=-1.0)
    assert measure_loudness(out).rms_dbfs == pytest.approx(-16.0, abs=0.2)
    assert measure_loudness(out).peak_dbfs <= -1.0 + 1e-9
    assert gain_db == pytest.approx(-16.0 - measure_loudness(audio).rms_dbfs, abs=0.1)


def test_normalize_caps_gain_at_ceiling_for_high_crest_audio():
    # Mostly quiet, one full-scale spike: the gain that hits the RMS
    # target would blow the peak past the ceiling, so the ceiling wins.
    samples = [0.01 * math.sin(2 * math.pi * 440 * i / 16_000) for i in range(16_000)]
    samples[8000] = 1.0
    audio = AudioData(samples, 16_000)
    out, _ = normalize_loudness(audio, target_dbfs=-16.0, ceiling_dbfs=-1.0)
    assert out.peak <= 10.0 ** (-1.0 / 20.0) + 1e-9
    assert measure_loudness(out).rms_dbfs < -16.0  # ceiling capped the gain


def test_normalize_leaves_silence_alone_and_rejects_bad_params():
    silent = make_silence(0.5)
    out, gain_db = normalize_loudness(silent)
    assert out.samples == silent.samples
    assert gain_db == 0.0
    audio = make_sine(440.0, 0.1)
    with pytest.raises(EnhanceError, match="<= 0 dBFS"):
        normalize_loudness(audio, target_dbfs=3.0)
    with pytest.raises(EnhanceError, match="below the peak ceiling"):
        normalize_loudness(audio, target_dbfs=-1.0, ceiling_dbfs=-6.0)


def test_high_pass_keeps_speech_and_cuts_hum():
    speech = make_sine(440.0, 1.0, amplitude=0.5)
    # Steady state (skip the first 100 ms of filter settling).
    # Two cascaded stages pass 440 Hz at ~0.94 (-0.5 dB) — essentially
    # untouched for speech — while 50 Hz hum is cut to ~0.28.
    tail = AudioData(high_pass(speech).samples[1600:], 16_000)
    assert _rms(tail) >= 0.92 * _rms(speech)
    hum = make_sine(50.0, 1.0, amplitude=0.5)
    assert _rms(high_pass(hum)) <= 0.35 * _rms(hum)
    dc = AudioData([0.4] * 16_000, 16_000)
    assert _rms(high_pass(dc)) <= 0.01 * _rms(dc)
    assert high_pass(AudioData([], 16_000)).samples == []
    with pytest.raises(EnhanceError, match="positive"):
        high_pass(speech, cutoff_hz=0.0)
    with pytest.raises(EnhanceError, match="Nyquist"):
        high_pass(speech, cutoff_hz=9000.0)


def test_noise_gate_quietens_pauses_but_keeps_speech():
    quiet = make_sine(440.0, 0.5, amplitude=0.001)
    loud = make_sine(440.0, 0.5, amplitude=0.5)
    gated = noise_gate(concat(quiet, loud))
    # Settled tail of the quiet half (the 100 ms release takes a moment
    # to close the gate, so the first frames are still audible).
    gated_quiet_tail = AudioData(gated.samples[4800:8000], 16_000)
    assert _rms(gated_quiet_tail) <= 0.05 * _rms(quiet)  # pushed toward the floor
    gated_loud = AudioData(gated.samples[8000:], 16_000)
    assert _rms(gated_loud) == pytest.approx(_rms(loud), rel=0.15)
    assert noise_gate(make_silence(0.2)).samples == [0.0] * 3200
    with pytest.raises(EnhanceError, match="below the threshold"):
        noise_gate(loud, threshold_dbfs=-45.0, floor_db=-20.0)


def test_trim_silence_keeps_margin_and_reports_trimmed():
    audio = concat(make_silence(0.3), make_sine(440.0, 0.5, amplitude=0.5), make_silence(0.3))
    trimmed, trimmed_s = trim_silence(audio)
    assert trimmed.duration == pytest.approx(0.6, abs=0.02)  # speech + 2 x 50 ms margin
    assert trimmed_s == pytest.approx(0.5, abs=0.02)
    assert _rms(trimmed) > 0.2  # the speech survived
    empty, trimmed_all = trim_silence(make_silence(0.5))
    assert empty.samples == []
    assert trimmed_all == pytest.approx(0.5, abs=0.02)


def test_enhance_audio_pipeline_hits_target_and_reports():
    audio = concat(make_silence(0.2), make_sine(440.0, 0.5, amplitude=0.1), make_silence(0.2))
    out, report = enhance_audio(audio)
    assert measure_loudness(out).rms_dbfs == pytest.approx(-16.0, abs=0.5)
    assert measure_loudness(out).peak_dbfs <= -1.0 + 1e-9
    assert report.trimmed_s == pytest.approx(0.3, abs=0.05)
    assert report.gain_db > 0.0  # quiet input was turned up
    assert report.before.rms_dbfs < report.after.rms_dbfs
    text = report.to_markdown()
    assert "RMS:" in text and "-16.0 dBFS" in text
    assert report.to_dict()["gain_db"] == pytest.approx(report.gain_db)


def test_enhance_audio_no_trim_keeps_duration_for_timed_dubs():
    audio = concat(make_silence(0.2), make_sine(440.0, 0.5, amplitude=0.1), make_silence(0.2))
    out, report = enhance_audio(audio, trim=False)
    assert out.duration == pytest.approx(0.9)
    assert report.trimmed_s == 0.0
    assert measure_loudness(out).rms_dbfs == pytest.approx(-16.0, abs=1.0)
