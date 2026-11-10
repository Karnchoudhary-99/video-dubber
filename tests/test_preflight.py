"""Tests for the pre-flight audio checks."""

import json

import pytest

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.preflight import PreflightReport, run_preflight
from video_dubber.vad import VadConfig


def _speech() -> AudioData:
    """Two 1s tone bursts separated by 1s of silence — the VAD hears speech."""
    return concat(
        make_sine(440.0, 1.0, amplitude=0.3),
        make_silence(1.0),
        make_sine(440.0, 1.0, amplitude=0.3),
        make_silence(1.0),
    )


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_clean_speech_passes():
    report = run_preflight(_speech())
    assert report.verdict == "pass"
    assert report.failures == []
    assert report.warnings == []
    assert len(report.checks) == 8


def test_empty_audio_fails_duration():
    report = run_preflight(AudioData([], 16_000))
    assert report.verdict == "fail"
    assert _check(report, "duration").status == "fail"


def test_very_short_audio_warns():
    audio = concat(make_sine(440.0, 0.4, amplitude=0.3), make_silence(0.4))
    assert _check(run_preflight(audio), "duration").status == "warn"


def test_non_16k_sample_rate_warns():
    audio = AudioData(_speech().samples, 8000)
    check = _check(run_preflight(audio), "sample_rate")
    assert check.status == "warn"
    assert "8000" in check.detail


def test_clipped_audio_fails():
    audio = AudioData([1.0] * 16_000, 16_000)
    assert _check(run_preflight(audio), "clipping").status == "fail"
    assert run_preflight(audio).verdict == "fail"


def test_a_few_clipped_samples_warn():
    samples = [0.0] * 16_000
    samples[100] = 1.0
    samples[200] = -1.0
    audio = AudioData(samples, 16_000)
    assert _check(run_preflight(audio), "clipping").status == "warn"


def test_dc_offset_warns_then_fails():
    mild = AudioData([0.02] * 16_000, 16_000)
    assert _check(run_preflight(mild), "dc_offset").status == "warn"
    bad = AudioData([0.06] * 16_000, 16_000)
    assert _check(run_preflight(bad), "dc_offset").status == "fail"


def test_silence_fails_level_and_speech():
    report = run_preflight(make_silence(2.0))
    assert _check(report, "level").status == "fail"
    assert _check(report, "speech_coverage").status == "fail"
    assert report.verdict == "fail"


def test_quiet_audio_warns_level():
    audio = concat(
        make_sine(440.0, 1.0, amplitude=0.05),
        make_silence(1.0),
        make_sine(440.0, 1.0, amplitude=0.05),
        make_silence(1.0),
    )
    assert _check(run_preflight(audio), "level").status == "warn"


def test_mostly_silence_warns_speech_coverage():
    audio = concat(make_sine(440.0, 0.5, amplitude=0.3), make_silence(4.5))
    check = _check(run_preflight(audio), "speech_coverage")
    assert check.status == "warn"
    assert check.value == pytest.approx(0.1, abs=0.05)


def test_noisy_recording_fails_snr():
    # Speech bursts at 0.3 amplitude, gaps filled with a loud 0.19 hum:
    # SNR ~4 dB. A fixed VAD threshold keeps the hum out of "speech".
    speech = make_sine(440.0, 1.0, amplitude=0.3)
    hum = make_sine(50.0, 1.0, amplitude=0.19)
    audio = concat(speech, hum, speech, hum)
    report = run_preflight(audio, vad_config=VadConfig(energy_threshold=0.02))
    assert _check(report, "snr").status == "fail"
    assert _check(report, "snr").value == pytest.approx(4.0, abs=1.0)


def test_clean_recording_passes_snr():
    # Same bursts, gaps with only a faint 0.01 hum: SNR ~26 dB.
    speech = make_sine(440.0, 1.0, amplitude=0.3)
    hum = make_sine(50.0, 1.0, amplitude=0.01)
    audio = concat(speech, hum, speech, hum)
    assert _check(run_preflight(audio), "snr").status == "pass"


def test_long_leading_silence_warns():
    audio = concat(make_silence(3.0), make_sine(440.0, 1.0, amplitude=0.3), make_silence(0.5))
    check = _check(run_preflight(audio), "edge_silence")
    assert check.status == "warn"
    assert check.value == pytest.approx(3.0, abs=0.2)


def test_report_json_round_trip():
    report = run_preflight(_speech())
    rebuilt = PreflightReport.from_json(report.to_json())
    assert rebuilt.verdict == report.verdict
    assert [(c.name, c.status) for c in rebuilt.checks] == [
        (c.name, c.status) for c in report.checks
    ]


def test_report_from_bad_json_raises():
    with pytest.raises(ValueError):
        PreflightReport.from_json("{not json")
    with pytest.raises(ValueError):
        PreflightReport.from_json(json.dumps({"verdict": "pass"}))


def test_markdown_names_the_verdict():
    text = run_preflight(make_silence(2.0)).to_markdown()
    assert "fail" in text
    assert "[FAIL] level" in text
