"""Demo: clean up a raw dub track with audio enhancement.

Builds a synthetic "raw" recording — a quiet voice tone buried under a
50 Hz hum, with half a second of room silence at each end — then runs
the full enhancement chain (trim, high-pass, noise gate, loudness
normalize) and prints the before/after report. A second pass shows the
dub-track variant (``trim=False``), where the duration is preserved so
segment timings stay put.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_enhance.py``.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import AudioData, concat, make_silence, make_sine
from video_dubber.enhance import enhance_audio, measure_loudness


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _make_raw() -> AudioData:
    """Quiet speech + a low room hum running the whole clip, silence at both ends.

    The hum (amplitude 0.002, about -57 dBFS) sits below the trim
    threshold, so the edge silence still trims — but it is loud enough
    to measure in the gaps before and after enhancement.
    """
    speech = make_sine(440.0, 1.0, amplitude=0.05)
    hum = make_sine(50.0, 2.0, amplitude=0.002)
    middle = AudioData(
        [
            max(-1.0, min(1.0, a + b))
            for a, b in zip(speech.samples, hum.samples[: len(speech.samples)])
        ],
        speech.sample_rate,
    )
    raw = concat(make_silence(0.5), middle, make_silence(0.5))
    return AudioData(
        [s + h for s, h in zip(raw.samples, hum.samples)],
        raw.sample_rate,
    )


def main() -> int:
    raw = _make_raw()
    print(f"raw: {raw.duration:.2f}s, RMS {measure_loudness(raw).rms_dbfs:.1f} dBFS")

    enhanced, report = enhance_audio(raw)
    print(report.to_markdown(), end="")
    print(
        f"enhanced: {enhanced.duration:.2f}s, peak {measure_loudness(enhanced).peak_dbfs:.1f} dBFS"
    )
    assert abs(measure_loudness(enhanced).rms_dbfs - (-16.0)) < 0.5
    assert measure_loudness(enhanced).peak_dbfs <= -1.0
    assert enhanced.duration < raw.duration  # the edge silence was trimmed

    # Dub-track variant: same chain, but the duration must not move.
    timed, timed_report = enhance_audio(raw, trim=False)
    assert timed.duration == raw.duration
    print(
        f"timed (no trim): {timed.duration:.2f}s, "
        f"RMS {timed_report.after.rms_dbfs:.1f} dBFS "
        f"(gain {timed_report.gain_db:+.1f} dB)"
    )
    # The hum is measurably quieter after the high-pass: compare the
    # first 0.5 s (silence + hum only) before and after.
    raw_gap = _rms(raw.samples[:8000])
    timed_gap = _rms(timed.samples[:8000])
    print(f"gap RMS before {raw_gap:.4f} -> after {timed_gap:.4f}")
    assert timed_gap < raw_gap
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
