"""Demo: even out per-segment loudness in a dubbed track.

Builds a synthetic two-line dub — a quiet line at -29 dBFS, a loud
line at -9 dBFS, half a second of silence between them — then levels
both lines toward -16 dBFS and prints the report. A second pass shows
the default gain cap honestly: with the cap left at 12 dB the quiet
line can only reach -17 dBFS, and the report says so (``at cap``)
instead of pretending it hit the target.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_level.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.level import level_segments, measure_segment_levels
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def main() -> int:
    audio = concat(
        make_sine(440.0, 0.5, amplitude=0.05),  # quiet line: -29 dBFS
        make_silence(0.5),
        make_sine(440.0, 0.5, amplitude=0.5),  # loud line: -9 dBFS
    )
    transcript = Transcript(
        segments=[_seg("quiet line here", 0.0, 0.5), _seg("loud line here", 1.0, 1.5)]
    )
    before = measure_segment_levels(audio, transcript)
    print(
        "before: "
        + ", ".join(f"segment {level.index} RMS {level.rms_dbfs:.1f} dBFS" for level in before)
    )

    levelled, report = level_segments(audio, transcript, max_gain_db=20.0)
    print(report.to_markdown(), end="")
    after = measure_segment_levels(levelled, transcript)
    assert abs(after[0].rms_dbfs - (-16.0)) < 0.5
    assert abs(after[1].rms_dbfs - (-16.0)) < 0.5
    assert levelled.duration == audio.duration  # timing never moves

    # With the default 12 dB cap the quiet line cannot reach the target:
    # it needs +13 dB, so it lands at -17 dBFS and the report flags it.
    _capped, capped_report = level_segments(audio, transcript)
    print(
        f"capped: quiet line gain {capped_report.levels[0].gain_db:+.1f} dB, "
        f"at cap: {capped_report.levels[0].at_cap}, "
        f"max deviation after: {capped_report.max_deviation_after_db:.1f} dB"
    )
    assert capped_report.levels[0].at_cap
    assert capped_report.max_deviation_after_db > 1.0
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
