"""Demo: shift subtitle cues to the speech in the rendered dub audio.

Builds a synthetic two-line dub — each line's tone starts 0.3s
after its planned window opens (synthesis leading silence) — then
aligns the cues to the detected onsets and prints the report. A
second pass shows the shift clamp honestly: with ``max_shift_s=0.1``
the cues can only move 100 ms, and the report says ``clamped``
instead of pretending they landed on the speech.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_align.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.align import AlignConfig, align_transcript
from video_dubber.audio import concat, make_silence, make_sine
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def main() -> int:
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
    print("planned: " + ", ".join(f"cue {s.start:.2f}-{s.end:.2f}s" for s in transcript.segments))

    aligned, report = align_transcript(audio, transcript)
    print(report.to_markdown(), end="")
    assert abs(aligned.segments[0].start - 0.24) < 0.02
    assert abs(aligned.segments[1].start - 1.44) < 0.02
    assert abs(report.mean_abs_shift_ms - 240.0) < 10.0

    # With the shift clamped to 100 ms the cues cannot reach the speech:
    # they land at +100 ms and the report flags them.
    _capped, capped_report = align_transcript(audio, transcript, AlignConfig(max_shift_s=0.1))
    print(
        f"clamped: cue 1 start shift {capped_report.rows[0].shift_ms:+.0f} ms, "
        f"status: {capped_report.rows[0].status}"
    )
    assert capped_report.rows[0].status == "clamped"
    assert capped_report.rows[0].shift_ms == 100.0
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
