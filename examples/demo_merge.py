"""Demo of segment merging.

A hesitant speaker arrives as five 0.4 s fragments with 0.2 s pauses
between them — five tiny syntheses, five fit factors, subtitles that
flash past. Merging joins them into two readable lines (the speaker
change at the end stays separate), and the report shows exactly which
segments joined.

Run it with ``python examples/demo_merge.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.merge import merge_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float, speaker: str | None = None) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[
            Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)
        ],
        speaker=speaker,
    )


def main() -> int:
    transcript = Transcript(
        segments=[
            _seg("welcome", 0.0, 0.4),
            _seg("to the", 0.6, 1.0),
            _seg("final", 1.2, 1.6),
            _seg("good luck", 3.0, 3.8, speaker="Ravi"),
            _seg("everyone", 4.0, 4.6, speaker="Ravi"),
        ],
        language="en",
    )
    merged, report = merge_transcript(transcript)
    print(report.to_markdown(), end="")
    print("--- merged transcript ---")
    for segment in merged.segments:
        print(f"[{segment.start:5.2f} - {segment.end:5.2f}] {segment.speaker or '-'}: {segment.text}")

    assert len(merged.segments) == 2
    assert merged.segments[0].text == "welcome to the final"
    assert merged.segments[1].text == "good luck everyone"
    assert merged.segments[1].speaker == "Ravi"
    assert report.segments_before == 5 and report.segments_after == 2
    print("checks passed: fragments joined, speaker change respected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
