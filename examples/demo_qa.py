"""Demo: readability-check a dubbed transcript's subtitles before muxing.

Builds a five-cue transcript with one of every QA problem — a line
too fast to read, a line too long for the screen and too long on it,
an overlapping cue, and a cue that is both too short and hard on the
heels of the last one — then prints the report. A second, clean
transcript shows the ``pass`` verdict.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_qa.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.qa import qa_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def main() -> int:
    messy = Transcript(
        segments=[
            _seg("Welcome to Mars, everyone.", 0.0, 2.0),
            # 47 chars in 1.0s = 47 chars/s, over the 20 cap.
            _seg("Pack as many words as you can into this cue now", 2.5, 3.5),
            # 8s on screen (max 7) and wraps to 3 lines at 42 chars.
            _seg(
                "This line goes on and on and on until it simply cannot fit on two lines at all, not even close",
                4.0,
                12.0,
            ),
            # Starts 500 ms before the previous cue ends.
            _seg("Second line here.", 11.5, 13.5),
            # 0.4s long (min 1s) and only 50 ms after the previous cue.
            _seg("Go!", 13.55, 13.95),
        ]
    )
    report = qa_transcript(messy)
    print(report.to_markdown(), end="")
    assert report.verdict == "review"
    assert report.errors == 1
    assert report.warnings == 5
    assert len(report.issues) == 6
    kinds = [issue.kind for issue in report.issues]
    assert kinds == ["too-fast", "too-long", "too-many-lines", "overlap", "tiny-gap", "too-short"]

    clean = Transcript(
        segments=[
            _seg("Welcome to Mars, everyone.", 0.0, 2.0),
            _seg("The match starts in five minutes.", 2.5, 5.0),
        ]
    )
    clean_report = qa_transcript(clean)
    print(f"clean transcript verdict: {clean_report.verdict}")
    assert clean_report.verdict == "pass"
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
