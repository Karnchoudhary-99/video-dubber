"""Demo: voice-consistency verification of a finished dub.

Builds a two-speaker dub from sine tones — Asha at 220 Hz, Ravi at
110 Hz — and verifies it against the voice library: the correct dub
passes. Then the same transcript is checked against a swapped dub
(Ravi's voice on Asha's lines): the report flips to ``review`` — each
swapped segment clears the 0.80 bar for its claimed speaker (0.81)
but sits far closer to the other voice (1.00), so the near-twin rule
drops it to ``uncertain`` and names the voice it actually sounds
like.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_verify.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import concat, make_sine
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.verify import build_verify_report, verify_dub
from video_dubber.voice import VoiceLibrary, clone_from_references


def _library() -> VoiceLibrary:
    library = VoiceLibrary()
    library.add(clone_from_references("Asha", [make_sine(220.0, 1.0)]))
    library.add(clone_from_references("Ravi", [make_sine(110.0, 1.0)]))
    return library


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                words=[Word("hello", 0.0, 0.5), Word("mars", 0.5, 1.0)],
                speaker="Asha",
            ),
            Segment(
                words=[Word("good", 1.0, 1.5), Word("morning", 1.5, 2.0)],
                speaker="Ravi",
            ),
        ],
        language="en",
    )


def main() -> int:
    library = _library()
    correct = concat(make_sine(220.0, 1.0), make_sine(110.0, 1.0))
    report = build_verify_report(verify_dub(correct, _transcript(), library))
    print(report.to_markdown(), end="")
    assert report.verdict == "pass"
    assert report.matches == 2

    swapped = concat(make_sine(110.0, 1.0), make_sine(220.0, 1.0))
    swapped_report = build_verify_report(verify_dub(swapped, _transcript(), library))
    print(swapped_report.to_markdown(), end="")
    assert swapped_report.verdict == "review"
    assert swapped_report.uncertain == 2
    assert swapped_report.mismatches == 0
    assert swapped_report.checks[0].best_match == "Ravi"
    assert "closer to Ravi" in swapped_report.checks[0].detail
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
