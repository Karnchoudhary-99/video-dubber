"""Demo of dub-synced subtitles.

Dubs a two-segment Hindi transcript where the first segment is one
short word in a long window — its stretch clamps at 1.6x, so the dubbed
speech ends well before the original window does. Prints the SRT as it
used to be written (original windows) next to the retimed SRT (placed
spans), and checks every cue ends where the placed audio ends.

Run it with ``python examples/demo_subtitles.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import dub_transcript
from video_dubber.audio import make_sine
from video_dubber.subtitles import retime_transcript
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    translated = Transcript(
        segments=[
            Segment(
                words=[Word("नमस्ते", 0.0, 4.0)],
                speaker="Asha",
            ),
            Segment(
                words=[
                    Word("शानदार", 5.0, 5.5),
                    Word("गोल", 5.5, 6.0),
                ],
                speaker="Ravi",
            ),
        ],
        language="hi",
    )
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Voice", [make_sine(180.0, 1.0)])
    _audio, fit_factors, placed = dub_transcript(translated, cloner, profile)
    print(f"fit factors: {[round(f, 2) for f in fit_factors]}")
    print(f"placed spans: {[(round(s, 2), round(d, 2)) for s, d in placed]}")

    print("\n--- SRT with original windows (what dub --subtitles used to write) ---")
    print(translated.to_srt())
    synced = retime_transcript(translated, placed)
    print("--- SRT retimed to the dubbed audio ---")
    print(synced.to_srt())

    for segment, (start, duration) in zip([s for s in synced.segments if s.words], placed):
        # A cue ends at its placed end — or at the 0.5s readable minimum
        # when the placed clip is shorter than that, never later.
        assert abs(segment.start - start) < 0.01, "cue start must match placed start"
        assert segment.end + 0.01 >= start + duration, "cue must cover the placed audio"
        assert segment.end <= start + max(duration, 0.5) + 0.01, "cue must not overrun"
    print("checks passed: every cue ends where the placed dub audio ends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
