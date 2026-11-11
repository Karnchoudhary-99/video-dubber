"""Demo of styled ASS subtitle export.

Dubs a two-speaker Hindi transcript, then prints the .ass file the dub
writes next to the audio: one named style per speaker (Asha white,
Ravi yellow), with the cues retimed to the placed dub audio. The same
file is what ``dubber render --subtitles`` burns into the picture, so
the colours on screen match the colours in this file.

Run it with ``python examples/demo_ass.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import dub_transcript
from video_dubber.audio import make_sine
from video_dubber.subtitles import retime_transcript, to_ass
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    translated = Transcript(
        segments=[
            Segment(
                words=[
                    Word("नमस्ते", 0.0, 0.5),
                    Word("दुनिया", 0.5, 1.0),
                ],
                speaker="Asha",
            ),
            Segment(
                words=[Word("शानदार", 2.0, 2.5)],
                speaker="Ravi",
            ),
        ],
        language="hi",
    )
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Voice", [make_sine(180.0, 1.0)])
    _audio, _fit_factors, placed = dub_transcript(translated, cloner, profile)
    synced = retime_transcript(translated, placed)

    ass = to_ass(synced)
    print(ass)
    assert "[V4+ Styles]" in ass
    assert "Style: Asha," in ass and "Style: Ravi," in ass
    assert ass.count("Dialogue:") == 2
    # Asha's cue uses Asha's style (white), Ravi's uses Ravi's (yellow).
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Asha,Asha,0,0,0,,नमस्ते दुनिया" in ass
    assert "Dialogue: 0,0:00:02.00,0:00:02.50,Ravi,Ravi,0,0,0,,शानदार" in ass
    print("checks passed: two speakers, two styles, cues retimed to the dub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
