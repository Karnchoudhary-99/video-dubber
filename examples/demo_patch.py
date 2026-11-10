"""Demo of segment patching: fix one bad line without re-dubbing everything.

Dubs a two-segment Hindi transcript, then patches the second segment
with corrected text. Prints the quality report before and after the
patch, and checks the first segment's audio is bit-identical while
the second segment's text (and sound) changed.

Run it with ``python examples/demo_patch.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import SegmentRow, build_report, dub_transcript
from video_dubber.audio import make_sine
from video_dubber.patch import patch_dub, patch_transcript, refresh_report
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    translated = Transcript(
        segments=[
            Segment(
                words=[
                    Word("नमस्ते", 0.0, 0.4),
                    Word("दुनिया", 0.4, 0.8),
                    Word("दोस्तों", 0.8, 1.2),
                    Word("स्वागत", 1.2, 1.6),
                    Word("है", 1.6, 2.0),
                ],
            ),
            Segment(
                words=[
                    Word("शानदार", 3.0, 3.4),
                    Word("गोल", 3.4, 3.8),
                    Word("देखिए", 3.8, 4.2),
                    Word("फिर", 4.2, 4.6),
                    Word("से", 4.6, 5.0),
                ],
                speaker="Ravi",
            ),
        ],
        language="hi",
    )
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Voice", [make_sine(180.0, 1.0)])
    dub_audio, fit_factors, placed = dub_transcript(translated, cloner, profile)
    report = build_report(translated, placed, fit_factors, dub_audio)
    print("--- report before the patch ---")
    print(report.to_markdown(), end="")

    patched_transcript = patch_transcript(translated, 2, "बदला हुआ गोल देखिए")
    patched_audio, factor, placed_duration = patch_dub(
        dub_audio, translated, 2, "बदला हुआ गोल देखिए", cloner, profile, placed=placed
    )
    segment = [s for s in patched_transcript.segments if s.words][1]
    refreshed = refresh_report(
        report,
        SegmentRow(
            index=2,
            start=segment.start,
            end=segment.end,
            drift=abs((segment.start + placed_duration) - segment.end),
            fit_factor=factor,
            speaker=segment.speaker,
            text="बदला हुआ गोल देखिए",
            placed_duration=placed_duration,
        ),
    )
    print("--- report after patching segment 2 ---")
    print(refreshed.to_markdown(), end="")
    print("--- patched transcript (segment 2) ---")
    print(patched_transcript.segments[1].text)

    assert patched_audio.duration == dub_audio.duration, "patch must not move timing"
    assert patched_audio.samples[: 3 * 16_000] == dub_audio.samples[: 3 * 16_000], (
        "segment 1 must be bit-identical after patching segment 2"
    )
    assert patched_audio.samples[3 * 16_000 :] != dub_audio.samples[3 * 16_000 :], (
        "segment 2 must actually change"
    )
    assert refreshed.segments[1].text == "बदला हुआ गोल देखिए"
    print("checks passed: one line fixed, the rest of the dub untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
