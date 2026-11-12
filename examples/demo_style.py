"""Demo: dub the same transcript in three delivery styles.

A style is performance, not identity: the same cloned voice reads
faster and higher as ``energetic``, slower and lower as ``dramatic``.
The demo dubs one two-line transcript plain, energetic and dramatic
and prints each dub's mean fit factor — the faster the style, the
larger the stretch needed to fill the same windows — plus the verdict
and the style recorded in the quality report.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_style.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.style import list_styles
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import EMBEDDING_DIM, ScriptedVoiceCloner, SpeakerProfile


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def main() -> int:
    print(f"{len(list_styles())} dubbing style(s)")
    for style in list_styles():
        print(
            f"  {style.name}: {style.rate_factor:.2f}x rate, "
            f"{style.pitch_semitones:+.0f} semitones — {style.description}"
        )
    print()

    transcript = Transcript(
        segments=[
            _seg("welcome to the mars sports league", 0.0, 2.5),
            _seg("and now the final over begins", 2.5, 5.0),
        ],
        language="hi",
    )
    profile = SpeakerProfile(name="Demo", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)
    cloner = ScriptedVoiceCloner()

    fits: dict[str, float] = {}
    for style_name in (None, "energetic", "dramatic"):
        dub, factors, placed = dub_transcript(transcript, cloner, profile, style=style_name)
        report = build_report(transcript, placed, factors, dub, style=style_name)
        label = style_name or "plain"
        fits[label] = report.mean_fit_factor
        print(
            f"{label}: mean fit {report.mean_fit_factor:.2f} "
            f"(max {report.max_fit_factor:.2f}), "
            f"verdict {report.verdict}, style recorded as {report.style!r}"
        )
        assert report.style == style_name
    # Faster speech needs a bigger stretch to fill the same windows;
    # slower speech needs a smaller one. The ordering is the point.
    assert fits["energetic"] > fits["plain"] > fits["dramatic"]
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
