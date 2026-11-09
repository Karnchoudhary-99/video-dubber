"""End-to-end demo of speaker diarization.

A two-person conversation arrives as one unlabelled transcript plus its
audio. Diarization matches each segment's audio against the voice
library (Asha at 220 Hz, Ravi at 110 Hz) and labels the segments, so
the transcript is ready for multi-speaker dubbing with no hand-labelling
— the full chain is ``transcribe -> diarize -> dub``.

The audio here is sine tones standing in for speech (the same labelled
stand-in the test suite uses); the labelling logic is the real one.
Run it with ``python examples/demo_diarize.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import concat, make_silence, make_sine, save_wav
from video_dubber.diarize import diarize_transcript
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner, VoiceLibrary


def _words(texts: list[str], start: float, step: float = 0.5) -> list[Word]:
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def main() -> int:
    cloner = ScriptedVoiceCloner()
    library = VoiceLibrary()
    library.add(cloner.clone("Asha", [make_sine(220.0, 1.0)]))
    library.add(cloner.clone("Ravi", [make_sine(110.0, 1.0)]))
    print(f"library: {', '.join(library.names())} ({len(library)} voices)")

    # Asha speaks 0-2 s, a half-second pause, Ravi speaks 2.5-4.5 s.
    audio = concat(make_sine(220.0, 2.0), make_silence(0.5), make_sine(110.0, 2.0))
    transcript = Transcript(
        segments=[
            Segment(words=_words(["hello", "mars", "welcome", "back"], 0.0)),
            Segment(words=_words(["good", "morning", "from", "phobos"], 2.5)),
        ],
        language="en",
    )
    labelled, report = diarize_transcript(transcript, audio, library)
    for label in report.labels:
        print(
            f"[{label.start:7.2f} - {label.end:7.2f}] "
            f"{label.speaker or 'unknown'} "
            f"(similarity {label.similarity:.3f}) — {label.text}"
        )
    print(
        f"{len(report.labels)} segments: {report.matched} matched, "
        f"{report.kept} kept, {report.unknown} unknown"
    )
    assert [s.speaker for s in labelled.segments] == ["Asha", "Ravi"], (
        "demo mislabelled a segment — the voices did not separate"
    )
    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-diarize-"))
    save_wav(str(out_dir / "clip.wav"), audio)
    (out_dir / "labelled.json").write_text(labelled.to_json(), encoding="utf-8")
    (out_dir / "report.json").write_text(report.to_json(), encoding="utf-8")
    print(f"outputs in {out_dir} (clip.wav, labelled.json, report.json)")
    print("labelled transcript is ready for multi-speaker dubbing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
