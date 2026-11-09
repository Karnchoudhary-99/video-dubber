"""End-to-end demo of multi-speaker dubbing.

A two-person Hindi conversation (Asha + Ravi) is dubbed from a
translated transcript: each speaker's segment is rendered in their own
cloned voice, the library identifies a fresh recording as Asha's, and
the translated subtitles are written out as SRT + WebVTT.

No model and no real recordings needed — run it with
``python examples/demo_multispeaker.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.audio import make_sine, save_wav
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import (
    ScriptedVoiceCloner,
    VoiceLibrary,
    cosine_similarity,
    extract_embedding,
)


def _words(texts: list[str], start: float, step: float = 0.5) -> list[Word]:
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def main() -> int:
    cloner = ScriptedVoiceCloner()
    asha = cloner.clone("Asha", [make_sine(220.0, 1.0)])
    ravi = cloner.clone("Ravi", [make_sine(110.0, 1.0)])
    library = VoiceLibrary()
    library.add(asha)
    library.add(ravi)
    print(f"library: {', '.join(library.names())} ({len(library)} voices)")
    print(
        "fresh 220 Hz sample matches: "
        f"{library.match(extract_embedding(make_sine(220.0, 1.0)))} "
        f"(similarity {cosine_similarity(extract_embedding(make_sine(220.0, 1.0)), asha.embedding):.3f})"
    )

    transcript = Transcript(
        segments=[
            Segment(words=_words(["नमस्ते", "दोस्तों"], 0.0, step=0.5), speaker="Asha"),
            Segment(words=_words(["नमस्ते", "आशा"], 2.0, step=0.45), speaker="Ravi"),
            Segment(
                words=_words(["चलो", "शुरू", "करते", "हैं"], 4.0, step=0.4), speaker="Asha"
            ),
        ],
        language="hi",
    )
    narrator = cloner.clone("Narrator", [make_sine(150.0, 1.0)])
    audio, fit_factors, placed = dub_transcript(
        transcript,
        cloner,
        narrator,
        profiles={"Asha": asha, "Ravi": ravi},
    )
    report = build_report(transcript, placed, fit_factors, audio)
    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-multispeaker-"))
    save_wav(str(out_dir / "dub.wav"), audio)
    (out_dir / "dub.srt").write_text(transcript.to_srt(), encoding="utf-8")
    (out_dir / "dub.vtt").write_text(transcript.to_vtt(), encoding="utf-8")
    print(f"dubbed {report.segments_dubbed} segments, {report.speakers_dubbed} speakers")
    print(
        f"sync drift mean {report.mean_sync_drift:.3f}s / max {report.max_sync_drift:.3f}s, "
        f"mean fit {report.mean_fit_factor:.2f}, verdict: {report.verdict}"
    )
    print(f"outputs in {out_dir} (dub.wav, dub.srt, dub.vtt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
