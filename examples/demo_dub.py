"""End-to-end demo of the full dubbing pipeline.

Transcribe (scripted) -> translate to Hindi -> clone a voice from a
reference tone -> render the dubbed track -> quality report. Prints the
report and saves the dubbed WAV next to a copy of the translated
transcript.

No model, no network, no ffmpeg needed for the audio path — run it with
``python examples/demo_dub.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.audio import make_sine, save_wav
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    source = Transcript(
        segments=[
            Segment(
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("and", 0.4, 0.6),
                    Word("welcome", 0.6, 1.1),
                    Word("to", 1.1, 1.3),
                    Word("Mars", 1.3, 1.7),
                    Word("Cricket", 1.7, 2.2),
                ],
                speaker="commentator",
            ),
            Segment(
                words=[
                    Word("What", 2.5, 2.8),
                    Word("a", 2.8, 2.9),
                    Word("goal", 2.9, 3.3),
                    Word("by", 3.3, 3.5),
                    Word("the", 3.5, 3.6),
                    Word("champion", 3.6, 4.1),
                ],
                speaker="commentator",
            ),
        ],
        language="en",
    )
    translator = DictionaryTranslator(source="en", target="hi")
    translated = translate_transcript(source, translator, "hi")
    print(f"translated ({translator.name}): {translated.text!r}")
    print(f"dictionary coverage: {translator.last_coverage:.0%}")

    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Commentator", [make_sine(180.0, 1.0)])
    print(f"cloned voice: {profile.name}, base pitch {profile.base_pitch_hz:.0f} Hz")

    dub_audio, fit_factors, placed = dub_transcript(translated, cloner, profile)
    report = build_report(
        translated,
        placed,
        fit_factors,
        dub_audio,
        translation_coverage=translator.last_coverage,
    )
    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-dub-"))
    wav_path = out_dir / "dubbed.wav"
    save_wav(str(wav_path), dub_audio)
    report_path = out_dir / "report.md"
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"\ndubbed track: {dub_audio.duration:.2f}s -> {wav_path}")
    print(f"fit factors: {[round(f, 2) for f in fit_factors]}")
    print(f"\n{report.to_markdown()}")
    print(f"report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
