"""Demo: keep the original music bed under the dub.

Builds a synthetic original — a low music tone running the whole clip,
with speech-like tones over the two spoken segments — then dubs it
twice: once plain (silence between lines) and once with the music bed
kept at level 0.3. Prints both quality reports and shows the bed is
audible in the gap in the second dub and silent in the first.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_bed.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.audio import AudioData, make_sine, save_wav
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript
from video_dubber.voice import ScriptedVoiceCloner


def _make_original() -> AudioData:
    """Music tone throughout + speech-like tones in the spoken windows."""
    music = make_sine(113.7, 4.1, amplitude=0.15)
    speech = make_sine(440.0, 4.1, amplitude=0.4)
    samples = list(music.samples)
    for i, sample in enumerate(speech.samples):
        t = i / speech.sample_rate
        if t < 2.2 or t >= 2.5:  # only inside the two spoken windows
            samples[i] += sample
    return AudioData([max(-1.0, min(1.0, s)) for s in samples], music.sample_rate)


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
    original = _make_original()
    translator = DictionaryTranslator(source="en", target="hi")
    translated = translate_transcript(source, translator, "hi")
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Commentator", [make_sine(180.0, 1.0)])

    out_dir = Path(tempfile.mkdtemp(prefix="video-dubber-bed-"))
    gap = int(2.35 * 16_000)  # between the two segments: music only
    for label, bed_level in (("plain", None), ("with-bed", 0.3)):
        dub_audio, fit_factors, placed = dub_transcript(
            translated, cloner, profile, original, bed_level=bed_level
        )
        report = build_report(
            translated,
            placed,
            fit_factors,
            dub_audio,
            original,
            translation_coverage=translator.last_coverage,
            bed_level=bed_level,
        )
        wav_path = out_dir / f"dubbed-{label}.wav"
        save_wav(str(wav_path), dub_audio)
        gap_level = abs(dub_audio.samples[gap])
        bed_there = abs(original.samples[gap])
        print(f"\n[{label}] dub -> {wav_path}")
        print(f"  gap (2.35s) level: {gap_level:.3f} (original bed there: {bed_there:.3f})")
        print("  " + report.to_markdown().replace("\n", "\n  "))
        assert bed_there > 0.05, "demo music tone is silent at the probe point"
        if bed_level is None:
            assert gap_level < 0.01, "plain dub should be silent in the gap"
        else:
            assert abs(gap_level - bed_there * 0.3) < 0.01, (
                "bedded dub should keep the music at 0.3 in the gap"
            )
    print(f"\noutputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
