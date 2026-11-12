"""Demo: pronounce names and acronyms the way the lexicon says.

A synthesiser guesses at proper nouns — "NASA" comes out letter by
letter, "Karn" with the wrong vowel. The demo dubs one two-line
transcript plain and with a small lexicon (NASA -> "N A S A",
Karn -> "kar-n", New Delhi -> "nyoo delhi") and prints the fit
factors: the lexicon's longer respellings need a stronger stretch to
fit the same windows, and the quality report records exactly how many
pronunciations were applied.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_lexicon.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.lexicon import PronunciationLexicon, apply_lexicon
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import EMBEDDING_DIM, ScriptedVoiceCloner, SpeakerProfile


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def main() -> int:
    lexicon = PronunciationLexicon(
        entries={
            "NASA": "N A S A",
            "Karn": "kar-n",
            "New Delhi": "nyoo delhi",
        }
    )
    print(f"{len(lexicon)} pronunciation(s) in the lexicon")
    applied = apply_lexicon("Karn reports from New Delhi for NASA", lexicon)
    print(f"  {applied.text}")
    for matched, spoken in applied.replacements:
        print(f"    {matched} -> {spoken}")
    assert applied.applied_count == 3

    transcript = Transcript(
        segments=[
            _seg("Karn reports from New Delhi", 0.0, 2.5),
            _seg("for NASA tonight", 2.5, 4.5),
        ],
        language="hi",
    )
    profile = SpeakerProfile(name="Demo", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)
    cloner = ScriptedVoiceCloner()

    plain, plain_fit, plain_placed = dub_transcript(transcript, cloner, profile)
    plain_report = build_report(transcript, plain_placed, plain_fit, plain)
    dubbed, lex_fit, lex_placed = dub_transcript(transcript, cloner, profile, lexicon=lexicon)
    applied_count = sum(
        apply_lexicon(segment.text, lexicon).applied_count for segment in transcript.segments
    )
    lex_report = build_report(
        transcript, lex_placed, lex_fit, dubbed, lexicon_applied=applied_count
    )
    print(f"plain: mean fit {plain_report.mean_fit_factor:.2f}, verdict {plain_report.verdict}")
    print(
        f"lexicon: mean fit {lex_report.mean_fit_factor:.2f}, "
        f"verdict {lex_report.verdict}, "
        f"pronunciations applied {lex_report.lexicon_applied}"
    )
    assert lex_report.lexicon_applied == 3
    # Longer respellings need a stronger stretch to fill the same windows.
    assert lex_report.mean_fit_factor < plain_report.mean_fit_factor
    print("demo ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
