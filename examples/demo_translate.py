"""End-to-end demo of the translation pipeline.

Takes a scripted English transcript (the Mars Sports League commentary
style the project demos with), translates it to Hindi with the built-in
dictionary backend, applies a glossary pin for the team name, and prints
the translated subtitles alongside the coverage report.

No model and no network needed — run it with ``python examples/demo_translate.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import (
    DictionaryTranslator,
    Glossary,
    translate_transcript,
)


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
    glossary = Glossary(terms={"Mars Cricket": "मार्स क्रिकेट"})
    translated = translate_transcript(source, translator, "hi", glossary=glossary, max_chars=42)
    print(f"source: {source.text!r}")
    print(f"translated ({translator.name}): {translated.text!r}")
    print(f"dictionary coverage: {translator.last_coverage:.0%}")
    if translator.last_unknown:
        print(f"untranslated words: {', '.join(translator.last_unknown)}")
    print(f"glossary applied: {', '.join(glossary.all_applied) or 'none'}")
    print(f"segments: {len(source.segments)} -> {len(translated.segments)} cues")
    print("\nSRT output:\n")
    print(translated.to_srt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
