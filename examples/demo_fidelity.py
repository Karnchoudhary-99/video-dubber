"""Demo: translation fidelity via round-trip (back-translation) check.

Translates a short English transcript to Hindi, back-translates each
segment, and reports the fidelity score. The first segment round-trips
cleanly; the second ("tomorrow") comes back as "yesterday" — both words
share the Hindi word "कल", so the dictionary cannot tell them apart —
and the report flags it for a human look. That flag is the check
working, not a failure: it names the exact segment to fix before
dubbing starts.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_fidelity.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.fidelity import check_fidelity
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript


def _words(texts, start=0.0, step=0.5):
    return [
        Word(text=t, start=start + i * step, end=start + (i + 1) * step)
        for i, t in enumerate(texts)
    ]


def main() -> int:
    source = Transcript(
        segments=[
            Segment(words=_words(["Hello", "world", "good", "morning"], 0.0)),
            Segment(words=_words(["See", "you", "tomorrow"], 3.0)),
        ],
        language="en",
    )
    translator = DictionaryTranslator(source="en", target="hi")
    translated = translate_transcript(source, translator, target="hi")
    report = check_fidelity(source, translated, translator, target="hi")
    print(f"translated: {[s.text for s in translated.segments]}")
    print(report.to_markdown())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fidelity.md"
        out.write_text(report.to_markdown(), encoding="utf-8")
        assert "Segment 1" in out.read_text(encoding="utf-8")
    assert report.verdict == "review"
    assert report.flagged[0].source_text == "See you tomorrow"
    assert report.flagged[0].back_text == "see you yesterday"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
