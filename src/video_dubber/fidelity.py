"""Translation fidelity: round-trip (back-translation) stability check.

A dub is only as good as its translation, and a bad translation is
expensive to discover *after* the voice has been cloned and the dub
rendered. This module catches the cheap failures early: translate a
segment to the target language, translate it back to the source, and
compare the round trip with the original. A segment that survives the
round trip intact is *stable*; one that comes back mangled is flagged
for a human look before dubbing starts.

Honest limits, stated up front:

* A perfect round-trip score means the translation is **stable**, not
  that it is **correct**. A wrong-but-consistent mapping (or a word the
  dictionary does not know, which passes through untouched both ways)
  scores 1.0. Fidelity is a smoke alarm, not a proofreader.
* The built-in dictionary is lossy in reverse: two source words can
  share one target word (``tomorrow`` and ``yesterday`` both become
  ``कल``), so the back-translation can only recover one of them. Such
  collisions score low and are flagged — that is the check working,
  not a bug in the check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from video_dubber.transcript import Transcript
from video_dubber.translate import ArgosTranslator, DictionaryTranslator, Translator

__all__ = [
    "FidelityError",
    "FidelityReport",
    "SegmentFidelity",
    "check_fidelity",
    "reverse_translator",
    "round_trip_text",
    "word_jaccard",
]

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class FidelityError(Exception):
    """Raised when a fidelity check cannot run (e.g. no reverse backend)."""


def word_jaccard(first: str, second: str) -> float:
    """Word-level Jaccard similarity between two strings (0.0-1.0).

    Tokens are lower-cased words; two empty strings score 1.0 (nothing
    to lose), one empty and one non-empty scores 0.0.
    """
    first_words = set(_WORD_RE.findall(first.lower()))
    second_words = set(_WORD_RE.findall(second.lower()))
    if not first_words and not second_words:
        return 1.0
    if not first_words or not second_words:
        return 0.0
    return len(first_words & second_words) / len(first_words | second_words)


def reverse_translator(translator: Translator) -> Translator:
    """Return a translator for the opposite direction of ``translator``.

    The dictionary backend inverts its table (``hi -> en`` from the
    ``en -> hi`` table); Argos just needs a fresh adapter, since it
    takes the languages per call. Anything else cannot be reversed
    safely and raises :class:`FidelityError`.
    """
    if isinstance(translator, DictionaryTranslator):
        return translator.reversed()
    if isinstance(translator, ArgosTranslator):
        return ArgosTranslator(timeout=translator.timeout)
    raise FidelityError(
        f"cannot reverse translator {translator.name!r}: "
        "fidelity checks need a dictionary or argos backend"
    )


def round_trip_text(
    text: str, translator: Translator, source: str, target: str
) -> tuple[str, str, float]:
    """Translate ``text`` to ``target`` and back to ``source``.

    Returns ``(translated, back_translated, score)`` where ``score`` is
    the word-level Jaccard similarity between the original and the
    back-translation. Raises :class:`FidelityError` if the translator
    cannot be reversed.
    """
    if not text.strip():
        return text, text, 1.0
    translated = translator.translate(text, source, target)
    back = reverse_translator(translator)
    back_translated = back.translate(translated, target, source)
    return translated, back_translated, word_jaccard(text, back_translated)


@dataclass
class SegmentFidelity:
    """Round-trip result for one transcript segment."""

    index: int
    source_text: str
    translated_text: str
    back_text: str
    score: float
    flagged: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "back_text": self.back_text,
            "score": round(self.score, 4),
            "flagged": self.flagged,
        }


@dataclass
class FidelityReport:
    """Round-trip fidelity for a whole transcript.

    ``verdict`` is ``good`` when every segment clears the threshold,
    ``review`` when any segment is flagged. The report names the
    flagged segments so a human knows exactly where to look.
    """

    source: str
    target: str
    backend: str
    threshold: float
    segments: list[SegmentFidelity] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if not self.segments:
            return 1.0
        return sum(s.score for s in self.segments) / len(self.segments)

    @property
    def min_score(self) -> float:
        if not self.segments:
            return 1.0
        return min(s.score for s in self.segments)

    @property
    def flagged(self) -> list[SegmentFidelity]:
        return [s for s in self.segments if s.flagged]

    @property
    def verdict(self) -> str:
        return "review" if self.flagged else "good"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "backend": self.backend,
            "threshold": self.threshold,
            "verdict": self.verdict,
            "mean_score": round(self.mean_score, 4),
            "min_score": round(self.min_score, 4),
            "flagged_count": len(self.flagged),
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            "# Translation fidelity report",
            "",
            f"- Direction: {self.source} -> {self.target} (via {self.backend})",
            f"- Verdict: **{self.verdict}** "
            f"(mean {self.mean_score:.2f}, min {self.min_score:.2f}, "
            f"threshold {self.threshold:.2f})",
            f"- Flagged segments: {len(self.flagged)} of {len(self.segments)}",
            "",
        ]
        if self.flagged:
            lines.append("## Needs review")
            lines.append("")
            for segment in self.flagged:
                lines.append(
                    f"- Segment {segment.index} (score {segment.score:.2f}): "
                    f"“{segment.source_text}” -> “{segment.translated_text}” "
                    f"-> back “{segment.back_text}”"
                )
            lines.append("")
        lines.append(
            "_A perfect score means the translation is stable, not that it is "
            "correct — unknown words pass through untouched both ways._"
        )
        return "\n".join(lines) + "\n"
