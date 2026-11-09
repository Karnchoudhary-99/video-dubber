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
