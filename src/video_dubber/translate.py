"""Translation pipeline: source-language transcript -> target-language transcript.

The pipeline is backend-agnostic: a :class:`Translator` turns one string
into another, and :func:`translate_transcript` lifts that over a whole
:class:`~video_dubber.transcript.Transcript` while preserving the segment
timings the dub needs later (translated words inherit timings
proportionally, so a translated segment still spans exactly the same
window as the source segment).

Backends shipped here:

* :class:`DictionaryTranslator` — a small built-in phrase/word dictionary
  (en -> hi, en -> es). Offline, deterministic, and honest about its
  limits: unknown words pass through untranslated and are counted, so the
  caller can see coverage instead of guessing.
* :class:`ArgosTranslator` — shells out to the ``argos-translate`` CLI when
  installed (LibreTranslate/Argos models), for real translation quality.
* A :class:`Glossary` applies caller-supplied term overrides (brand names,
  character names, domain terms) before the dictionary runs.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from video_dubber.transcript import Segment, Transcript, Word

__all__ = [
    "ArgosTranslator",
    "DictionaryTranslator",
    "Glossary",
    "TranslationError",
    "Translator",
    "chunk_segment",
    "translate_text",
    "translate_transcript",
]


class TranslationError(Exception):
    """Raised when a translation backend fails or the input is unusable."""


class Translator(Protocol):
    """Translates one string from ``source`` to ``target`` language.

    Language codes are BCP-47-ish short codes (``en``, ``hi``, ``es``).
    Implementations raise :class:`TranslationError` on failure and must
    never return ``None``.
    """

    def translate(self, text: str, source: str, target: str) -> str: ...

    @property
    def name(self) -> str: ...


# --- Built-in demo dictionaries (en -> hi / en -> es) -----------------------
# These cover the everyday words a demo dub needs. They are a baseline, not
# a translation engine: anything missing passes through unchanged and is
# reported in the coverage stats, so a demo never silently invents text.

_EN_HI: dict[str, str] = {
    "hello": "नमस्ते",
    "welcome": "स्वागत है",
    "good": "अच्छा",
    "morning": "सुबह",
    "evening": "शाम",
    "night": "रात",
    "day": "दिन",
    "today": "आज",
    "tomorrow": "कल",
    "yesterday": "कल",
    "thank": "धन्यवाद",
    "you": "आप",
    "please": "कृपया",
    "yes": "हाँ",
    "no": "नहीं",
    "friend": "दोस्त",
    "friends": "दोस्तों",
    "world": "दुनिया",
    "mars": "मंगल",
    "game": "खेल",
    "match": "मैच",
    "win": "जीत",
    "wins": "जीतता है",
    "team": "टीम",
    "player": "खिलाड़ी",
    "goal": "गोल",
    "score": "स्कोर",
    "champion": "चैंपियन",
    "championship": "चैंपियनशिप",
    "final": "फाइनल",
    "amazing": "शानदार",
    "great": "बढ़िया",
    "fantastic": "शानदार",
    "watch": "देखो",
    "watching": "देख रहे हैं",
    "welcome to": "में आपका स्वागत है",
    "good morning": "सुप्रभात",
    "good evening": "शुभ संध्या",
    "thank you": "धन्यवाद",
    "see you": "फिर मिलेंगे",
    "let's go": "चलो",
    "what a": "क्या",
    "this is": "यह है",
    "we are": "हम हैं",
}

_EN_ES: dict[str, str] = {
    "hello": "hola",
    "welcome": "bienvenido",
    "good": "bueno",
    "morning": "mañana",
    "evening": "tarde",
    "night": "noche",
    "day": "día",
    "today": "hoy",
    "tomorrow": "mañana",
    "thank": "gracias",
    "you": "tú",
    "please": "por favor",
    "yes": "sí",
    "no": "no",
    "friend": "amigo",
    "friends": "amigos",
    "world": "mundo",
    "mars": "marte",
    "game": "juego",
    "match": "partido",
    "win": "ganar",
    "wins": "gana",
    "team": "equipo",
    "player": "jugador",
    "goal": "gol",
    "score": "marcador",
    "champion": "campeón",
    "championship": "campeonato",
    "final": "final",
    "amazing": "increíble",
    "great": "genial",
    "fantastic": "fantástico",
    "watch": "mira",
    "welcome to": "bienvenido a",
    "good morning": "buenos días",
    "good evening": "buenas tardes",
    "thank you": "gracias",
    "see you": "hasta luego",
    "let's go": "vamos",
    "this is": "esto es",
    "we are": "somos",
}

_BUILTIN: dict[tuple[str, str], dict[str, str]] = {
    ("en", "hi"): _EN_HI,
    ("en", "es"): _EN_ES,
}

_TOKEN_RE = re.compile(r"(\w+(?:'\w+)?|[^\w\s]+|\s+)", re.UNICODE)


@dataclass
class DictionaryTranslator:
    """Offline word/phrase dictionary translator.

    Phrases (multi-word keys) are matched first, longest first, with word
    boundaries; single words next; anything unknown passes through verbatim
    and is counted in :attr:`last_unknown` / :attr:`last_coverage`. Case of
    the source word is preserved for title-case and upper-case words.
    """

    source: str = "en"
    target: str = "hi"
    dictionary: dict[str, str] | None = None
    last_coverage: float = field(default=0.0, init=False)
    last_unknown: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.dictionary is None:
            self.dictionary = dict(_BUILTIN.get((self.source, self.target), {}))
        # Longest phrase first so "good morning" wins over "good".
        self._phrases = sorted(
            (k for k in self.dictionary if " " in k), key=lambda k: -len(k.split())
        )

    @property
    def name(self) -> str:
        return f"dictionary:{self.source}->{self.target}"

    def supported(self, source: str, target: str) -> bool:
        return (source, target) in _BUILTIN or (
            self.source == source and self.target == target and bool(self.dictionary)
        )

    def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return text
        if source != self.source or target != self.target:
            raise TranslationError(
                f"dictionary backend has no {source}->{target} table "
                f"(loaded {self.source}->{self.target}); use --backend argos "
                "or a custom dictionary"
            )
        return self._translate(text)

    def _translate(self, text: str) -> str:
        # Coverage is measured against the ORIGINAL words: a word counts as
        # known if it sits inside a matched phrase or in the dictionary.
        # (Measuring after substitution would count translated target words
        # as unknown, since they are not in the source dictionary.)
        phrase_spans: list[tuple[int, int]] = []
        for phrase in self._phrases:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE):
                phrase_spans.append(match.span())
        known = 0
        total = 0
        unknown: list[str] = []
        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if token.isspace() or not any(ch.isalnum() for ch in token):
                continue
            total += 1
            span = match.span()
            covered = any(s <= span[0] and span[1] <= e for s, e in phrase_spans)
            if covered or token.lower() in self.dictionary:
                known += 1
            else:
                unknown.append(token)
        self.last_unknown = unknown
        self.last_coverage = known / total if total else 1.0
        # 1. Phrase substitution (longest first, case-insensitive, word
        #    boundaries). Already-translated phrases are not in the source
        #    dictionary, so step 2 leaves them untouched.
        work = text
        for phrase in self._phrases:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            work = pattern.sub(self.dictionary[phrase], work)
        # 2. Word-by-word lookup with case preservation.
        out: list[str] = []
        for token in _TOKEN_RE.findall(work):
            if token.isspace() or not any(ch.isalnum() for ch in token):
                out.append(token)
                continue
            replacement = self.dictionary.get(token.lower())
            out.append(_match_case(token, replacement) if replacement else token)
        return re.sub(r" {2,}", " ", "".join(out)).strip()


def _match_case(source_token: str, replacement: str) -> str:
    """Copy the source word's capitalisation onto the replacement."""
    if source_token.isupper():
        return replacement.upper()
    if source_token[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


@dataclass
class Glossary:
    """Caller-supplied term overrides, applied before any backend runs.

    Brand names, character names and domain terms must survive translation
    verbatim or with a fixed rendering — e.g. ``{"Mars Rovers": "मार्स
    रोवर्स"}`` or ``{"NASA": "NASA"}`` (keep the English spelling). Terms
    are matched case-insensitively with word boundaries, longest first,
    and the applied overrides are reported so the dub report can show
    exactly which terms were pinned.
    """

    terms: dict[str, str] = field(default_factory=dict)
    last_applied: list[str] = field(default_factory=list, init=False)

    @classmethod
    def from_json(cls, data: str) -> Glossary:
        """Parse ``{"source term": "target term", ...}`` JSON."""
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as exc:
            raise TranslationError(f"invalid glossary JSON: {exc}") from exc
        if not isinstance(raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw.items()
        ):
            raise TranslationError("glossary JSON must be an object of string->string")
        return cls(terms=dict(raw))

    @classmethod
    def from_json_file(cls, path: str) -> Glossary:
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    def apply(self, text: str) -> str:
        """Replace glossary terms in ``text``; records what was applied."""
        applied: list[str] = []
        work = text
        for term in sorted(self.terms, key=lambda t: -len(t.split())):
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(work):
                applied.append(term)
                work = pattern.sub(self.terms[term], work)
        self.last_applied = applied
        return work


def translate_text(
    text: str,
    translator: Translator,
    source: str,
    target: str,
    glossary: Glossary | None = None,
) -> str:
    """Translate one string, applying the glossary first if given."""
    if glossary is not None:
        text = glossary.apply(text)
    return translator.translate(text, source, target)


def translate_transcript(
    transcript: Transcript,
    translator: Translator,
    target: str,
    source: str | None = None,
    glossary: Glossary | None = None,
) -> Transcript:
    """Translate a whole transcript, preserving segment timing windows.

    Each translated segment spans exactly the same ``[start, end]`` window
    as its source segment; translated words share that window in
    proportion to their character length (longer words get more time).
    Speaker labels survive; the result's ``language`` is the target and
    ``source`` records the backend that produced it. Empty segments stay
    empty. Raises :class:`TranslationError` if the source language is
    unknown and no ``source`` was given.
    """
    src = source or transcript.language
    if not src:
        raise TranslationError(
            "source language unknown: pass --source or use a transcript "
            "with a language set"
        )
    segments: list[Segment] = []
    for segment in transcript.segments:
        text = translate_text(segment.text, translator, src, target, glossary)
        words = _retime(segment, text.split(), segment.mean_confidence)
        segments.append(Segment(words=words, speaker=segment.speaker))
    return Transcript(segments=segments, language=target, source=translator.name)


def _retime(segment: Segment, words: list[str], confidence: float) -> list[Word]:
    """Spread ``words`` across the segment window, proportional to length.

    A word's share is ``len(word) + 1`` (the trailing space it would have
    had in speech); the last word ends exactly at the segment end so the
    total duration is preserved bit-for-bit.
    """
    if not words or segment.duration <= 0:
        return [Word(text=w, start=segment.start, end=segment.end, confidence=confidence) for w in words]
    weights = [len(w) + 1 for w in words]
    total = sum(weights)
    out: list[Word] = []
    cursor = segment.start
    for index, (word, weight) in enumerate(zip(words, weights)):
        share = segment.duration * weight / total
        end = segment.end if index == len(words) - 1 else cursor + share
        out.append(Word(text=word, start=cursor, end=end, confidence=confidence))
        cursor = end
    return out
