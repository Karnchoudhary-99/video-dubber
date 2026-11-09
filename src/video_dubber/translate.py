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
import unicodedata
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


def _is_word_char(ch: str) -> bool:
    """A character that belongs inside a word: letter, mark, number, ``_``.

    Unicode category M (combining marks) matters here: Devanagari vowel
    signs such as ``े`` are marks, not letters, so ``\\w`` shatters
    "नमस्ते" into consonant fragments and ``\\b`` never matches around
    it. Grouping marks with their base letter keeps every script's
    words whole — the old ``_TOKEN_RE`` broke Hindi back-translation
    completely (caught by the fidelity tests, 2026-11-09).
    """
    return ch == "_" or unicodedata.category(ch)[0] in "LMN"


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split text into ``(token, start, end)`` triples.

    Tokens are runs of word characters (with one internal apostrophe
    kept, e.g. "let's"), runs of whitespace, or runs of
    punctuation/symbols. The triples cover the whole string with no
    gaps, so spans can be mapped back onto the original text.
    """
    tokens: list[tuple[str, int, int]] = []
    index = 0
    size = len(text)
    while index < size:
        ch = text[index]
        if ch.isspace():
            end = index + 1
            while end < size and text[end].isspace():
                end += 1
        elif _is_word_char(ch):
            end = index + 1
            while end < size:
                nxt = text[end]
                if _is_word_char(nxt):
                    end += 1
                elif nxt == "'" and end + 1 < size and _is_word_char(text[end + 1]):
                    end += 2
                    while end < size and _is_word_char(text[end]):
                        end += 1
                else:
                    break
        else:
            end = index + 1
            while end < size and not text[end].isspace() and not _is_word_char(text[end]):
                end += 1
        tokens.append((text[index:end], index, end))
        index = end
    return tokens


def _find_phrase_spans(
    text: str,
    tokens: list[tuple[str, int, int]],
    phrases: list[str],
    dictionary: dict[str, str],
) -> list[tuple[int, int, str]]:
    """Find non-overlapping phrase occurrences, longest phrase first.

    A phrase matches a run of consecutive word tokens equal to the
    phrase's words (case-insensitive), separated only by whitespace.
    Matching on the token stream instead of ``\\b`` regexes is what
    makes phrases work in scripts whose words end in combining marks.
    Returns ``(start, end, replacement)`` spans into ``text``.
    """
    words = [
        (token, start, end) for token, start, end in tokens if any(ch.isalnum() for ch in token)
    ]
    claimed: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []
    for phrase in phrases:
        wanted = [word.casefold() for word in phrase.split()]
        width = len(wanted)
        for at in range(len(words) - width + 1):
            run = words[at : at + width]
            if [token.casefold() for token, _, _ in run] != wanted:
                continue
            start, end = run[0][1], run[-1][2]
            # Only whitespace may sit between the phrase's words.
            if any(not text[run[i][2] : run[i + 1][1]].isspace() for i in range(width - 1)):
                continue
            if any(start < taken_end and taken_start < end for taken_start, taken_end in claimed):
                continue
            claimed.append((start, end))
            spans.append((start, end, dictionary[phrase]))
    return spans


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
        # Coverage counting skips tokens with no ASCII letters only when
        # the source language itself is ASCII-scripted: there they are
        # already-translated target text (e.g. a glossary term applied
        # before this call). When the source script is non-Latin (a
        # reversed hi -> en table), those tokens ARE the source words
        # and must count, known or unknown.
        self._source_is_ascii = bool(self.dictionary) and all(
            any("a" <= ch.lower() <= "z" for ch in key) for key in self.dictionary
        )
        self._total_known = 0
        self._total_words = 0
        self._all_unknown: list[str] = []

    @property
    def name(self) -> str:
        return f"dictionary:{self.source}->{self.target}"

    def reset_stats(self) -> None:
        """Clear the running coverage totals (called by translate_transcript)."""
        self._total_known = 0
        self._total_words = 0
        self._all_unknown = []
        self.last_coverage = 0.0
        self.last_unknown = []

    def reversed(self) -> DictionaryTranslator:
        """Return a translator for the opposite direction (target -> source).

        The table is inverted, so a round trip can be scored. The
        inversion is lossy where two source words share one target word
        (``tomorrow``/``yesterday`` both become ``कल``): the later entry
        wins, and the other word scores low on the round trip — which is
        exactly what the fidelity check is meant to surface.
        """
        inverted = {target: source for source, target in self.dictionary.items()}
        return DictionaryTranslator(source=self.target, target=self.source, dictionary=inverted)

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
        tokens = _tokenize(text)
        phrase_spans = _find_phrase_spans(text, tokens, self._phrases, self.dictionary)
        known = 0
        total = 0
        unknown: list[str] = []
        for token, start, end in tokens:
            if token.isspace() or not any(ch.isalnum() for ch in token):
                continue
            # Tokens with no ASCII letters are already target-language text
            # (e.g. a glossary term applied before this call) or numbers —
            # they are not untranslated source words and do not count.
            # (Skipped only for ASCII-scripted sources; see __post_init__.)
            if self._source_is_ascii and not any("a" <= ch.lower() <= "z" for ch in token):
                continue
            total += 1
            covered = any(s <= start and end <= e for s, e, _ in phrase_spans)
            if covered or token.casefold() in self.dictionary:
                known += 1
            else:
                unknown.append(token)
        self.last_unknown = unknown
        self.last_coverage = known / total if total else 1.0
        # Running totals: last_coverage/last_unknown describe everything
        # translated since the last reset_stats(), so a whole transcript
        # reports one honest number instead of only its last segment.
        self._total_known += known
        self._total_words += total
        self._all_unknown.extend(unknown)
        if self._total_words:
            self.last_coverage = self._total_known / self._total_words
        self.last_unknown = list(self._all_unknown)
        # 1. Phrase substitution (longest first, word boundaries via the
        # token stream). Already-translated phrases are not in the source
        # dictionary, so step 2 leaves them untouched. Spans are applied
        # right-to-left so earlier offsets stay valid.
        work = text
        for start, end, replacement in sorted(phrase_spans, reverse=True):
            work = work[:start] + replacement + work[end:]
        # 2. Word-by-word lookup with case preservation.
        out: list[str] = []
        for token, _, _ in _tokenize(work):
            if token.isspace() or not any(ch.isalnum() for ch in token):
                out.append(token)
                continue
            replacement = self.dictionary.get(token.casefold())
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
class ArgosTranslator:
    """Translation via the ``argos-translate`` CLI (LibreTranslate models).

    Mirrors :class:`~video_dubber.transcribe.WhisperCliTranscriber`: the
    heavy model lives in an external tool, this class is just the adapter.
    Requires ``argos-translate`` on PATH with the ``source -> target``
    language package installed (``argostranslate-cli``). Each call is one
    subprocess invocation, so callers translating a whole transcript
    should prefer :func:`translate_transcript` with a dictionary backend
    for demos and reserve Argos for the final pass.
    """

    timeout: float = 120.0

    @property
    def name(self) -> str:
        return "argos-translate"

    def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return text
        try:
            result = subprocess.run(
                ["argos-translate", "--from-lang", source, "--to-lang", target, text],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise TranslationError(
                "argos-translate CLI not found on PATH; install it "
                "(pip install argostranslate) and the language package, "
                "or use --backend dictionary"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TranslationError(f"argos-translate timed out after {self.timeout}s") from exc
        if result.returncode != 0:
            raise TranslationError(
                f"argos-translate failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
            )
        translated = result.stdout.strip()
        if not translated:
            raise TranslationError("argos-translate returned empty output")
        return translated


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
    all_applied: list[str] = field(default_factory=list, init=False)

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
        # Cumulative across calls, so a whole-transcript run can report
        # every pinned term (translate_transcript calls apply per segment).
        for term in applied:
            if term not in self.all_applied:
                self.all_applied.append(term)
        return work

    def reset(self) -> None:
        """Clear the applied-term reports (called by translate_transcript)."""
        self.last_applied = []
        self.all_applied = []


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
    max_chars: int | None = None,
) -> Transcript:
    """Translate a whole transcript, preserving segment timing windows.

    Each translated segment spans exactly the same ``[start, end]`` window
    as its source segment; translated words share that window in
    proportion to their character length (longer words get more time).
    Speaker labels survive; the result's ``language`` is the target and
    ``source`` records the backend that produced it. Empty segments stay
    empty. With ``max_chars`` set, long translated segments are split into
    subtitle-sized cues via :func:`chunk_segment`. Raises
    :class:`TranslationError` if the source language is unknown and no
    ``source`` was given.
    """
    src = source or transcript.language
    if not src:
        raise TranslationError(
            "source language unknown: pass --source or use a transcript with a language set"
        )
    # Fresh per-run stats: a translator/glossary reused across transcripts
    # must not carry the previous run's coverage or pinned terms.
    reset = getattr(translator, "reset_stats", None)
    if callable(reset):
        reset()
    if glossary is not None:
        glossary.reset()
    segments: list[Segment] = []
    for segment in transcript.segments:
        text = translate_text(segment.text, translator, src, target, glossary)
        words = _retime(segment, text.split(), segment.mean_confidence)
        translated = Segment(words=words, speaker=segment.speaker)
        if max_chars is not None:
            segments.extend(chunk_segment(translated, max_chars))
        else:
            segments.append(translated)
    return Transcript(segments=segments, language=target, source=translator.name)


def _retime(segment: Segment, words: list[str], confidence: float) -> list[Word]:
    """Spread ``words`` across the segment window, proportional to length.

    A word's share is ``len(word) + 1`` (the trailing space it would have
    had in speech); the last word ends exactly at the segment end so the
    total duration is preserved bit-for-bit.
    """
    if not words or segment.duration <= 0:
        return [
            Word(text=w, start=segment.start, end=segment.end, confidence=confidence) for w in words
        ]
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


def chunk_segment(segment: Segment, max_chars: int = 84) -> list[Segment]:
    """Split a segment into subtitle-sized cues (<= ``max_chars`` chars).

    Dubbed text is often longer than the source (Hindi runs ~15% longer
    than English); a single long cue is unreadable on screen, so cues are
    cut at word boundaries, preferring breaks after sentence punctuation,
    and each chunk keeps a proportional slice of the original timing
    window. A segment that already fits is returned unchanged. Raises
    :class:`TranslationError` on a non-positive ``max_chars``.
    """
    if max_chars <= 0:
        raise TranslationError(f"max_chars must be positive, got {max_chars}")
    if not segment.words or len(segment.text) <= max_chars:
        return [segment]
    words = segment.words
    chunks: list[list[Word]] = []
    current: list[Word] = []
    current_len = 0
    for word in words:
        add = len(word.text) + (1 if current else 0)
        if current and current_len + add > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word.text) + (1 if len(current) > 1 else 0)
        # Prefer a break right after sentence punctuation when the cue is
        # already reasonably full (avoids mid-sentence cuts).
        if word.text.endswith((".", "!", "?", "।")) and current_len >= max_chars // 2:
            chunks.append(current)
            current = []
            current_len = 0
    if current:
        chunks.append(current)
    out: list[Segment] = []
    for chunk in chunks:
        out.append(Segment(words=list(chunk), speaker=segment.speaker))
    return out
