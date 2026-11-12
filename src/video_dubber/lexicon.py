"""Pronunciation lexicon: teach the dubber how to say names, acronyms and loanwords.

A synthesiser mispronounces proper nouns, acronyms and loanwords: "Karn"
with the wrong vowel, "NASA" spelled out letter by letter when it
should be one word, "Bengaluru" flattened into something unrecognisable.
A lexicon maps the written form to the form the synthesiser should speak
— a respelling (``"Nguyen" -> "win"``), a spaced-out acronym
(``"NASA" -> "N A S A"``), or a simpler spelling (``"Bengaluru" ->
"bangalore"``). The demo double has no phoneme inventory, so the
pronunciation is itself plain text the synthesiser reads aloud; a real
TTS backend can interpret the same string as a respelling or a phoneme
hint, the way Whisper and Argos are adapters behind the same interfaces.

Matching is case-insensitive with real word boundaries, longest phrase
first (``"New Delhi"`` wins over ``"Delhi"``), and it reuses the
grapheme-aware tokenizer from :mod:`video_dubber.translate` — unlike the
glossary's ``\\b`` regex, this keeps Devanagari words whole, so a Hindi
entry such as ``"नमस्ते" -> "namaste"`` matches instead of shattering
at the combining marks. Case is preserved where it matters: an
all-caps match gets an all-caps pronunciation, a capitalised match gets
a capitalised one.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from video_dubber.translate import _is_word_char, _tokenize

__all__ = [
    "LexiconApplication",
    "LexiconError",
    "PronunciationEntry",
    "PronunciationLexicon",
    "apply_lexicon",
    "load_lexicon",
]


class LexiconError(Exception):
    """Raised when a lexicon file or entry is unusable."""


def _clean(text: str, what: str) -> str:
    """Strip and collapse whitespace; reject empties and control chars."""
    cleaned = " ".join(text.split())
    if not cleaned:
        raise LexiconError(f"lexicon {what} must not be empty")
    if any(unicodedata.category(ch) == "Cc" for ch in cleaned):
        raise LexiconError(f"lexicon {what} contains control characters: {cleaned!r}")
    return cleaned


def _normalise_key(word: str) -> str:
    """Lookup key: cleaned, lowercased — matching is case-insensitive."""
    return _clean(word, "word").lower()


@dataclass(frozen=True)
class PronunciationEntry:
    """One mapping: ``word`` as written, ``pronunciation`` as spoken."""

    word: str
    pronunciation: str

    def __post_init__(self) -> None:
        word = _clean(self.word, "word")
        if not any(ch.isalpha() for ch in word):
            raise LexiconError(f"lexicon word needs at least one letter: {word!r}")
        if any(not (_is_word_char(ch) or ch in " '-.") for ch in word):
            raise LexiconError(f"lexicon word has unsupported characters: {word!r}")
        pronunciation = _clean(self.pronunciation, "pronunciation")
        if not any(ch.isalpha() for ch in pronunciation):
            raise LexiconError(
                f"lexicon pronunciation needs at least one letter: {pronunciation!r}"
            )
        object.__setattr__(self, "word", word)
        object.__setattr__(self, "pronunciation", pronunciation)

    @property
    def phrase_words(self) -> int:
        """How many words the entry's key spans (1 for a single word)."""
        return len(self.word.split())


@dataclass
class PronunciationLexicon:
    """A validated word -> pronunciation map, keyed case-insensitively."""

    entries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalised: dict[str, str] = {}
        for word, pronunciation in self.entries.items():
            key = _normalise_key(word)
            if key in normalised:
                raise LexiconError(f"duplicate lexicon word (case-insensitive): {word!r}")
            entry = PronunciationEntry(word=word, pronunciation=pronunciation)
            normalised[key] = entry.pronunciation
        self.entries = normalised

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, word: str) -> bool:
        try:
            return _normalise_key(word) in self.entries
        except LexiconError:
            return False

    def get(self, word: str) -> str | None:
        """Pronunciation for ``word`` (case-insensitive), or None."""
        try:
            return self.entries.get(_normalise_key(word))
        except LexiconError:
            return None

    @property
    def max_phrase_words(self) -> int:
        """Longest entry span, for the longest-first matcher."""
        return max((len(key.split()) for key in self.entries), default=0)

    def with_entry(self, entry: PronunciationEntry) -> PronunciationLexicon:
        """A new lexicon with ``entry`` added (overwrites the same word)."""
        merged = dict(self.entries)
        merged[_normalise_key(entry.word)] = entry.pronunciation
        return PronunciationLexicon(entries=merged)

    def to_dict(self) -> dict[str, str]:
        """Entries as a plain ``{word: pronunciation}`` mapping."""
        return dict(self.entries)

    def to_json(self) -> str:
        return json.dumps({"entries": self.to_dict()}, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> PronunciationLexicon:
        """Parse ``{"entries": {"word": "pronunciation", ...}}``."""
        if not isinstance(data, dict):
            raise LexiconError("lexicon JSON must be an object")
        entries = data.get("entries", data) if "entries" in data else data
        if not isinstance(entries, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in entries.items()
        ):
            raise LexiconError("lexicon entries must be an object of string->string")
        return cls(entries=dict(entries))

    @classmethod
    def from_json(cls, text: str) -> PronunciationLexicon:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LexiconError(f"invalid lexicon JSON: {exc}") from exc
        return cls.from_dict(data)


def load_lexicon(path: str) -> PronunciationLexicon:
    """Load a lexicon JSON file; missing/unreadable files are hard errors."""
    try:
        with open(path, encoding="utf-8") as fh:
            return PronunciationLexicon.from_json(fh.read())
    except OSError as exc:
        raise LexiconError(f"cannot read lexicon {path!r}: {exc}") from exc


def _match_case(matched: str, replacement: str) -> str:
    """Shape ``replacement`` to the case pattern of the ``matched`` text.

    All-caps source ("NASA") gets an all-caps pronunciation; a
    capitalised source ("Karn") capitalises the pronunciation's first
    letter; anything else is used exactly as written in the lexicon.
    """
    if matched.isupper():
        return replacement.upper()
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


@dataclass(frozen=True)
class LexiconApplication:
    """What :func:`apply_lexicon` did: the rewritten text and the audit trail."""

    text: str
    replacements: list[tuple[str, str]] = field(default_factory=list)
    # (matched text as written, pronunciation applied), in order.

    @property
    def applied_count(self) -> int:
        return len(self.replacements)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "applied_count": self.applied_count,
            "replacements": [
                {"matched": matched, "pronunciation": spoken}
                for matched, spoken in self.replacements
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> LexiconApplication:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LexiconError(f"invalid lexicon application JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise LexiconError("lexicon application JSON needs a 'text' string")
        replacements = data.get("replacements", [])
        if not isinstance(replacements, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("matched"), str)
            and isinstance(item.get("pronunciation"), str)
            for item in replacements
        ):
            raise LexiconError(
                "lexicon application replacements must be matched/pronunciation pairs"
            )
        return cls(
            text=data["text"],
            replacements=[(item["matched"], item["pronunciation"]) for item in replacements],
        )

    def to_markdown(self) -> str:
        lines = [
            "# Pronunciation lexicon applied",
            "",
            f"- Replacements: {self.applied_count}",
            "",
        ]
        if self.replacements:
            lines.append("| As written | Spoken as |")
            lines.append("| --- | --- |")
            for matched, spoken in self.replacements:
                lines.append(f"| {matched} | {spoken} |")
            lines.append("")
        lines.append(f"Result: {self.text}")
        return "\n".join(lines) + "\n"


def apply_lexicon(text: str, lexicon: PronunciationLexicon) -> LexiconApplication:
    """Rewrite ``text`` with the lexicon's pronunciations.

    Longest phrase first, word boundaries from the grapheme-aware
    tokenizer (Devanagari and other scripts stay whole), single pass —
    a replacement is never re-scanned, so a pronunciation that happens
    to contain another entry's word is left alone. An empty lexicon
    returns the text untouched.
    """
    if not lexicon.entries or not text:
        return LexiconApplication(text=text)
    tokens = _tokenize(text)
    word_tokens = [
        (token, start, end) for token, start, end in tokens if token and _is_word_char(token[0])
    ]
    if not word_tokens:
        return LexiconApplication(text=text)

    out: list[str] = []
    replacements: list[tuple[str, str]] = []
    cursor = 0  # next unwritten index in the original text
    index = 0  # next word token to consider
    max_words = lexicon.max_phrase_words
    while index < len(word_tokens):
        matched = None
        for span in range(min(max_words, len(word_tokens) - index), 0, -1):
            group = word_tokens[index : index + span]
            # A phrase only matches across single spaces in the original.
            if any(text[group[i][2] : group[i + 1][1]] != " " for i in range(len(group) - 1)):
                continue
            key = " ".join(token.lower() for token, _, _ in group)
            pronunciation = lexicon.entries.get(key)
            if pronunciation is not None:
                matched = (group, pronunciation)
                break
        if matched is None:
            index += 1
            continue
        group, pronunciation = matched
        start, end = group[0][1], group[-1][2]
        out.append(text[cursor:start])
        spoken = _match_case(text[start:end], pronunciation)
        out.append(spoken)
        replacements.append((text[start:end], spoken))
        cursor = end
        index += len(group)
    out.append(text[cursor:])
    return LexiconApplication(text="".join(out), replacements=replacements)
