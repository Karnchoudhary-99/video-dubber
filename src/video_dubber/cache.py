"""Segment synthesis cache: skip re-synthesising what was already rendered.

Dubbing the same line twice — a re-run, a patch that touches one segment,
a watch-folder file re-dubbed after an edit — re-synthesises every
segment from scratch. Synthesis is the slowest step of the pipeline
(the demo double renders a harmonic tone per word), so this module
caches rendered segments on disk, keyed by everything that determines
the sound: the cloner, the exact text, the speaker's embedding and
prosody, the speaking rate and the sample rate.

The cache is content-addressed and self-maintaining:

* :func:`cache_key` hashes a canonical JSON document of those inputs,
  so the same request always maps to the same file and any change —
  one word, a different voice, a faster rate — maps to a different one.
* :class:`SynthesisCache` stores one 16-bit WAV per key plus a JSON
  index (hits, bytes, last-used). It evicts least-recently-used entries
  when the byte or entry budget is exceeded, and a corrupt or missing
  index simply starts empty — the cache is an optimisation, never a
  source of truth, so it must never break a dub.
* :class:`CachedVoiceCloner` wraps any :class:`VoiceCloner` behind the
  same protocol, so ``dub``, ``patch`` and ``synthesize`` use it
  without knowing a cache exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from video_dubber.audio import AudioData, load_wav, save_wav
from video_dubber.voice import DEFAULT_SAMPLE_RATE, SpeakerProfile, VoiceCloner

__all__ = [
    "CacheError",
    "CacheStats",
    "CachedVoiceCloner",
    "SynthesisCache",
    "cache_key",
    "default_cache_dir",
]

_INDEX_FILE = "index.json"
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB of rendered segments
_DEFAULT_MAX_ENTRIES = 1000


class CacheError(Exception):
    """Raised when the synthesis cache cannot be used as requested."""


def default_cache_dir() -> Path:
    """Where the cache lives when the caller does not pick a directory."""
    return Path.home() / ".cache" / "video-dubber" / "synth"


def cache_key(
    cloner_name: str,
    text: str,
    profile: SpeakerProfile,
    words_per_minute: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Content-addressed key for one synthesis request.

    The key covers the cloner, the exact text, the speaker's identity
    (name + embedding rounded to 6 decimals — float noise must not
    split the cache) and prosody (base pitch, effective speaking rate),
    and the sample rate. Two requests with the same key render the
    same audio; anything that changes the sound changes the key.
    """
    wpm = profile.speaking_rate_wpm if words_per_minute is None else words_per_minute
    if wpm <= 0:
        raise CacheError(f"words_per_minute must be positive, got {wpm}")
    if sample_rate <= 0:
        raise CacheError(f"sample_rate must be positive, got {sample_rate}")
    canonical = {
        "cloner": cloner_name,
        "text": text,
        "profile": profile.name,
        "embedding": [round(v, 6) for v in profile.embedding],
        "base_pitch_hz": profile.base_pitch_hz,
        "words_per_minute": wpm,
        "sample_rate": sample_rate,
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


@dataclass
class CacheStats:
    """A snapshot of what the cache holds and how it is performing."""

    entries: int = 0
    total_bytes: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from the cache (0.0 with no lookups)."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_markdown(self) -> str:
        return (
            f"## Synthesis cache\n\n"
            f"- entries: {self.entries}\n"
            f"- size: {self.total_bytes} bytes\n"
            f"- hits: {self.hits}\n"
            f"- misses: {self.misses}\n"
            f"- hit rate: {self.hit_rate:.2f}\n"
        )


@dataclass
class _Entry:
    file: str
    size_bytes: int
    hits: int = 0
    last_used: float = 0.0
    created: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "size_bytes": self.size_bytes,
            "hits": self.hits,
            "last_used": self.last_used,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> _Entry:
        return cls(
            file=str(raw["file"]),
            size_bytes=int(raw["size_bytes"]),
            hits=int(raw.get("hits", 0)),
            last_used=float(raw.get("last_used", 0.0)),
            created=float(raw.get("created", 0.0)),
        )


class SynthesisCache:
    """On-disk cache of synthesised segments, keyed by :func:`cache_key`.

    Entries live as ``<key>.wav`` files under ``directory``; an
    ``index.json`` tracks hits, sizes and last-use times so eviction
    can be least-recently-used. The index is written atomically (temp
    file + rename) and a corrupt index is discarded with a warning
    rather than failing the caller — losing the cache costs time, not
    correctness.
    """

    def __init__(
        self,
        directory: str | Path,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        if max_bytes <= 0:
            raise CacheError(f"max_bytes must be positive, got {max_bytes}")
        if max_entries <= 0:
            raise CacheError(f"max_entries must be positive, got {max_entries}")
        self.directory = Path(directory)
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(f"cannot create cache directory {self.directory}: {exc}") from exc
        self._hits = 0
        self._misses = 0
        self._entries: dict[str, _Entry] = {}
        self._load_index()

    # -- index ---------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.directory / _INDEX_FILE

    def _load_index(self) -> None:
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            self._hits = int(raw.get("hits", 0))
            self._misses = int(raw.get("misses", 0))
            self._entries = {
                key: _Entry.from_dict(value) for key, value in raw.get("entries", {}).items()
            }
        except FileNotFoundError:
            return  # first use: an empty cache is normal
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(
                f"warning: synthesis cache index {self._index_path} is unreadable "
                f"({exc}); starting with an empty cache",
                file=sys.stderr,
            )
            self._hits = 0
            self._misses = 0
            self._entries = {}

    def _save_index(self) -> None:
        payload = {
            "version": 1,
            "hits": self._hits,
            "misses": self._misses,
            "entries": {key: entry.to_dict() for key, entry in self._entries.items()},
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._index_path)

    # -- lookups ---------------------------------------------------------------

    def get(self, key: str) -> AudioData | None:
        """Return the cached audio for ``key``, or ``None`` on a miss.

        A hit bumps the entry's hit count and last-used time. A stale
        entry (index says the file exists but it does not, or the WAV
        is unreadable) is dropped and counts as a miss — disk state
        wins over the index.
        """
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            self._save_index()
            return None
        path = self.directory / entry.file
        if not path.exists():
            del self._entries[key]
            self._misses += 1
            self._save_index()
            return None
        try:
            audio = load_wav(str(path))
        except (OSError, ValueError):
            del self._entries[key]
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self._misses += 1
            self._save_index()
            return None
        self._hits += 1
        entry.hits += 1
        entry.last_used = time.time()
        self._save_index()
        return audio

    def put(self, key: str, audio: AudioData) -> None:
        """Store ``audio`` under ``key``, evicting LRU entries if needed."""
        filename = f"{key}.wav"
        tmp = self.directory / f".{filename}.tmp"
        save_wav(str(tmp), audio)
        os.replace(tmp, self.directory / filename)
        now = time.time()
        previous = self._entries.get(key)
        self._entries[key] = _Entry(
            file=filename,
            size_bytes=(self.directory / filename).stat().st_size,
            hits=previous.hits if previous is not None else 0,
            last_used=now,
            created=previous.created if previous is not None else now,
        )
        self._prune()
        self._save_index()

    # -- maintenance -----------------------------------------------------------

    def _total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self._entries.values())

    def _prune(self) -> None:
        """Evict least-recently-used entries until both budgets hold."""
        while self._entries and (
            self._total_bytes() > self.max_bytes or len(self._entries) > self.max_entries
        ):
            oldest_key = min(self._entries, key=lambda k: self._entries[k].last_used)
            entry = self._entries.pop(oldest_key)
            try:
                (self.directory / entry.file).unlink(missing_ok=True)
            except OSError:
                pass

    def stats(self) -> CacheStats:
        """Current size and hit/miss counts."""
        return CacheStats(
            entries=len(self._entries),
            total_bytes=self._total_bytes(),
            hits=self._hits,
            misses=self._misses,
        )

    def clear(self) -> int:
        """Delete every cached file and the index; returns entries removed."""
        removed = len(self._entries)
        for entry in self._entries.values():
            try:
                (self.directory / entry.file).unlink(missing_ok=True)
            except OSError:
                pass
        self._entries = {}
        self._hits = 0
        self._misses = 0
        try:
            self._index_path.unlink(missing_ok=True)
        except OSError:
            pass
        return removed


@dataclass
class CachedVoiceCloner:
    """A :class:`VoiceCloner` that caches synthesis through a :class:`SynthesisCache`.

    ``clone`` delegates to the wrapped cloner (profiles are cheap to
    build, and the cache keys on the profile's content anyway).
    ``synthesize`` serves a cache hit without touching the wrapped
    cloner, and stores a miss — so a re-dub of the same transcript
    in the same voice renders each segment once, ever. A cache that
    cannot store (full disk, permissions) warns and keeps going: the
    dub still renders, it just costs the synthesis again.
    """

    inner: VoiceCloner
    cache: SynthesisCache
    hits: int = 0
    misses: int = 0

    @property
    def name(self) -> str:
        return f"{self.inner.name}+cache"

    def clone(
        self,
        name: str,
        audios: list[AudioData],
        reference_files: list[str] | None = None,
    ) -> SpeakerProfile:
        return self.inner.clone(name, audios, reference_files)

    def synthesize(
        self, text: str, profile: SpeakerProfile, words_per_minute: float | None = None
    ) -> AudioData:
        key = cache_key(self.inner.name, text, profile, words_per_minute)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        audio = self.inner.synthesize(text, profile, words_per_minute)
        try:
            self.cache.put(key, audio)
        except (CacheError, OSError) as exc:
            print(f"warning: could not cache synthesis ({exc})", file=sys.stderr)
        return audio
