"""Dub cost/time estimator: predict a dub before running it.

Dubbing a long video is the most expensive step in the pipeline — it
synthesises every segment, and a bad fit only shows up afterwards as
stretched audio. :func:`estimate_dub` answers the planning questions
up front, from the translated transcript alone:

* How long will the dubbed speech be, segment by segment, and how
  hard will each segment have to stretch to fit its window?
* How much synthesis wall time will it take, for this backend?
* How much of it is already in the synthesis cache?
* What will it cost, if the backend charges per character or per minute?

The speech-duration model is exact for the scripted backend: it is
the same per-word formula :func:`video_dubber.voice.synthesize_text`
uses (``max(0.12, len(word) / 5 * 60 / wpm)`` per word plus a 50 ms
gap), so a ``scripted`` estimate predicts the real rendered duration
to the sample. For other backends the same word-rate model is the
best available guess and is labelled as such — a neural TTS voice
paces words differently, and the estimate says so rather than
pretending to know.

Wall time and cost are estimates, not promises: the per-backend
real-time factors in :data:`BACKEND_RTF` are rough defaults (a
backend renders ``rtf`` seconds of audio per second of wall time).
Measure your own backend and override with ``real_time_factor`` (or
``--rtf`` on the CLI) when the numbers matter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from video_dubber.cache import SynthesisCache, cache_key
from video_dubber.transcript import Transcript
from video_dubber.voice import SpeakerProfile

__all__ = [
    "BACKEND_RTF",
    "DubEstimate",
    "EstimateError",
    "EstimateRow",
    "estimate_dub",
    "predict_speech_duration",
]

# Rough real-time factors (seconds of audio rendered per second of wall
# time) for known backends. These are planning defaults, not benchmarks:
# override them when you have measured numbers for your own setup.
BACKEND_RTF: dict[str, float] = {
    "scripted": 20.0,  # in-process tone synthesis, no model load per segment
    "coqui": 2.0,
    "openvoice": 1.5,
    "elevenlabs": 0.5,  # network round-trip per request dominates
}

DEFAULT_WPM = 150.0
# Same constants as synthesize_text: 5 chars ≈ 1 word, 50 ms word gap,
# 0.12 s minimum per word.
CHARS_PER_WORD = 5.0
WORD_GAP_S = 0.05
MIN_WORD_S = 0.12
# Fit factors outside this band are flagged extreme, matching the
# quality report's definition (see assemble.DubReport.extreme_fits).
EXTREME_FIT_LOW = 0.8
EXTREME_FIT_HIGH = 1.25
# Per-segment fixed overhead (process spawn, file I/O) in the wall-time
# estimate, in seconds.
OVERHEAD_PER_SEGMENT_S = 0.05


class EstimateError(Exception):
    """Raised when an estimate cannot be computed from the inputs."""


def predict_speech_duration(text: str, words_per_minute: float) -> float:
    """Predicted speech duration for ``text`` at ``words_per_minute``.

    This is the scripted synthesiser's own timing model, so for the
    ``scripted`` backend the prediction is exact: each word takes
    ``max(0.12, len(word) / 5 * 60 / wpm)`` seconds plus a 50 ms gap.
    Empty text predicts silence (0.0 s).
    """
    if words_per_minute <= 0:
        raise EstimateError(f"words_per_minute must be positive, got {words_per_minute}")
    words = text.split()
    if not words:
        return 0.0
    total = 0.0
    for word in words:
        total += max(MIN_WORD_S, (len(word) / CHARS_PER_WORD) * (60.0 / words_per_minute))
        total += WORD_GAP_S
    return total


@dataclass
class EstimateRow:
    """One segment's estimate: window, predicted speech, fit, cache hit."""

    index: int  # 1-based, matching the report's numbering
    start: float
    end: float
    window_s: float
    words: int
    chars: int
    predicted_s: float
    fit_factor: float  # predicted_s / window_s; >1 means stretch
    extreme: bool  # fit outside [0.8, 1.25]
    cached: bool  # already in the synthesis cache — no synthesis needed

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "window_s": round(self.window_s, 3),
            "words": self.words,
            "chars": self.chars,
            "predicted_s": round(self.predicted_s, 3),
            "fit_factor": round(self.fit_factor, 3),
            "extreme": self.extreme,
            "cached": self.cached,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EstimateRow:
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            window_s=float(data["window_s"]),
            words=int(data["words"]),
            chars=int(data["chars"]),
            predicted_s=float(data["predicted_s"]),
            fit_factor=float(data["fit_factor"]),
            extreme=bool(data["extreme"]),
            cached=bool(data["cached"]),
        )


@dataclass
class DubEstimate:
    """A whole-transcript dub estimate, JSON-serialisable."""

    backend: str
    target: str | None
    words_per_minute: float
    real_time_factor: float
    rows: list[EstimateRow] = field(default_factory=list)
    # Cost inputs, echoed so a saved estimate stays auditable.
    price_per_char: float | None = None
    price_per_minute: float | None = None
    currency: str = "USD"
    # True when the duration model is exact (scripted backend); other
    # backends pace words differently, so their numbers are approximate.
    duration_exact: bool = False

    @property
    def segments(self) -> int:
        return len(self.rows)

    @property
    def source_duration_s(self) -> float:
        """Span of the source transcript, first word start to last end."""
        if not self.rows:
            return 0.0
        return self.rows[-1].end - self.rows[0].start

    @property
    def total_words(self) -> int:
        return sum(row.words for row in self.rows)

    @property
    def total_chars(self) -> int:
        return sum(row.chars for row in self.rows)

    @property
    def predicted_speech_s(self) -> float:
        return sum(row.predicted_s for row in self.rows)

    @property
    def mean_fit_factor(self) -> float:
        if not self.rows:
            return 0.0
        return sum(row.fit_factor for row in self.rows) / len(self.rows)

    @property
    def max_fit_factor(self) -> float:
        return max((row.fit_factor for row in self.rows), default=0.0)

    @property
    def extreme_fits(self) -> int:
        return sum(1 for row in self.rows if row.extreme)

    @property
    def cached_segments(self) -> int:
        return sum(1 for row in self.rows if row.cached)

    @property
    def chars_to_synthesize(self) -> int:
        """Characters that still need synthesis (cache hits excluded)."""
        return sum(row.chars for row in self.rows if not row.cached)

    @property
    def est_synth_wall_s(self) -> float:
        """Estimated synthesis wall time for the uncached segments."""
        uncached_s = sum(row.predicted_s for row in self.rows if not row.cached)
        return uncached_s / self.real_time_factor

    @property
    def est_total_wall_s(self) -> float:
        """Synthesis wall time plus per-segment overhead."""
        return self.est_synth_wall_s + OVERHEAD_PER_SEGMENT_S * len(self.rows)

    @property
    def est_cost(self) -> float | None:
        """Estimated cost in ``currency``, or None when unpriced."""
        if self.price_per_char is not None:
            return self.chars_to_synthesize * self.price_per_char
        if self.price_per_minute is not None:
            uncached_s = sum(row.predicted_s for row in self.rows if not row.cached)
            return (uncached_s / 60.0) * self.price_per_minute
        return None

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "target": self.target,
            "words_per_minute": self.words_per_minute,
            "real_time_factor": self.real_time_factor,
            "duration_exact": self.duration_exact,
            "segments": self.segments,
            "source_duration_s": round(self.source_duration_s, 3),
            "total_words": self.total_words,
            "total_chars": self.total_chars,
            "predicted_speech_s": round(self.predicted_speech_s, 3),
            "mean_fit_factor": round(self.mean_fit_factor, 3),
            "max_fit_factor": round(self.max_fit_factor, 3),
            "extreme_fits": self.extreme_fits,
            "cached_segments": self.cached_segments,
            "chars_to_synthesize": self.chars_to_synthesize,
            "est_synth_wall_s": round(self.est_synth_wall_s, 3),
            "est_total_wall_s": round(self.est_total_wall_s, 3),
            "est_cost": (
                round(self.est_cost, 4) if self.est_cost is not None else None
            ),
            "currency": self.currency,
            "price_per_char": self.price_per_char,
            "price_per_minute": self.price_per_minute,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> DubEstimate:
        return cls(
            backend=data["backend"],
            target=data.get("target"),
            words_per_minute=float(data["words_per_minute"]),
            real_time_factor=float(data["real_time_factor"]),
            rows=[EstimateRow.from_dict(row) for row in data.get("rows", [])],
            price_per_char=data.get("price_per_char"),
            price_per_minute=data.get("price_per_minute"),
            currency=data.get("currency", "USD"),
            duration_exact=bool(data.get("duration_exact", False)),
        )

    @classmethod
    def from_json(cls, data: str) -> DubEstimate:
        try:
            return cls.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EstimateError(f"invalid dub estimate JSON: {exc}") from exc

    def to_markdown(self) -> str:
        """Human-readable estimate: headline numbers plus a per-segment table."""
        lines = ["# Dub estimate", ""]
        lines.append(f"- Backend: `{self.backend}`" + (" (exact durations)" if self.duration_exact else " (approximate durations)"))
        if self.target:
            lines.append(f"- Target language: `{self.target}`")
        lines.append(f"- Speaking rate: {self.words_per_minute:.0f} wpm")
        lines.append(f"- Segments: {self.segments} over {self.source_duration_s:.2f}s of source")
        lines.append(f"- Words: {self.total_words} ({self.total_chars} chars)")
        lines.append(
            f"- Predicted speech: {self.predicted_speech_s:.2f}s "
            f"(mean fit {self.mean_fit_factor:.2f}x, max {self.max_fit_factor:.2f}x, "
            f"{self.extreme_fits} extreme)"
        )
        if self.cached_segments:
            lines.append(
                f"- Cache: {self.cached_segments}/{self.segments} segments already "
                f"cached — {self.chars_to_synthesize} of {self.total_chars} chars to synthesise"
            )
        lines.append(
            f"- Estimated wall time: {self.est_total_wall_s:.1f}s "
            f"({self.est_synth_wall_s:.1f}s synthesis at {self.real_time_factor:g}x real-time "
            f"+ {OVERHEAD_PER_SEGMENT_S * len(self.rows):.1f}s overhead)"
        )
        if self.est_cost is not None:
            lines.append(f"- Estimated cost: {self.est_cost:.4f} {self.currency}")
        lines.append("")
        lines.append("| # | window | words | predicted | fit | cached |")
        lines.append("|---:|---:|---:|---:|---:|:---:|")
        for row in self.rows:
            lines.append(
                f"| {row.index} | {row.start:.2f}–{row.end:.2f}s "
                f"| {row.words} | {row.predicted_s:.2f}s "
                f"| {row.fit_factor:.2f}x{' ⚠' if row.extreme else ''} "
                f"| {'yes' if row.cached else 'no'} |"
            )
        return "\n".join(lines) + "\n"


def estimate_dub(
    transcript: Transcript,
    *,
    backend: str = "scripted",
    target: str | None = None,
    words_per_minute: float = DEFAULT_WPM,
    real_time_factor: float | None = None,
    price_per_char: float | None = None,
    price_per_minute: float | None = None,
    currency: str = "USD",
    cache: SynthesisCache | None = None,
    profile: SpeakerProfile | None = None,
    cloner_name: str | None = None,
) -> DubEstimate:
    """Estimate dubbing ``transcript`` (already translated) with ``backend``.

    ``words_per_minute`` is the speaker's rate (from the profile when
    one is given); ``real_time_factor`` overrides the backend default
    in :data:`BACKEND_RTF`. Pass ``cache`` + ``profile`` to count
    cache hits — the probe reads the cache index only, never the audio,
    so estimating never bumps hit counts or rewrites the index.
    """
    if not transcript.segments:
        raise EstimateError("cannot estimate an empty transcript")
    if words_per_minute <= 0:
        raise EstimateError(f"words_per_minute must be positive, got {words_per_minute}")
    if price_per_char is not None and price_per_minute is not None:
        raise EstimateError("pass at most one of price_per_char / price_per_minute")
    if price_per_char is not None and price_per_char < 0:
        raise EstimateError(f"price_per_char must be >= 0, got {price_per_char}")
    if price_per_minute is not None and price_per_minute < 0:
        raise EstimateError(f"price_per_minute must be >= 0, got {price_per_minute}")
    if real_time_factor is None:
        if backend not in BACKEND_RTF:
            raise EstimateError(
                f"unknown backend {backend!r}: pass --rtf with a measured "
                f"real-time factor (known backends: {', '.join(BACKEND_RTF)})"
            )
        real_time_factor = BACKEND_RTF[backend]
    if real_time_factor <= 0:
        raise EstimateError(f"real_time_factor must be positive, got {real_time_factor}")
    if cache is not None and profile is None:
        raise EstimateError("cache probing needs a speaker profile (--profile)")

    key_cloner = cloner_name or backend
    rows: list[EstimateRow] = []
    for index, segment in enumerate(transcript.segments, start=1):
        text = segment.text
        predicted = predict_speech_duration(text, words_per_minute)
        window = segment.duration
        fit = predicted / window if window > 0 else 0.0
        cached = False
        if cache is not None and profile is not None:
            key = cache_key(key_cloner, text, profile, words_per_minute)
            cached = cache.contains(key)
        rows.append(
            EstimateRow(
                index=index,
                start=segment.start,
                end=segment.end,
                window_s=window,
                words=len(text.split()),
                chars=len(text),
                predicted_s=predicted,
                fit_factor=fit,
                extreme=not (EXTREME_FIT_LOW <= fit <= EXTREME_FIT_HIGH),
                cached=cached,
            )
        )
    return DubEstimate(
        backend=backend,
        target=target,
        words_per_minute=words_per_minute,
        real_time_factor=real_time_factor,
        rows=rows,
        price_per_char=price_per_char,
        price_per_minute=price_per_minute,
        currency=currency,
        duration_exact=(backend == "scripted"),
    )
