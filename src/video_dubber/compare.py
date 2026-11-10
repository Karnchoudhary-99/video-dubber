"""A/B comparison of two dub quality reports.

When two dubs of the same source exist — different voices, different
backends, before/after a fix — the question is which one is better.
:func:`compare_reports` answers it from the numbers in the two
:class:`~video_dubber.assemble.DubReport` objects instead of by ear:
one :class:`ComparisonMetric` per quality number (MOS proxy, sync
drift, fit factors, loudness error, translation coverage, WER), each
naming its winner, and an overall verdict from the metric wins.

The verdict rule is deliberately plain: the side that wins more
metrics wins overall; a tie on wins is broken by the MOS proxy, and
if that ties too the comparison is a draw. Metrics that only one
report has (a WER needs a same-language reference; loudness error
needs the original) are skipped, not counted against the report that
lacks them — a missing number is not a loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from video_dubber.assemble import DubReport

__all__ = ["ComparisonMetric", "DubComparison", "compare_reports"]


@dataclass(frozen=True)
class ComparisonMetric:
    """One quality number, compared between the two dubs."""

    name: str
    a_value: float | str
    b_value: float | str
    # "a" | "b" | "tie" — which side is better on this metric.
    winner: str
    # What "better" means for this metric, for the report reader.
    better: str  # "higher" | "lower"
    detail: str


@dataclass
class DubComparison:
    """The outcome of :func:`compare_reports` for two dubs."""

    label_a: str
    label_b: str
    metrics: list[ComparisonMetric] = field(default_factory=list)

    @property
    def a_wins(self) -> int:
        return sum(1 for m in self.metrics if m.winner == "a")

    @property
    def b_wins(self) -> int:
        return sum(1 for m in self.metrics if m.winner == "b")

    @property
    def verdict(self) -> str:
        """``label_a`` | ``label_b`` | ``"tie"`` — which dub is better overall."""
        if self.a_wins > self.b_wins:
            return self.label_a
        if self.b_wins > self.a_wins:
            return self.label_b
        mos = next((m for m in self.metrics if m.name == "MOS proxy"), None)
        if mos is not None and mos.winner != "tie":
            return self.label_a if mos.winner == "a" else self.label_b
        return "tie"

    def to_dict(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "verdict": self.verdict,
            "metrics": [
                {
                    "name": m.name,
                    "a": m.a_value,
                    "b": m.b_value,
                    "winner": m.winner,
                    "better": m.better,
                    "detail": m.detail,
                }
                for m in self.metrics
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> DubComparison:
        """Rebuild from :meth:`to_json` output."""
        if not isinstance(data, dict):
            raise TypeError("dub comparison must be a JSON object")
        try:
            return cls(
                label_a=str(data["label_a"]),
                label_b=str(data["label_b"]),
                metrics=[
                    ComparisonMetric(
                        name=str(item["name"]),
                        a_value=item["a"],
                        b_value=item["b"],
                        winner=str(item["winner"]),
                        better=str(item["better"]),
                        detail=str(item["detail"]),
                    )
                    for item in data["metrics"]
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid dub comparison: {exc}") from exc

    @classmethod
    def from_json(cls, text: str) -> DubComparison:
        """Rebuild from :meth:`to_json` output."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        try:
            return cls.from_dict(data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def _higher(name: str, a: float, b: float, detail: str) -> ComparisonMetric:
    winner = "a" if a > b else ("b" if b > a else "tie")
    return ComparisonMetric(name, round(a, 3), round(b, 3), winner, "higher", detail)


def _lower(name: str, a: float, b: float, detail: str) -> ComparisonMetric:
    winner = "a" if a < b else ("b" if b < a else "tie")
    return ComparisonMetric(name, round(a, 3), round(b, 3), winner, "lower", detail)


def _optional_lower(
    name: str, a: float | None, b: float | None, detail: str
) -> ComparisonMetric | None:
    """A metric both reports must have; None when either side lacks it."""
    if a is None or b is None:
        return None
    return _lower(name, a, b, detail)


def _optional_higher(
    name: str, a: float | None, b: float | None, detail: str
) -> ComparisonMetric | None:
    if a is None or b is None:
        return None
    return _higher(name, a, b, detail)


def compare_reports(
    a: DubReport, b: DubReport, label_a: str = "A", label_b: str = "B"
) -> DubComparison:
    """Compare two dub reports metric by metric.

    ``Mean fit factor`` is scored on distance from 1.0 (no stretch is
    best), and ``Loudness error`` on distance from 0 dB. The report
    verdict (``good`` beats ``review``) is a metric too, so a dub that
    the pipeline itself flagged as needing review cannot win on
    numbers alone without that counting against it.
    """
    metrics: list[ComparisonMetric] = [
        _higher("MOS proxy", a.mos_proxy, b.mos_proxy, "heuristic 1-5 quality score"),
        _lower(
            "Mean sync drift",
            a.mean_sync_drift,
            b.mean_sync_drift,
            "seconds, lower is tighter to the original timing",
        ),
        _lower(
            "Max sync drift",
            a.max_sync_drift,
            b.max_sync_drift,
            "seconds, worst single segment",
        ),
        _lower(
            "Segments over 0.10s drift",
            float(a.segments_over_drift),
            float(b.segments_over_drift),
            "segments the listener would notice drifting",
        ),
        _lower(
            "Mean fit factor",
            abs(a.mean_fit_factor - 1.0),
            abs(b.mean_fit_factor - 1.0),
            "distance from 1.0 — no time-stretch is best",
        ),
        _lower(
            "Extreme fits",
            float(a.extreme_fits),
            float(b.extreme_fits),
            "segments stretched outside 0.8-1.25",
        ),
        ComparisonMetric(
            "Verdict",
            a.verdict,
            b.verdict,
            "a"
            if a.verdict == "good" and b.verdict != "good"
            else ("b" if b.verdict == "good" and a.verdict != "good" else "tie"),
            "higher",
            "good beats review",
        ),
    ]
    optional = [
        _optional_lower(
            "Loudness error",
            None if a.loudness_error_db is None else abs(a.loudness_error_db),
            None if b.loudness_error_db is None else abs(b.loudness_error_db),
            "dB from the original's loudness",
        ),
        _optional_higher(
            "Translation coverage",
            a.translation_coverage,
            b.translation_coverage,
            "fraction of words the dictionary covered",
        ),
        _optional_lower(
            "Word error rate",
            a.word_error_rate,
            b.word_error_rate,
            "vs the same-language reference, when one was given",
        ),
    ]
    metrics.extend(m for m in optional if m is not None)
    return DubComparison(label_a=label_a, label_b=label_b, metrics=metrics)
