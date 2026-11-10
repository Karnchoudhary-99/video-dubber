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
