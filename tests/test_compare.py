"""Tests for dub A/B comparison."""

import json

import pytest

from video_dubber.assemble import DubReport
from video_dubber.compare import DubComparison, compare_reports


def _report(**overrides) -> DubReport:
    base = {
        "segments_dubbed": 2,
        "total_duration": 2.5,
        "mean_sync_drift": 0.02,
        "max_sync_drift": 0.05,
        "segments_over_drift": 0,
        "mean_fit_factor": 1.02,
        "max_fit_factor": 1.1,
        "extreme_fits": 0,
        "loudness_error_db": 0.5,
        "translation_coverage": 0.9,
        "word_error_rate": None,
        "mos_proxy": 4.5,
        "verdict": "good",
    }
    base.update(overrides)
    return DubReport(**base)


def _metric(comparison, name):
    return next(m for m in comparison.metrics if m.name == name)


def test_better_dub_wins_every_metric():
    good = _report()
    bad = _report(
        mean_sync_drift=0.15,
        max_sync_drift=0.4,
        segments_over_drift=1,
        mean_fit_factor=0.7,
        extreme_fits=1,
        loudness_error_db=2.0,
        translation_coverage=0.6,
        mos_proxy=3.2,
        verdict="review",
    )
    comparison = compare_reports(good, bad, "asha", "ravi")
    assert comparison.verdict == "asha"
    assert comparison.a_wins == 9
    assert comparison.b_wins == 0
    assert _metric(comparison, "MOS proxy").winner == "a"
    assert _metric(comparison, "Mean fit factor").winner == "a"
    assert _metric(comparison, "Verdict").winner == "a"


def test_identical_reports_tie():
    comparison = compare_reports(_report(), _report())
    assert comparison.verdict == "tie"
    assert all(m.winner == "tie" for m in comparison.metrics)


def test_win_count_beats_mos_on_tiebreak():
    # A wins 2 metrics, B wins 1 (the MOS proxy) — A still wins overall.
    a = _report(mean_sync_drift=0.01, max_sync_drift=0.02, mos_proxy=4.0)
    b = _report(mos_proxy=4.8)
    comparison = compare_reports(a, b, "asha", "ravi")
    assert comparison.a_wins == 2
    assert comparison.b_wins == 1
    assert comparison.verdict == "asha"


def test_equal_wins_broken_by_mos():
    # Wins split 3-3 including MOS; the better MOS proxy decides.
    a = _report(
        mean_sync_drift=0.01,
        max_sync_drift=0.02,
        extreme_fits=1,
        mos_proxy=4.6,
    )
    b = _report(loudness_error_db=0.1, translation_coverage=0.95, mos_proxy=3.0)
    comparison = compare_reports(a, b)
    assert comparison.a_wins == 3
    assert comparison.b_wins == 3
    assert comparison.verdict == "A"


def test_missing_optional_metrics_are_skipped_not_lost():
    # Neither report has a WER: no WER row, and no side is penalised.
    comparison = compare_reports(_report(), _report())
    assert not any(m.name == "Word error rate" for m in comparison.metrics)
    # Only one side has a WER: still skipped — a missing number is not a loss.
    with_wer = _report(word_error_rate=0.1)
    comparison = compare_reports(_report(), with_wer)
    assert not any(m.name == "Word error rate" for m in comparison.metrics)


def test_wer_compared_when_both_have_it():
    a = _report(word_error_rate=0.05)
    b = _report(word_error_rate=0.25)
    comparison = compare_reports(a, b)
    assert _metric(comparison, "Word error rate").winner == "a"


def test_fit_factor_scored_on_distance_from_one():
    # 0.9 (0.1 away) beats 1.2 (0.2 away), even though 1.2 is the bigger number.
    a = _report(mean_fit_factor=0.9)
    b = _report(mean_fit_factor=1.2)
    assert _metric(compare_reports(a, b), "Mean fit factor").winner == "a"


def test_comparison_json_round_trip():
    comparison = compare_reports(_report(), _report(mos_proxy=3.0), "asha", "ravi")
    rebuilt = DubComparison.from_json(comparison.to_json())
    assert rebuilt.verdict == comparison.verdict
    assert [(m.name, m.winner) for m in rebuilt.metrics] == [
        (m.name, m.winner) for m in comparison.metrics
    ]


def test_comparison_from_bad_json_raises():
    with pytest.raises(ValueError):
        DubComparison.from_json("{not json")
    with pytest.raises(ValueError):
        DubComparison.from_json(json.dumps({"label_a": "asha"}))


def test_markdown_names_winner_and_metrics():
    text = compare_reports(_report(), _report(mos_proxy=3.0), "asha", "ravi").to_markdown()
    assert "Verdict: **asha**" in text
    assert "| MOS proxy | 4.5 | 3.0 | asha |" in text
