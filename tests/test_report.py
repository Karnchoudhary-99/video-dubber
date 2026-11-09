"""Tests for the HTML report renderer (video_dubber.report)."""

from video_dubber.assemble import DubReport, SegmentRow
from video_dubber.batch import BatchJobResult, BatchSummary
from video_dubber.report import render_batch_html, render_html_report


def _report(**overrides) -> DubReport:
    base = {
        "segments_dubbed": 2,
        "total_duration": 4.0,
        "mean_sync_drift": 0.02,
        "max_sync_drift": 0.05,
        "segments_over_drift": 0,
        "mean_fit_factor": 1.05,
        "max_fit_factor": 1.1,
        "extreme_fits": 0,
        "loudness_error_db": None,
        "translation_coverage": None,
    }
    base.update(overrides)
    return DubReport(**base)


def test_good_verdict_banner_and_mos_bar():
    html = render_html_report(_report(mos_proxy=4.5))
    assert 'class="banner good"' in html
    assert "Verdict: good" in html
    assert "MOS proxy 4.50 / 5" in html
    assert 'style="width: 90%"' in html
    assert "<title>Dub quality report</title>" in html


def test_review_verdict_banner():
    html = render_html_report(_report(verdict="review", mos_proxy=2.0))
    assert 'class="banner review"' in html
    assert "Verdict: review" in html
    assert 'style="width: 40%"' in html


def test_segment_table_flags_drift_and_extreme_fit():
    report = _report(
        segments=[
            SegmentRow(1, 0.0, 2.0, 0.02, 1.05, speaker="Asha", text="hello world"),
            SegmentRow(2, 2.0, 4.0, 0.30, 1.05, speaker="Ravi", text="late line"),
            SegmentRow(3, 4.0, 6.0, 0.01, 1.55, text="rushed line"),
        ]
    )
    html = render_html_report(report)
    assert "<h2>Segments</h2>" in html
    assert "hello world" in html and "Asha" in html
    # Drifted + stretched rows are flagged; the clean row is not.
    assert html.count('class="flagged"') == 2
    assert "0.300" in html and "1.550" in html


def test_segment_text_is_escaped():
    report = _report(segments=[SegmentRow(1, 0.0, 1.0, 0.0, 1.0, text="<script>alert(1)</script>")])
    html = render_html_report(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_no_segment_table_when_no_segments():
    assert "<h2>Segments</h2>" not in render_html_report(_report())


def test_optional_metrics_only_when_present():
    plain = render_html_report(_report())
    assert "Loudness error" not in plain
    assert "Word error rate" not in plain
    assert "Speakers dubbed" not in plain
    rich = render_html_report(
        _report(
            loudness_error_db=1.25,
            translation_coverage=0.9,
            word_error_rate=0.05,
            speakers_dubbed=2,
            bed_level=0.3,
        )
    )
    assert "Loudness error vs original" in rich and "1.25 dB" in rich
    assert "Translation coverage" in rich and "90%" in rich
    assert "Word error rate vs reference" in rich and "0.050" in rich
    assert "Speakers dubbed" in rich
    assert "Background bed level" in rich and "0.30" in rich


def test_batch_html_counts_and_failed_rows():
    summary = BatchSummary(
        results=[
            BatchJobResult("one", "ok", verdict="good", mos_proxy=4.8, duration_s=12.0),
            BatchJobResult("two", "failed", error="ffmpeg missing", duration_s=0.0),
            BatchJobResult("three", "skipped", duration_s=0.0),
        ]
    )
    html = render_batch_html(summary)
    assert 'class="banner review"' in html  # a failure makes the banner amber
    assert "1 ok, 1 failed, 1 skipped out of 3 jobs" in html
    assert html.count('class="flagged"') == 1
    assert "<h2>Failures</h2>" in html
    assert "ffmpeg missing" in html
    assert "4.80" in html


def test_batch_html_all_ok_is_good_and_escapes_names():
    summary = BatchSummary(
        results=[
            BatchJobResult("<b>weird</b>", "ok", verdict="good", mos_proxy=5.0, duration_s=3.0)
        ]
    )
    html = render_batch_html(summary)
    assert 'class="banner good"' in html
    assert "<b>weird</b>" not in html
    assert "&lt;b&gt;weird&lt;/b&gt;" in html
    assert "<h2>Failures</h2>" not in html
