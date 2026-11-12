"""Tests for subtitle QA (qa.qa_transcript)."""

import json

import pytest

from video_dubber.cli import main
from video_dubber.qa import QaConfig, QaError, QaReport, qa_transcript
from video_dubber.transcript import Segment, Transcript, Word


def _seg(text: str, start: float, end: float) -> Segment:
    words = text.split()
    span = (end - start) / max(1, len(words))
    return Segment(
        words=[Word(w, start + i * span, start + (i + 1) * span) for i, w in enumerate(words)]
    )


def _kinds(report: QaReport) -> list[str]:
    return [issue.kind for issue in report.issues]


def test_clean_transcript_passes():
    transcript = Transcript(
        segments=[
            _seg("Welcome to Mars, everyone.", 0.0, 2.0),
            _seg("The match starts in five minutes.", 2.5, 5.0),
        ]
    )
    report = qa_transcript(transcript)
    assert report.verdict == "pass"
    assert report.errors == 0
    assert report.warnings == 0
    assert report.issues == []
    assert report.segments_checked == 2
    assert report.segments_skipped == 0


def test_too_fast_cue_is_a_warning():
    # 47 chars in 1.0s = 47 chars/s, well above the 20 cap.
    transcript = Transcript(
        segments=[_seg("Pack as many words as you can into this cue now", 0.0, 1.0)]
    )
    report = qa_transcript(transcript)
    assert report.verdict == "review"
    assert _kinds(report) == ["too-fast"]
    assert report.issues[0].severity == "warning"
    assert "47.0 chars/s" in report.issues[0].detail


def test_too_many_lines_and_long_word():
    # ~95 chars wraps to 3 lines at 42 chars.
    transcript = Transcript(
        segments=[
            _seg(
                "This line goes on and on and on until it simply cannot fit on two lines at all, not even close",
                0.0,
                6.0,
            )
        ]
    )
    report = qa_transcript(transcript)
    assert "too-many-lines" in _kinds(report)
    assert "3 lines" in report.issues[0].detail

    # One 48-char word can never be wrapped onto a 42-char line.
    wordy = Transcript(
        segments=[_seg("supercalifragilisticexpialidociousnesswinsbigger", 0.0, 3.0)]
    )
    wordy_report = qa_transcript(wordy)
    assert _kinds(wordy_report) == ["long-word"]


def test_too_short_and_too_long_cues():
    short = Transcript(segments=[_seg("Go!", 0.0, 0.5)])
    assert _kinds(qa_transcript(short)) == ["too-short"]

    long = Transcript(segments=[_seg("A very long pause.", 0.0, 8.0)])
    assert _kinds(qa_transcript(long)) == ["too-long"]


def test_overlap_is_the_only_error():
    transcript = Transcript(
        segments=[
            _seg("First line here.", 0.0, 2.0),
            _seg("Second line here.", 1.5, 3.5),
        ]
    )
    report = qa_transcript(transcript)
    assert _kinds(report) == ["overlap"]
    assert report.errors == 1
    assert report.issues[0].severity == "error"
    assert "500 ms" in report.issues[0].detail


def test_tiny_gap_between_cues():
    transcript = Transcript(
        segments=[
            _seg("First line here.", 0.0, 2.0),
            _seg("Second line here.", 2.05, 4.0),
        ]
    )
    report = qa_transcript(transcript)
    assert _kinds(report) == ["tiny-gap"]
    assert report.issues[0].severity == "warning"


def test_wordless_segments_are_skipped_not_flagged():
    transcript = Transcript(
        segments=[
            _seg("Spoken line.", 0.0, 2.0),
            Segment(words=[]),
        ]
    )
    report = qa_transcript(transcript)
    assert report.verdict == "pass"
    assert report.segments_checked == 1
    assert report.segments_skipped == 1


def test_config_validation():
    good = Transcript(segments=[_seg("Fine.", 0.0, 2.0)])
    for bad in [
        QaConfig(max_cps=0),
        QaConfig(max_line_chars=0),
        QaConfig(max_lines=0),
        QaConfig(min_cue_s=-1),
        QaConfig(max_cue_s=0),
        QaConfig(min_cue_s=5.0, max_cue_s=2.0),
        QaConfig(min_gap_s=-0.1),
    ]:
        with pytest.raises(QaError):
            qa_transcript(good, bad)


def test_empty_transcript_is_an_error():
    with pytest.raises(QaError):
        qa_transcript(Transcript(segments=[Segment(words=[])]))


def test_json_round_trip():
    transcript = Transcript(
        segments=[
            _seg("Pack as many words as you can into this cue now", 0.0, 1.0),
            _seg("Second line here.", 0.5, 2.5),
        ]
    )
    report = qa_transcript(transcript)
    rebuilt = QaReport.from_json(report.to_json())
    assert rebuilt == report
    assert rebuilt.verdict == "review"


def test_markdown_lists_every_issue():
    transcript = Transcript(
        segments=[_seg("Pack as many words as you can into this cue now", 0.0, 1.0)]
    )
    md = qa_transcript(transcript).to_markdown()
    assert "Verdict: **review**" in md
    assert "too-fast" in md
    assert "47.0 chars/s" in md


def _write_transcript(tmp_path, transcript):
    path = tmp_path / "t.json"
    path.write_text(transcript.to_json(), encoding="utf-8")
    return path


def test_cli_qa_pass_exits_zero(tmp_path, capsys):
    path = _write_transcript(
        tmp_path,
        Transcript(segments=[_seg("Welcome to Mars, everyone.", 0.0, 2.0)]),
    )
    assert main(["qa", str(path)]) == 0
    assert "Verdict: **pass**" in capsys.readouterr().out


def test_cli_qa_error_exits_one(tmp_path, capsys):
    path = _write_transcript(
        tmp_path,
        Transcript(
            segments=[
                _seg("First line here.", 0.0, 2.0),
                _seg("Second line here.", 1.5, 3.5),
            ]
        ),
    )
    assert main(["qa", str(path)]) == 1
    assert "overlap" in capsys.readouterr().out


def test_cli_qa_warnings_exit_zero_unless_fail_on_warn(tmp_path):
    path = _write_transcript(
        tmp_path,
        Transcript(segments=[_seg("Pack as many words as you can into this cue now", 0.0, 1.0)]),
    )
    assert main(["qa", str(path)]) == 0
    assert main(["qa", str(path), "--fail-on-warn"]) == 1


def test_cli_qa_json_and_report_out(tmp_path):
    path = _write_transcript(
        tmp_path,
        Transcript(segments=[_seg("Welcome to Mars, everyone.", 0.0, 2.0)]),
    )
    report_out = tmp_path / "qa.md"
    assert main(["qa", str(path), "--report-out", str(report_out)]) == 0
    assert "Verdict: **pass**" in report_out.read_text(encoding="utf-8")

    json_out = tmp_path / "qa.json"
    assert main(["qa", str(path), "--json", "--report-out", str(json_out)]) == 0
    assert json.loads(json_out.read_text(encoding="utf-8"))["verdict"] == "pass"


def test_cli_qa_bad_config_is_a_clean_error(tmp_path, capsys):
    path = _write_transcript(
        tmp_path,
        Transcript(segments=[_seg("Welcome to Mars, everyone.", 0.0, 2.0)]),
    )
    assert main(["qa", str(path), "--max-cps", "0"]) == 1
    assert "error:" in capsys.readouterr().err
