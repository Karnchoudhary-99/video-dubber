"""Tests for environment diagnostics (doctor.run_doctor)."""

import subprocess

import pytest

from video_dubber.doctor import (
    DoctorCheck,
    DoctorError,
    DoctorReport,
    _check_binary,
    _check_disk_space,
    _check_subtitles_filter,
    run_doctor,
)


class _FakeUsage:
    def __init__(self, free: int) -> None:
        self.free = free
        self.total = free
        self.used = 0


def _fake_run_factory(outputs: dict[str, str]):
    """A subprocess.run stand-in returning canned stdout per command."""

    def fake_run(command, **kwargs):
        class Result:
            returncode = 0
            stdout = outputs.get(command[0], "")

        return Result()

    return fake_run


def test_check_status_validation():
    with pytest.raises(DoctorError, match="pass/warn/fail"):
        DoctorCheck(name="x", status="maybe", detail="?")


def test_binary_found_reports_version(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(
        "video_dubber.doctor.subprocess.run",
        _fake_run_factory({"ffmpeg": "ffmpeg version 6.1.1\nbuilt with gcc"}),
    )
    check = _check_binary("ffmpeg", ["-version"], True, "install ffmpeg")
    assert check.status == "pass"
    assert "ffmpeg version 6.1.1" in check.detail
    assert "/usr/bin/ffmpeg" in check.detail


def test_binary_missing_is_fail_when_required_warn_when_optional(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: None)
    required = _check_binary("ffmpeg", ["-version"], True, "install ffmpeg")
    assert required.status == "fail"
    assert required.required is True
    optional = _check_binary("whisper", ["--version"], False, "pip install openai-whisper")
    assert optional.status == "warn"
    assert optional.required is False


def test_binary_version_unreadable_still_passes(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: "/usr/bin/ffmpeg")

    def failing_run(command, **kwargs):
        raise OSError("cannot exec")

    monkeypatch.setattr("video_dubber.doctor.subprocess.run", failing_run)
    check = _check_binary("ffmpeg", ["-version"], True, "install ffmpeg")
    assert check.status == "pass"
    assert "version unreadable" in check.detail


def test_subtitles_filter_parses_filter_list(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "video_dubber.doctor.subprocess.run",
        _fake_run_factory(
            {
                "ffmpeg": "Filters:\n"
                " .. ass               V->V       Render ASS subtitles\n"
                " .. subtitles         V->V       Render text subtitles\n"
            }
        ),
    )
    assert _check_subtitles_filter().status == "pass"


def test_subtitles_filter_missing_fails(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "video_dubber.doctor.subprocess.run",
        _fake_run_factory({"ffmpeg": "Filters:\n .. scale V->V Scale the input video\n"}),
    )
    check = _check_subtitles_filter()
    assert check.status == "fail"
    assert "no subtitles filter" in check.detail


def test_subtitles_filter_skipped_without_ffmpeg(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: None)
    check = _check_subtitles_filter()
    assert check.status == "fail"
    assert "without ffmpeg" in check.detail


def test_disk_space_pass_and_warn(monkeypatch):
    monkeypatch.setattr(
        "video_dubber.doctor.shutil.disk_usage",
        lambda path: _FakeUsage(free=5 * 1024**3),
    )
    assert _check_disk_space("/tmp").status == "pass"
    monkeypatch.setattr(
        "video_dubber.doctor.shutil.disk_usage",
        lambda path: _FakeUsage(free=int(0.2 * 1024**3)),
    )
    check = _check_disk_space("/tmp")
    assert check.status == "warn"
    assert "0.2 GiB free" in check.detail
    with pytest.raises(DoctorError, match="must be positive"):
        _check_disk_space("/tmp", minimum_gib=0)


def test_disk_space_unmeasurable_warns(monkeypatch):
    def boom(path):
        raise OSError("no such directory")

    monkeypatch.setattr("video_dubber.doctor.shutil.disk_usage", boom)
    check = _check_disk_space("/nope")
    assert check.status == "warn"
    assert "cannot measure" in check.detail


def test_report_verdict_rules():
    ok = DoctorReport(checks=[DoctorCheck(name="a", status="pass", detail="fine")])
    assert ok.verdict == "ok"
    warnings = DoctorReport(
        checks=[
            DoctorCheck(name="a", status="pass", detail="fine"),
            DoctorCheck(name="b", status="warn", detail="missing", required=False),
        ]
    )
    assert warnings.verdict == "ok_with_warnings"
    # An optional failure is a warning, not a failure.
    optional_fail = DoctorReport(
        checks=[DoctorCheck(name="b", status="fail", detail="missing", required=False)]
    )
    assert optional_fail.verdict == "ok_with_warnings"
    required_fail = DoctorReport(checks=[DoctorCheck(name="a", status="fail", detail="missing")])
    assert required_fail.verdict == "fail"
    assert required_fail.failed_required[0].name == "a"


def test_report_round_trip_and_markdown():
    report = DoctorReport(
        checks=[
            DoctorCheck(name="ffmpeg", status="pass", detail="ffmpeg version 6.1"),
            DoctorCheck(name="whisper", status="warn", detail="not found", required=False),
        ]
    )
    restored = DoctorReport.from_json(report.to_json())
    assert restored.verdict == "ok_with_warnings"
    assert [c.name for c in restored.checks] == ["ffmpeg", "whisper"]
    assert restored.checks[1].required is False
    markdown = report.to_markdown()
    assert "# Doctor: ok_with_warnings" in markdown
    assert "[✓] ffmpeg (required): ffmpeg version 6.1" in markdown
    assert "[~] whisper (optional): not found" in markdown
    with pytest.raises(DoctorError, match="invalid doctor report"):
        DoctorReport.from_json("{nope")


def test_run_doctor_all_missing_fails(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: None)
    monkeypatch.setattr(
        "video_dubber.doctor.shutil.disk_usage",
        lambda path: _FakeUsage(free=10 * 1024**3),
    )
    report = run_doctor()
    assert report.verdict == "fail"
    assert {c.name for c in report.failed_required} == {
        "ffmpeg",
        "ffprobe",
        "ffmpeg subtitles filter",
    }
    # Optional tools warn, they never fail the run.
    assert all(
        c.status == "warn" for c in report.checks if c.name in ("whisper", "argos-translate")
    )
    assert "Fix the failed required checks" in report.to_markdown()


def test_run_doctor_honours_timeout(monkeypatch):
    monkeypatch.setattr("video_dubber.doctor._which", lambda cmd: f"/usr/bin/{cmd}")

    def slow_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=15)

    monkeypatch.setattr("video_dubber.doctor.subprocess.run", slow_run)
    # A hung ffmpeg -version is a pass (binary exists); the filter check
    # degrades to fail because it cannot list filters.
    report = run_doctor()
    ffmpeg_check = next(c for c in report.checks if c.name == "ffmpeg")
    assert ffmpeg_check.status == "pass"
    filter_check = next(c for c in report.checks if c.name == "ffmpeg subtitles filter")
    assert filter_check.status == "fail"
