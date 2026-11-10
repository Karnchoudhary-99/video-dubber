"""Tests for watch-folder scanning, state, and job naming."""

import json
import os
import time

import pytest

from video_dubber.watch import (
    WatchConfig,
    WatchError,
    WatchState,
    fingerprint_file,
    sanitize_job_name,
    scan_new_files,
)


def _watch_dir(tmp_path):
    """A watch dir with two media files, a text file, and a dotfile."""
    watch = tmp_path / "watch"
    watch.mkdir()
    (watch / "Clip One.mp4").write_bytes(b"fake video one")
    (watch / "clip-two.MKV").write_bytes(b"fake video two")
    (watch / "notes.txt").write_text("not media")
    (watch / ".hidden.mp4").write_bytes(b"hidden")
    return watch


def _config(tmp_path, watch, **overrides):
    profile = tmp_path / "asha.json"
    profile.write_text("{}")
    base = {
        "watch_dir": str(watch),
        "out_dir": str(tmp_path / "out"),
        "target": "hi",
        "profile": str(profile),
    }
    base.update(overrides)
    return WatchConfig(**base)


def test_sanitize_job_name_lowercases_and_dashes(tmp_path):
    taken: set[str] = set()
    assert sanitize_job_name("Clip One", taken) == "clip-one"
    assert sanitize_job_name("  spaced__out  ", taken) == "spaced-out"
    assert sanitize_job_name("Café au lait!", taken) == "caf-au-lait"


def test_sanitize_job_name_empty_becomes_clip_and_dedupes(tmp_path):
    taken: set[str] = set()
    assert sanitize_job_name("!!!", taken) == "clip"
    taken.add("clip")
    assert sanitize_job_name("???", taken) == "clip-2"
    taken.add("clip-2")
    assert sanitize_job_name("...", taken) == "clip-3"


def test_sanitize_job_name_dedupes_same_stem(tmp_path):
    taken = {"clip-one"}
    assert sanitize_job_name("Clip One", taken) == "clip-one-2"
    taken.add("clip-one-2")
    assert sanitize_job_name("clip one", taken) == "clip-one-3"


def test_fingerprint_changes_with_content_and_mtime(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"one")
    first = fingerprint_file(str(path))
    time.sleep(0.01)
    os.utime(path, ns=(first["mtime_ns"] + 1_000_000, first["mtime_ns"] + 1_000_000))
    second = fingerprint_file(str(path))
    assert second["mtime_ns"] != first["mtime_ns"]
    path.write_bytes(b"one more byte")
    third = fingerprint_file(str(path))
    assert third["size"] != first["size"]


def test_scan_finds_media_only_sorted(tmp_path):
    watch = _watch_dir(tmp_path)
    found = scan_new_files(str(watch), WatchState())
    names = [os.path.basename(path) for path, _, _ in found]
    assert names == ["Clip One.mp4", "clip-two.MKV"]
    assert all(changed is False for _, _, changed in found)


def test_scan_skips_processed_and_flags_changed(tmp_path):
    watch = _watch_dir(tmp_path)
    state = WatchState()
    first = scan_new_files(str(watch), state)
    for path, fingerprint, _ in first:
        state.mark_processed(os.path.relpath(path, str(watch)), fingerprint)
    assert scan_new_files(str(watch), state) == []
    # Editing a file in place brings it back, flagged as changed.
    (watch / "Clip One.mp4").write_bytes(b"fake video one, re-edited")
    second = scan_new_files(str(watch), state)
    assert len(second) == 1
    assert os.path.basename(second[0][0]) == "Clip One.mp4"
    assert second[0][2] is True


def test_scan_ignores_subdirectories(tmp_path):
    watch = _watch_dir(tmp_path)
    sub = watch / "sub"
    sub.mkdir()
    (sub / "nested.mp4").write_bytes(b"nested")
    names = [os.path.basename(p) for p, _, _ in scan_new_files(str(watch), WatchState())]
    assert "nested.mp4" not in names


def test_state_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "state.json"
    assert WatchState.load(str(path)).files == {}
    state = WatchState(files={"clip.mp4": {"size": 4, "mtime_ns": 7}})
    state.save(str(path))
    loaded = WatchState.load(str(path))
    assert loaded.is_processed("clip.mp4", {"size": 4, "mtime_ns": 7})
    assert not loaded.is_processed("clip.mp4", {"size": 5, "mtime_ns": 7})
    assert not loaded.is_processed("other.mp4", {"size": 4, "mtime_ns": 7})


def test_state_ignores_corrupt_file(tmp_path, capsys):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    assert WatchState.load(str(path)).files == {}
    assert "corrupt watch state" in capsys.readouterr().out
    path.write_text(json.dumps({"files": {"clip.mp4": "not a fingerprint"}}))
    assert WatchState.load(str(path)).files == {}


def test_config_rejects_bad_watch_dir_and_backend(tmp_path):
    with pytest.raises(WatchError, match="watch directory not found"):
        _config(tmp_path, tmp_path / "missing")
    watch = _watch_dir(tmp_path)
    with pytest.raises(WatchError, match="unknown backend"):
        _config(tmp_path, watch, backend="nope")
    with pytest.raises(WatchError, match="scripted needs a 'script'"):
        _config(tmp_path, watch, backend="scripted")
    with pytest.raises(WatchError, match="bed_level"):
        _config(tmp_path, watch, bed_level=1.5)
    with pytest.raises(WatchError, match="extensions must start"):
        _config(tmp_path, watch, extensions=("mp4",))
    with pytest.raises(WatchError, match="at least one file extension"):
        _config(tmp_path, watch, extensions=())


def test_config_default_state_path_is_under_out_dir(tmp_path):
    watch = _watch_dir(tmp_path)
    config = _config(tmp_path, watch)
    assert config.resolved_state_path == str(tmp_path / "out" / ".dubber-watch.json")
    custom = _config(tmp_path, watch, state_path=str(tmp_path / "elsewhere.json"))
    assert custom.resolved_state_path == str(tmp_path / "elsewhere.json")


# --- watch_once / watch_loop -------------------------------------------------

from video_dubber.batch import BatchJobResult
from video_dubber.watch import WatchScan, watch_loop, watch_once


class _StubRunner:
    """Fake pipeline: records calls, returns ok unless the name says fail."""

    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, job, out_dir, force=False):
        self.calls.append((job.name, out_dir, force))
        if "fail" in job.name:
            return BatchJobResult(name=job.name, status="failed", error="boom")
        return BatchJobResult(name=job.name, status="ok", verdict="good", mos_proxy=4.2)


def test_watch_once_dubs_new_files_and_remembers_them(tmp_path):
    watch = _watch_dir(tmp_path)
    config = _config(tmp_path, watch)
    runner = _StubRunner()
    scan = watch_once(config, runner=runner)
    assert scan.new_files == 2
    assert scan.succeeded == 2
    assert [name for name, _, _ in runner.calls] == ["clip-one", "clip-two"]
    assert all(force is False for _, _, force in runner.calls)
    state = WatchState.load(config.resolved_state_path)
    assert len(state.files) == 2
    # Second scan: nothing new, runner not called again.
    scan2 = watch_once(config, runner=runner)
    assert scan2.new_files == 0
    assert len(runner.calls) == 2


def test_watch_once_retries_failed_jobs_and_forces_changed_ones(tmp_path):
    watch = _watch_dir(tmp_path)
    (watch / "fail-me.mp4").write_bytes(b"will fail")
    config = _config(tmp_path, watch)
    runner = _StubRunner()
    scan = watch_once(config, runner=runner)
    assert scan.failed == 1
    assert scan.succeeded == 2
    state = WatchState.load(config.resolved_state_path)
    assert "fail-me.mp4" not in state.files
    # Failed job is retried on the next scan.
    scan2 = watch_once(config, runner=runner)
    assert scan2.new_files == 1
    assert [name for name, _, _ in runner.calls].count("fail-me") == 2
    # Editing a dubbed file re-dubs it with force=True.
    (watch / "Clip One.mp4").write_bytes(b"fake video one, re-edited")
    watch_once(config, runner=runner)
    forced = [name for name, _, force in runner.calls if force]
    assert forced == ["clip-one"]


def test_watch_once_uses_explicit_state_object(tmp_path):
    watch = _watch_dir(tmp_path)
    config = _config(tmp_path, watch)
    state = WatchState()
    watch_once(config, runner=_StubRunner(), state=state)
    assert len(state.files) == 2
    # The caller's state object is updated in place and also persisted,
    # so a later load sees the same processed files.
    assert WatchState.load(config.resolved_state_path).files == state.files


def test_watch_scan_report_formats(tmp_path):
    scan = WatchScan(
        results=[
            BatchJobResult(name="clip-one", status="ok", verdict="good", mos_proxy=4.2),
            BatchJobResult(name="fail-me", status="failed", error="boom"),
        ],
        new_files=2,
    )
    assert scan.total == 2
    assert scan.succeeded == 1
    assert scan.failed == 1
    report = json.loads(scan.to_json())
    assert report["new_files"] == 2
    assert report["jobs"][1]["error"] == "boom"
    markdown = scan.to_markdown()
    assert "| clip-one | ok | good | 4.2 |" in markdown
    assert "- **fail-me**: boom" in markdown


def test_watch_loop_picks_up_files_added_between_scans(tmp_path):
    watch = _watch_dir(tmp_path)
    config = _config(tmp_path, watch)
    runner = _StubRunner()
    seen: list[int] = []

    def on_scan(scan_no, scan):
        seen.append(scan.new_files)
        if scan_no == 1:
            (watch / "late-arrival.mp4").write_bytes(b"arrived late")

    sleeps: list[float] = []
    scans = watch_loop(
        config,
        interval_s=5.0,
        max_scans=2,
        runner=runner,
        sleep=sleeps.append,
        on_scan=on_scan,
    )
    assert [s.new_files for s in scans] == [2, 1]
    assert seen == [2, 1]
    assert sleeps == [5.0]  # slept once, between the two scans
    assert [name for name, _, _ in runner.calls] == [
        "clip-one",
        "clip-two",
        "late-arrival",
    ]


def test_watch_loop_rejects_bad_interval_and_scans(tmp_path):
    watch = _watch_dir(tmp_path)
    config = _config(tmp_path, watch)
    with pytest.raises(WatchError, match="interval must be positive"):
        watch_loop(config, interval_s=0, max_scans=1)
    with pytest.raises(WatchError, match="max_scans must be positive"):
        watch_loop(config, interval_s=1, max_scans=0)
