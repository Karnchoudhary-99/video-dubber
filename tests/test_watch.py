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
