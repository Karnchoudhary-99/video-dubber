"""Tests for the batch manifest model (parsing, defaults, validation)."""

import json

import pytest

from video_dubber.batch import BatchError, BatchJob, BatchManifest


def _manifest(tmp_path, jobs, defaults=None):
    """Write a manifest whose referenced files exist, and load it."""
    profile = tmp_path / "asha.json"
    profile.write_text("{}")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    script = tmp_path / "words.json"
    script.write_text("[]")
    manifest = {"jobs": jobs}
    if defaults is not None:
        manifest["defaults"] = defaults
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return BatchManifest.from_json_file(str(path))


def _job(name="clip1", **overrides):
    base = {
        "name": name,
        "target": "hi",
        "profile": "asha.json",
        "video": "clip.mp4",
        "backend": "scripted",
        "script": "words.json",
    }
    base.update(overrides)
    return base


def test_manifest_merges_defaults_and_resolves_paths(tmp_path):
    manifest = _manifest(
        tmp_path,
        [_job("clip1"), _job("clip2", target="es")],
        defaults={"target": "hi", "profile": "asha.json"},
    )
    assert [j.name for j in manifest.jobs] == ["clip1", "clip2"]
    assert manifest.jobs[0].target == "hi"
    assert manifest.jobs[1].target == "es"  # job wins over defaults
    assert manifest.jobs[0].profile == str(tmp_path / "asha.json")
    assert manifest.jobs[0].video == str(tmp_path / "clip.mp4")


def test_manifest_rejects_duplicate_names(tmp_path):
    with pytest.raises(BatchError, match="duplicate job name"):
        _manifest(tmp_path, [_job("clip1"), _job("clip1")])


def test_manifest_rejects_unknown_keys(tmp_path):
    with pytest.raises(BatchError, match="unknown keys"):
        _manifest(tmp_path, [_job("clip1", frobnicate=True)])
    with pytest.raises(BatchError, match="unknown top-level keys"):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"jobs": [_job("clip1")], "nope": 1}))
        BatchManifest.from_json_file(str(path))


def test_manifest_rejects_empty_jobs(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"jobs": []}))
    with pytest.raises(BatchError, match="non-empty 'jobs'"):
        BatchManifest.from_json_file(str(path))


def test_manifest_rejects_invalid_json():
    with pytest.raises(BatchError, match="not valid JSON"):
        BatchManifest.from_json("{not json")


def test_job_validation_rules():
    with pytest.raises(BatchError, match="non-empty name"):
        BatchJob(name="", target="hi", profile="p.json", video="v.mp4")
    with pytest.raises(BatchError, match="target language"):
        BatchJob(name="a", target="", profile="p.json", video="v.mp4")
    with pytest.raises(BatchError, match="unknown backend"):
        BatchJob(name="a", target="hi", profile="p.json", video="v.mp4", backend="nope")
    with pytest.raises(BatchError, match="exactly one of"):
        BatchJob(name="a", target="hi", profile="p.json")
    with pytest.raises(BatchError, match="exactly one of"):
        BatchJob(name="a", target="hi", profile="p.json", video="v.mp4", transcript="t.json")
    with pytest.raises(BatchError, match="needs a 'script'"):
        BatchJob(name="a", target="hi", profile="p.json", video="v.mp4", backend="scripted")
    with pytest.raises(BatchError, match="max_chars must be positive"):
        BatchJob(
            name="a",
            target="hi",
            profile="p.json",
            video="v.mp4",
            backend="whisper",
            max_chars=0,
        )


def test_manifest_rejects_missing_files(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"fake")
    (tmp_path / "words.json").write_text("[]")
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({"jobs": [_job("clip1", profile="ghost.json", video="clip.mp4")]}))
    with pytest.raises(BatchError, match="profile file not found"):
        BatchManifest.from_json_file(str(path))


def test_manifest_transcript_job_needs_no_script(tmp_path):
    transcript = tmp_path / "t.json"
    transcript.write_text("{}")
    manifest = _manifest(
        tmp_path,
        [{"name": "t1", "target": "hi", "profile": "asha.json", "transcript": "t.json"}],
    )
    assert manifest.jobs[0].transcript == str(transcript)
    assert manifest.jobs[0].backend == "whisper"
