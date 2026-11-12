"""Tests for project status: grading jobs from outputs on disk."""

import json
import time
from pathlib import Path

import pytest

from video_dubber.batch import BatchManifest, run_batch
from video_dubber.init import scaffold_project
from video_dubber.status import StatusError, project_status


def _scaffold(tmp_path: Path) -> Path:
    root = tmp_path / "my-dub"
    scaffold_project(root)
    return root


class TestProjectStatus:
    def test_fresh_project_is_not_started(self, tmp_path):
        root = _scaffold(tmp_path)
        status = project_status(root)
        assert status.has_config is True
        assert status.config_error is None
        assert status.target_language == "hi"
        assert status.has_manifest is True
        assert status.has_library is True
        assert [(job.name, job.state) for job in status.jobs] == [("example", "not started")]
        assert status.summary == "0/1 jobs done (1 not started)"

    def test_done_after_batch_with_verdict(self, tmp_path, monkeypatch):
        root = _scaffold(tmp_path)
        monkeypatch.chdir(root)
        run_batch(BatchManifest.from_json_file("manifest.json"), out_dir="out")
        status = project_status(root)
        (job,) = status.jobs
        assert job.state == "done"
        assert job.verdict == "review"
        assert job.mos_proxy == 3.25
        assert status.summary == "1/1 jobs done (1 done)"

    def test_stale_when_source_changes(self, tmp_path, monkeypatch):
        root = _scaffold(tmp_path)
        monkeypatch.chdir(root)
        run_batch(BatchManifest.from_json_file("manifest.json"), out_dir="out")
        time.sleep(0.02)
        (root / "source" / "example.wav").touch()
        status = project_status(root)
        (job,) = status.jobs
        assert job.state == "stale"
        assert "source changed" in (job.note or "")

    def test_partial_when_output_dir_has_no_dub(self, tmp_path):
        root = _scaffold(tmp_path)
        (root / "out" / "example").mkdir(parents=True)
        (root / "out" / "example" / "transcript.json").write_text("{}\n")
        status = project_status(root)
        (job,) = status.jobs
        assert job.state == "partial"
        assert "no dubbed.wav" in (job.note or "")

    def test_broken_manifest_reported_not_raised(self, tmp_path):
        root = _scaffold(tmp_path)
        (root / "manifest.json").write_text("{not json", encoding="utf-8")
        status = project_status(root)
        assert status.has_manifest is False
        assert "not valid JSON" in (status.manifest_error or "")
        assert status.jobs == []

    def test_broken_config_reported_not_raised(self, tmp_path):
        root = _scaffold(tmp_path)
        (root / "dubber.toml").write_text("[dubber\n", encoding="utf-8")
        status = project_status(root)
        assert status.has_config is False
        assert status.config_error is not None
        assert status.has_library is True  # voices.json still found

    def test_missing_files_reported(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        status = project_status(root)
        assert status.has_config is False
        assert status.has_manifest is False
        assert status.has_library is False
        assert status.summary == "0/0 jobs done"

    def test_not_a_directory(self, tmp_path):
        with pytest.raises(StatusError, match="not a directory"):
            project_status(tmp_path / "nope")

    def test_json_round_trip(self, tmp_path, monkeypatch):
        root = _scaffold(tmp_path)
        monkeypatch.chdir(root)
        run_batch(BatchManifest.from_json_file("manifest.json"), out_dir="out")
        data = json.loads(json.dumps(project_status(root).to_dict()))
        assert data["jobs"][0]["state"] == "done"
        assert data["jobs"][0]["verdict"] == "review"
        assert data["summary"].startswith("1/1")

    def test_markdown_lists_every_job(self, tmp_path):
        root = _scaffold(tmp_path)
        text = project_status(root).to_markdown()
        assert "# Project status:" in text
        assert "example (hi): not started" in text
        assert "Voice library: present" in text


class TestProjectStatusWithSource:
    def test_seeded_job_graded_too(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video")
        root = tmp_path / "my-dub"
        scaffold_project(root, source=video)
        status = project_status(root)
        assert [(job.name, job.state) for job in status.jobs] == [
            ("example", "not started"),
            ("clip", "not started"),
        ]
        assert status.summary == "0/2 jobs done (2 not started)"
