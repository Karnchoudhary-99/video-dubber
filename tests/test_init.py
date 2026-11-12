"""Tests for project scaffolding: ``dubber init`` writes a valid project."""

import json
from pathlib import Path

import pytest

from video_dubber.batch import BatchManifest, run_batch
from video_dubber.config import load_config
from video_dubber.init import InitError, scaffold_project
from video_dubber.voice import SpeakerProfile, VoiceLibrary

EXPECTED_FILES = {
    ".gitignore",
    "README.md",
    "dubber.toml",
    "manifest.json",
    "source/example-script.json",
    "source/example.wav",
    "voices.json",
    "voices/narrator.json",
}


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


class TestScaffold:
    def test_creates_every_file(self, tmp_path):
        root = tmp_path / "my-dub"
        written = scaffold_project(root)
        assert _files(root) == EXPECTED_FILES
        assert {str(p.relative_to(root)) for p in written} == EXPECTED_FILES
        assert (root / "out").is_dir()

    def test_config_loads_with_chosen_values(self, tmp_path):
        root = tmp_path / "my-dub"
        scaffold_project(
            root,
            source_language="en",
            target_language="es",
            backend="argos",
            style="documentary",
        )
        config = load_config(root / "dubber.toml")
        assert config.source_language == "en"
        assert config.target_language == "es"
        assert config.backend == "argos"
        assert config.voice_library == "voices.json"
        assert config.style == "documentary"
        assert config.enhance.target == -16.0

    def test_manifest_loads(self, tmp_path):
        root = tmp_path / "my-dub"
        scaffold_project(root)
        manifest = BatchManifest.from_json_file(str(root / "manifest.json"))
        assert len(manifest.jobs) == 1
        job = manifest.jobs[0]
        assert job.name == "example"
        assert job.backend == "scripted"
        assert job.target == "hi"
        assert job.mux is False

    def test_library_and_profile_load(self, tmp_path):
        root = tmp_path / "my-dub"
        scaffold_project(root)
        library = VoiceLibrary.from_json_file(str(root / "voices.json"))
        assert library.names() == ["narrator"]
        profile = SpeakerProfile.from_json(
            (root / "voices/narrator.json").read_text(encoding="utf-8")
        )
        assert profile.name == "narrator"
        assert profile.base_pitch_hz == 180.0
        assert len(profile.embedding) == 16
        # The library profile and the standalone profile are the same voice.
        assert library.get("narrator").embedding == profile.embedding

    def test_example_job_runs_offline(self, tmp_path, monkeypatch):
        root = tmp_path / "my-dub"
        scaffold_project(root)
        monkeypatch.chdir(root)
        manifest = BatchManifest.from_json_file("manifest.json")
        (summary,) = run_batch(manifest, out_dir="out")
        assert summary.status == "ok"
        assert summary.error is None
        assert (root / "out" / "example" / "dubbed.wav").is_file()
        assert (root / "out" / "example" / "report.json").is_file()

    def test_readme_names_the_project(self, tmp_path):
        root = tmp_path / "my-dub"
        scaffold_project(root)
        text = (root / "README.md").read_text(encoding="utf-8")
        assert "# my-dub" in text
        assert "dubber batch manifest.json" in text


class TestScaffoldRefusals:
    def test_refuses_nonempty_directory(self, tmp_path):
        root = tmp_path / "my-dub"
        root.mkdir()
        (root / "notes.txt").write_text("mine\n", encoding="utf-8")
        with pytest.raises(InitError, match="not empty"):
            scaffold_project(root)
        assert not (root / "manifest.json").exists()

    def test_force_overwrites_owned_files_only(self, tmp_path):
        root = tmp_path / "my-dub"
        root.mkdir()
        (root / "notes.txt").write_text("mine\n", encoding="utf-8")
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        scaffold_project(root, force=True)
        assert (root / "notes.txt").read_text(encoding="utf-8") == "mine\n"
        manifest = BatchManifest.from_json_file(str(root / "manifest.json"))
        assert manifest.jobs[0].name == "example"

    def test_refuses_file_as_root(self, tmp_path):
        root = tmp_path / "my-dub"
        root.write_text("not a dir\n", encoding="utf-8")
        with pytest.raises(InitError, match="not a directory"):
            scaffold_project(root)

    def test_rejects_bad_language(self, tmp_path):
        with pytest.raises(InitError, match="two-letter"):
            scaffold_project(tmp_path / "x", target_language="hindi")
        with pytest.raises(InitError, match="two-letter"):
            scaffold_project(tmp_path / "x", source_language="EN")

    def test_rejects_bad_backend(self, tmp_path):
        with pytest.raises(InitError, match="unknown translation backend"):
            scaffold_project(tmp_path / "x", backend="whisper")

    def test_rejects_bad_style(self, tmp_path):
        with pytest.raises(InitError, match="unknown dub style"):
            scaffold_project(tmp_path / "x", style="operatic")

    def test_missing_source_file(self, tmp_path):
        with pytest.raises(InitError, match="source file not found"):
            scaffold_project(tmp_path / "x", source=tmp_path / "nope.mp4")


class TestScaffoldWithSource:
    def test_seeds_a_whisper_job(self, tmp_path):
        video = tmp_path / "My Holiday Video.mp4"
        video.write_bytes(b"fake video")
        root = tmp_path / "my-dub"
        scaffold_project(root, source=video)
        manifest = BatchManifest.from_json_file(str(root / "manifest.json"))
        assert [job.name for job in manifest.jobs] == [
            "example",
            "my-holiday-video",
        ]
        seeded = manifest.jobs[1]
        assert seeded.backend == "whisper"
        assert seeded.video.endswith("My Holiday Video.mp4")
        text = (root / "README.md").read_text(encoding="utf-8")
        assert "my-holiday-video" in text

    def test_source_inside_project_uses_relative_path(self, tmp_path):
        root = tmp_path / "my-dub"
        root.mkdir()
        video = root / "clip.mp4"
        video.write_bytes(b"fake video")
        scaffold_project(root, source=video, force=True)
        raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert raw["jobs"][1]["video"] == "clip.mp4"
