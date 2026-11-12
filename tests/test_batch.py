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


# --- Runner -------------------------------------------------------------------


def _profile_file(tmp_path) -> str:
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    profile = SpeakerProfile(name="Asha", embedding=[0.25] * EMBEDDING_DIM, base_pitch_hz=180.0)
    path = tmp_path / "asha.json"
    path.write_text(profile.to_json())
    return str(path)


def _transcript_file(tmp_path, name: str, text: str, language: str = "en") -> str:
    from video_dubber.transcript import Segment, Transcript, Word

    # 0.4s per word = 150 wpm, the profile's speaking rate, so the dub
    # fits its windows without extreme stretching.
    words = [Word(text=w, start=i * 0.4, end=(i + 1) * 0.4) for i, w in enumerate(text.split())]
    transcript = Transcript(segments=[Segment(words=words)], language=language)
    path = tmp_path / name
    path.write_text(transcript.to_json())
    return str(path)


def test_run_job_transcript_job_produces_dub_and_report(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars good morning")
    job = BatchJob(
        name="clip1", target="hi", profile=_profile_file(tmp_path), transcript=transcript
    )
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "ok"
    assert result.verdict == "good"
    assert result.mos_proxy is not None and result.mos_proxy >= 4.0
    assert result.transcription_wer is None  # no reference given
    for key in ("dubbed_wav", "transcript", "translated", "report"):
        assert key in result.outputs
        assert result.outputs[key].endswith((".wav", ".json"))
    with open(result.outputs["translated"], encoding="utf-8") as fh:
        translated = json.loads(fh.read())
    assert "नमस्ते" in json.dumps(translated, ensure_ascii=False)


def test_run_job_enhance_flag_lands_in_report(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars good morning")
    job = BatchJob(
        name="clip1",
        target="hi",
        profile=_profile_file(tmp_path),
        transcript=transcript,
        enhance=True,
    )
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "ok"
    with open(result.outputs["report"], encoding="utf-8") as fh:
        report = json.loads(fh.read())
    assert report["enhancement"] is not None
    assert report["enhancement"]["gain_db"] != pytest.approx(0.0)
    # Timing preserved: the dub still spans the full transcript.
    from video_dubber.audio import load_wav

    assert load_wav(result.outputs["dubbed_wav"]).duration == pytest.approx(1.6)


def test_run_job_wer_against_reference(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars")
    reference = _transcript_file(tmp_path, "ref.json", "hello venus")
    job = BatchJob(
        name="clip1",
        target="hi",
        profile=_profile_file(tmp_path),
        transcript=transcript,
        reference=reference,
    )
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "ok"
    assert result.transcription_wer == pytest.approx(0.5)


def test_run_job_skips_when_output_exists(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars")
    job = BatchJob(
        name="clip1", target="hi", profile=_profile_file(tmp_path), transcript=transcript
    )
    out = str(tmp_path / "out")
    assert run_job(job, out).status == "ok"
    assert run_job(job, out).status == "skipped"
    assert run_job(job, out, force=True).status == "ok"


def test_run_job_failure_is_a_result_not_an_exception(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars")
    job = BatchJob(
        name="clip1",
        target="hi",
        profile=_profile_file(tmp_path),
        transcript=transcript,
    )
    # Corrupt the profile after validation so the run itself fails.
    with open(job.profile, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "failed"
    assert result.error


def test_run_batch_continues_after_failure_by_default(tmp_path):
    from video_dubber.batch import BatchJobResult, run_batch

    calls: list[str] = []

    def stub(job, out_dir, force=False):
        calls.append(job.name)
        if job.name == "bad":
            return BatchJobResult(name=job.name, status="failed", error="boom")
        return BatchJobResult(name=job.name, status="ok")

    manifest = BatchManifest(
        jobs=[
            BatchJob(name="good1", target="hi", profile="p.json", transcript="t.json"),
            BatchJob(name="bad", target="hi", profile="p.json", transcript="t.json"),
            BatchJob(name="good2", target="hi", profile="p.json", transcript="t.json"),
        ]
    )
    results = run_batch(manifest, str(tmp_path), runner=stub)
    assert [r.status for r in results] == ["ok", "failed", "ok"]
    assert calls == ["good1", "bad", "good2"]


def test_run_batch_stop_on_error(tmp_path):
    from video_dubber.batch import BatchJobResult, run_batch

    def stub(job, out_dir, force=False):
        return BatchJobResult(
            name=job.name,
            status="failed" if job.name == "bad" else "ok",
            error="boom" if job.name == "bad" else None,
        )

    manifest = BatchManifest(
        jobs=[
            BatchJob(name="bad", target="hi", profile="p.json", transcript="t.json"),
            BatchJob(name="good", target="hi", profile="p.json", transcript="t.json"),
        ]
    )
    results = run_batch(manifest, str(tmp_path), stop_on_error=True, runner=stub)
    assert [r.name for r in results] == ["bad"]


# --- Summary ------------------------------------------------------------------


def test_batch_summary_counts_and_means():
    from video_dubber.batch import BatchJobResult, BatchSummary

    summary = BatchSummary(
        results=[
            BatchJobResult(
                name="a", status="ok", verdict="good", mos_proxy=4.8, transcription_wer=0.1
            ),
            BatchJobResult(name="b", status="ok", verdict="review", mos_proxy=3.0),
            BatchJobResult(name="c", status="failed", error="boom"),
            BatchJobResult(name="d", status="skipped"),
        ]
    )
    assert (summary.total, summary.succeeded, summary.failed, summary.skipped) == (4, 2, 1, 1)
    assert summary.mean_wer == pytest.approx(0.1)  # only jobs with a WER count
    assert summary.mean_mos == pytest.approx(3.9)
    data = json.loads(summary.to_json())
    assert data["total"] == 4 and data["mean_mos_proxy"] == 3.9
    md = summary.to_markdown()
    assert "2 ok, 1 failed, 1 skipped" in md
    assert "| c | failed |" in md and "boom" in md


def test_batch_summary_empty_means_are_none():
    from video_dubber.batch import BatchJobResult, BatchSummary

    summary = BatchSummary(results=[BatchJobResult(name="a", status="skipped")])
    assert summary.mean_wer is None
    assert summary.mean_mos is None
    assert json.loads(summary.to_json())["mean_mos_proxy"] is None


def test_summarize_writes_report_files(tmp_path):
    from video_dubber.batch import BatchJobResult, summarize

    out = tmp_path / "out"
    summary = summarize(
        [BatchJobResult(name="a", status="ok", verdict="good", mos_proxy=5.0)],
        str(out),
    )
    assert summary.succeeded == 1
    assert (out / "batch-report.json").exists()
    assert (out / "batch-report.md").exists()
    assert "Batch dub report" in (out / "batch-report.md").read_text(encoding="utf-8")
    assert (out / "batch-report.html").exists()
    assert "<!DOCTYPE html>" in (out / "batch-report.html").read_text(encoding="utf-8")
    assert "Batch dub report" in (out / "batch-report.html").read_text(encoding="utf-8")


def test_manifest_resolves_profiles_paths(tmp_path):
    ravi = tmp_path / "ravi.json"
    ravi.write_text("{}")
    manifest = _manifest(
        tmp_path,
        [_job("clip1", profiles={"Ravi": "ravi.json"})],
    )
    assert manifest.jobs[0].profiles == {"Ravi": str(tmp_path / "ravi.json")}


def test_manifest_rejects_missing_profile_file(tmp_path):
    with pytest.raises(BatchError, match="profile for speaker 'Ravi' not found"):
        _manifest(tmp_path, [_job("clip1", profiles={"Ravi": "missing.json"})])


def test_job_rejects_non_dict_profiles():
    with pytest.raises(BatchError, match="'profiles' must be a speaker->path object"):
        BatchJob(name="x", target="hi", profile="p.json", transcript="t.json", profiles=["a"])


def test_manifest_accepts_bed_level_and_defaults(tmp_path):
    manifest = _manifest(tmp_path, [_job("clip1", bed_level=0.3)])
    assert manifest.jobs[0].bed_level == 0.3
    manifest = _manifest(tmp_path, [_job("clip1")], defaults={"bed_level": 0.25})
    assert manifest.jobs[0].bed_level == 0.25


def test_job_rejects_bad_bed_level():
    for level in (1.5, -0.1):
        with pytest.raises(BatchError, match="'bed_level' must be between 0 and 1"):
            BatchJob(
                name="x",
                target="hi",
                profile="p.json",
                video="v.mp4",
                backend="scripted",
                script="w.json",
                bed_level=level,
            )


def test_job_rejects_bed_level_without_video():
    with pytest.raises(BatchError, match="'bed_level' needs a video job"):
        BatchJob(name="x", target="hi", profile="p.json", transcript="t.json", bed_level=0.3)


def test_run_job_writes_translated_subtitles(tmp_path):
    from video_dubber.batch import run_job

    transcript = _transcript_file(tmp_path, "src.json", "hello mars good morning")
    job = BatchJob(
        name="clip1", target="hi", profile=_profile_file(tmp_path), transcript=transcript
    )
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "ok"
    assert result.outputs["translated_srt"].endswith("translated.srt")
    assert result.outputs["translated_vtt"].endswith("translated.vtt")
    with open(result.outputs["translated_srt"], encoding="utf-8") as fh:
        srt = fh.read()
    assert "नमस्ते" in srt  # translated text, timed cues
    assert "-->" in srt
    with open(result.outputs["translated_vtt"], encoding="utf-8") as fh:
        assert fh.read().startswith("WEBVTT")


def test_run_job_with_speaker_profiles_uses_them(tmp_path):
    from video_dubber.batch import run_job
    from video_dubber.transcript import Segment, Transcript, Word
    from video_dubber.voice import EMBEDDING_DIM, SpeakerProfile

    # Two speakers in the source transcript; each gets their own voice.
    words = [
        Word(text="hello", start=0.0, end=0.4),
        Word(text="mars", start=0.4, end=0.8),
        Word(text="good", start=1.0, end=1.4),
        Word(text="morning", start=1.4, end=1.8),
    ]
    transcript = Transcript(
        segments=[
            Segment(words=words[:2], speaker="Asha"),
            Segment(words=words[2:], speaker="Ravi"),
        ],
        language="en",
    )
    src = tmp_path / "src.json"
    src.write_text(transcript.to_json())
    ravi = SpeakerProfile(name="Ravi", embedding=[0.1] * EMBEDDING_DIM, base_pitch_hz=110.0)
    ravi_path = tmp_path / "ravi.json"
    ravi_path.write_text(ravi.to_json())
    job = BatchJob(
        name="clip1",
        target="hi",
        profile=_profile_file(tmp_path),
        transcript=str(src),
        profiles={"Ravi": str(ravi_path)},
    )
    result = run_job(job, str(tmp_path / "out"))
    assert result.status == "ok"
    with open(result.outputs["report"], encoding="utf-8") as fh:
        report = json.loads(fh.read())
    assert report["speakers_dubbed"] == 2


def test_shipped_multispeaker_manifest_runs_end_to_end(tmp_path):
    """The examples/multispeaker manifest is a working, runnable example."""
    import os

    from video_dubber.batch import BatchManifest, run_job

    manifest = BatchManifest.from_json_file("examples/multispeaker/manifest.json")
    assert len(manifest.jobs) == 1
    assert manifest.jobs[0].profiles == {"Ravi": os.path.abspath("examples/multispeaker/ravi.json")}
    result = run_job(manifest.jobs[0], str(tmp_path / "out"))
    assert result.status == "ok"
    assert result.outputs["translated_srt"].endswith("translated.srt")
    with open(result.outputs["report"], encoding="utf-8") as fh:
        assert json.loads(fh.read())["speakers_dubbed"] == 2


def test_manifest_accepts_style_and_defaults(tmp_path):
    manifest = _manifest(tmp_path, [_job("clip1", style="energetic")])
    assert manifest.jobs[0].style == "energetic"
    manifest = _manifest(tmp_path, [_job("clip1")], defaults={"style": "narration"})
    assert manifest.jobs[0].style == "narration"
    manifest = _manifest(
        tmp_path, [_job("clip1", style="dramatic")], defaults={"style": "narration"}
    )
    assert manifest.jobs[0].style == "dramatic"  # job wins over defaults


def test_job_rejects_unknown_style():
    with pytest.raises(BatchError, match="unknown dub style 'shouty'"):
        BatchJob(name="x", target="hi", profile="p.json", video="v.mp4", style="shouty")
