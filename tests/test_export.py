import hashlib
import zipfile

import pytest

from video_dubber.assemble import DubReport
from video_dubber.audio import make_silence, make_sine, save_wav
from video_dubber.export import (
    ExportError,
    ExportFile,
    ExportManifest,
    build_manifest,
    export_bundle,
    export_zip,
    hash_file,
    verify_bundle,
)


def _wav(path, seconds=0.5):
    save_wav(str(path), make_sine(440.0, seconds))


def _report(path, verdict="good", mos=4.5):
    report = DubReport(
        segments_dubbed=2,
        total_duration=1.0,
        mean_sync_drift=0.02,
        max_sync_drift=0.05,
        segments_over_drift=0,
        mean_fit_factor=1.0,
        max_fit_factor=1.1,
        extreme_fits=0,
        loudness_error_db=0.3,
        translation_coverage=0.9,
        mos_proxy=mos,
        verdict=verdict,
    )
    path.write_text(report.to_json(), encoding="utf-8")
    return report


def test_hash_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello dub")
    assert hash_file(path) == hashlib.sha256(b"hello dub").hexdigest()


def test_manifest_json_round_trip():
    manifest = ExportManifest(
        name="clip.hi",
        created_utc="2026-11-11T13:05:00+00:00",
        dubber_version="0.1.0",
        source_language="en",
        target_language="hi",
        source_file="clip.mp4",
        verdict="review",
        mos_proxy=3.53,
        speakers=["Asha"],
        files=[ExportFile("dubbed_audio", "clip.hi.wav", "a" * 64, 1234, "clip.wav")],
    )
    rebuilt = ExportManifest.from_json(manifest.to_json())
    assert rebuilt == manifest
    assert rebuilt.files[0].role == "dubbed_audio"


def test_manifest_markdown_names_files_and_verdict():
    manifest = ExportManifest(
        name="clip.hi",
        created_utc="2026-11-11T13:05:00+00:00",
        dubber_version="0.1.0",
        source_language="en",
        target_language="hi",
        verdict="review",
        mos_proxy=3.53,
        files=[ExportFile("dubbed_audio", "clip.hi.wav", "b" * 64, 10)],
    )
    text = manifest.to_markdown()
    assert "clip.hi" in text
    assert "review" in text
    assert "clip.hi.wav" in text
    assert "en -> hi" in text


def test_manifest_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="not valid JSON"):
        ExportManifest.from_json("{nope")
    with pytest.raises(ValueError, match="not a valid export manifest"):
        ExportManifest.from_dict({"name": "x"})
    with pytest.raises(ValueError, match="not a valid export file entry"):
        ExportFile.from_dict({"role": "dubbed_audio"})


def test_build_manifest_reads_report_and_hashes_files(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    srt = tmp_path / "dub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,500\nNamaste\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    _report(report_path, verdict="review", mos=3.53)
    manifest, sources = build_manifest(
        "clip.hi",
        dub_audio=dub,
        target="hi",
        subtitles_srt=srt,
        report=report_path,
        speakers=["Asha"],
        created_utc="2026-11-11T13:05:00+00:00",
    )
    assert manifest.verdict == "review"
    assert manifest.mos_proxy == 3.53
    assert manifest.speakers == ["Asha"]
    assert manifest.dubber_version == "0.1.0"
    assert [entry.role for entry in manifest.files] == [
        "dubbed_audio",
        "subtitles_srt",
        "quality_report",
    ]
    audio_entry = manifest.files[0]
    assert audio_entry.sha256 == hash_file(dub)
    assert audio_entry.size_bytes == dub.stat().st_size
    assert sources["dubbed_audio"] == dub


def test_build_manifest_rejects_missing_files_and_bad_names(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    with pytest.raises(ExportError, match="not found"):
        build_manifest("clip.hi", dub_audio=tmp_path / "gone.wav", target="hi")
    with pytest.raises(ExportError, match="not found"):
        build_manifest("clip.hi", dub_audio=dub, target="hi", subtitles_srt=tmp_path / "gone.srt")
    with pytest.raises(ExportError, match="must not be empty"):
        build_manifest("", dub_audio=dub, target="hi")
    with pytest.raises(ExportError, match="must not be empty"):
        build_manifest("clip.hi", dub_audio=dub, target="  ")


def test_build_manifest_prefixes_colliding_basenames(tmp_path):
    # The dub and the original are both often called clip.wav.
    original = tmp_path / "clip.wav"
    _wav(original)
    dubbed = tmp_path / "dubbed" / "clip.wav"
    dubbed.parent.mkdir()
    save_wav(str(dubbed), make_silence(0.25))
    manifest, _ = build_manifest("clip.hi", dub_audio=dubbed, target="hi", original=original)
    names = [entry.filename for entry in manifest.files]
    assert names == ["clip.wav", "original_clip.wav"]
    assert manifest.source_file == "clip.wav"


def test_export_bundle_copies_files_and_verifies(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    report_path = tmp_path / "report.json"
    _report(report_path)
    manifest, sources = build_manifest("clip.hi", dub_audio=dub, target="hi", report=report_path)
    out = tmp_path / "bundle"
    manifest_path = export_bundle(manifest, sources, out)
    assert manifest_path == out / "manifest.json"
    assert (out / "dub.wav").is_file()
    assert (out / "report.json").is_file()
    assert (out / "README.md").is_file()
    assert verify_bundle(out) == []
    # The manifest inside the bundle loads back identically.
    assert ExportManifest.from_json((out / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_verify_bundle_reports_tamper_missing_and_extra(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    manifest, sources = build_manifest("clip.hi", dub_audio=dub, target="hi")
    out = tmp_path / "bundle"
    export_bundle(manifest, sources, out)
    assert verify_bundle(out) == []

    (out / "dub.wav").write_bytes(b"tampered")
    problems = verify_bundle(out)
    assert any(p.startswith("changed: dub.wav") for p in problems)

    (out / "dub.wav").unlink()
    problems = verify_bundle(out)
    assert any(p.startswith("missing: dub.wav") for p in problems)

    _wav(out / "dub.wav")
    (out / "stray.txt").write_text("not in the manifest", encoding="utf-8")
    problems = verify_bundle(out)
    assert problems == ["unlisted file in bundle: stray.txt"]


def test_verify_bundle_without_manifest(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    assert verify_bundle(out) == [f"no manifest.json in {out}"]
    (out / "manifest.json").write_text("{broken", encoding="utf-8")
    assert verify_bundle(out)[0].startswith("manifest.json is invalid")


def test_export_bundle_refuses_nonempty_dir_without_force(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    manifest, sources = build_manifest("clip.hi", dub_audio=dub, target="hi")
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "keep.txt").write_text("pre-existing", encoding="utf-8")
    with pytest.raises(ExportError, match="not empty"):
        export_bundle(manifest, sources, out)
    export_bundle(manifest, sources, out, overwrite=True)
    assert verify_bundle(out) == ["unlisted file in bundle: keep.txt"]
    not_a_dir = tmp_path / "a-file"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(ExportError, match="not a directory"):
        export_bundle(manifest, sources, not_a_dir)


def test_export_zip_round_trip(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    srt = tmp_path / "dub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,500\nNamaste\n", encoding="utf-8")
    manifest, sources = build_manifest("clip.hi", dub_audio=dub, target="hi", subtitles_srt=srt)
    zipped = export_zip(manifest, sources, tmp_path / "clip.hi.zip")
    assert zipped.is_file()
    with zipfile.ZipFile(zipped) as zf:
        assert sorted(zf.namelist()) == [
            "README.md",
            "dub.srt",
            "dub.wav",
            "manifest.json",
        ]
        loaded = ExportManifest.from_json(zf.read("manifest.json").decode("utf-8"))
        assert loaded == manifest
        assert hashlib.sha256(zf.read("dub.wav")).hexdigest() == hash_file(dub)
    with pytest.raises(ExportError, match="already exists"):
        export_zip(manifest, sources, tmp_path / "clip.hi.zip")
    export_zip(manifest, sources, tmp_path / "clip.hi.zip", overwrite=True)
    with pytest.raises(ExportError, match="must end in .zip"):
        export_zip(manifest, sources, tmp_path / "clip.hi.tar")


def test_export_zip_verifies_after_extraction(tmp_path):
    dub = tmp_path / "dub.wav"
    _wav(dub)
    manifest, sources = build_manifest("clip.hi", dub_audio=dub, target="hi")
    zipped = export_zip(manifest, sources, tmp_path / "clip.hi.zip")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(zipped) as zf:
        zf.extractall(extracted)
    assert verify_bundle(extracted) == []


# --- CLI ---------------------------------------------------------------


def test_cli_export_dir_and_verify(tmp_path, capsys):
    from video_dubber.cli import main

    dub = tmp_path / "dub.wav"
    _wav(dub)
    srt = tmp_path / "dub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,500\nNamaste\n", encoding="utf-8")
    report_path = tmp_path / "report.json"
    _report(report_path, verdict="good", mos=4.5)
    out = tmp_path / "bundle"
    assert (
        main(
            [
                "export",
                str(dub),
                "--target",
                "hi",
                "--subtitles",
                str(srt),
                "--report",
                str(report_path),
                "--speakers",
                "Asha, Ravi",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "dub.hi" in printed  # default bundle name: stem + target
    assert "Asha, Ravi" in printed
    assert (out / "manifest.json").is_file()
    assert main(["export", "--verify", str(out)]) == 0
    assert "bundle intact" in capsys.readouterr().out


def test_cli_export_zip_and_verify_failure(tmp_path, capsys):
    from video_dubber.cli import main

    dub = tmp_path / "dub.wav"
    _wav(dub)
    zipped = tmp_path / "clip.hi.zip"
    assert main(["export", str(dub), "--target", "hi", "--out", str(zipped)]) == 0
    assert zipped.is_file()
    # A zip is not a directory bundle: --verify reports the missing manifest.
    assert main(["export", "--verify", str(zipped)]) == 1
    assert "no manifest.json" in capsys.readouterr().out


def test_cli_export_missing_args_and_missing_file(tmp_path, capsys):
    from video_dubber.cli import main

    dub = tmp_path / "dub.wav"
    _wav(dub)
    assert main(["export"]) == 1
    assert "needs a dubbed audio file" in capsys.readouterr().err
    assert main(["export", str(dub)]) == 1
    assert "needs --target" in capsys.readouterr().err
    assert main(["export", str(dub), "--target", "hi"]) == 1
    assert "needs --out" in capsys.readouterr().err
    assert (
        main(
            [
                "export",
                str(tmp_path / "gone.wav"),
                "--target",
                "hi",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        == 1
    )
    assert "not found" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["export", str(dub), "--target", "hi", "--out", "x", "--format", "tar"])
