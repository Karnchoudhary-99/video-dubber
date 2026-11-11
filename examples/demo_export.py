"""Demo: export a finished dub as a portable bundle.

Runs the small end-to-end dub from ``demo_dub.py`` (translate to Hindi,
clone a scripted voice, render the dubbed track, build the quality
report), writes the retimed subtitles, then packs the whole thing twice:
once as a directory bundle and once as a ``.zip`` — and verifies both,
including a tampered copy that must fail verification.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_export.py``.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.assemble import build_report, dub_transcript
from video_dubber.audio import make_sine, save_wav
from video_dubber.export import (
    build_manifest,
    export_bundle,
    export_zip,
    verify_bundle,
)
from video_dubber.subtitles import retime_transcript
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.translate import DictionaryTranslator, translate_transcript
from video_dubber.voice import ScriptedVoiceCloner


def main() -> int:
    source = Transcript(
        segments=[
            Segment(
                words=[
                    Word("Hello", 0.0, 0.4),
                    Word("and", 0.4, 0.6),
                    Word("welcome", 0.6, 1.1),
                    Word("to", 1.1, 1.3),
                    Word("Mars", 1.3, 1.7),
                    Word("Cricket", 1.7, 2.2),
                ],
                speaker="commentator",
            ),
            Segment(
                words=[
                    Word("What", 2.5, 2.8),
                    Word("a", 2.8, 2.9),
                    Word("goal", 2.9, 3.3),
                    Word("by", 3.3, 3.5),
                    Word("the", 3.5, 3.6),
                    Word("champion", 3.6, 4.1),
                ],
                speaker="commentator",
            ),
        ],
        language="en",
    )
    translator = DictionaryTranslator(source="en", target="hi")
    translated = translate_transcript(source, translator, "hi")
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Commentator", [make_sine(180.0, 1.0)])
    dub_audio, fit_factors, placed = dub_transcript(translated, cloner, profile)
    report = build_report(
        translated,
        placed,
        fit_factors,
        dub_audio,
        translation_coverage=translator.last_coverage,
    )
    synced = retime_transcript(translated, placed)

    with tempfile.TemporaryDirectory(prefix="video-dubber-export-") as tmp:
        work = Path(tmp)
        wav_path = work / "dubbed.wav"
        save_wav(str(wav_path), dub_audio)
        srt_path = work / "dubbed.srt"
        srt_path.write_text(synced.to_srt(), encoding="utf-8")
        report_path = work / "report.json"
        report_path.write_text(report.to_json(), encoding="utf-8")

        manifest, sources = build_manifest(
            "mars-cricket.hi",
            dub_audio=wav_path,
            target="hi",
            subtitles_srt=srt_path,
            report=report_path,
            speakers=["commentator"],
        )
        print(manifest.to_markdown(), end="")

        bundle_dir = work / "mars-cricket.hi"
        export_bundle(manifest, sources, bundle_dir)
        problems = verify_bundle(bundle_dir)
        print(f"directory bundle: {bundle_dir} — verify: {problems or 'intact'}")
        assert problems == []

        zipped = export_zip(manifest, sources, work / "mars-cricket.hi.zip")
        print(f"zip bundle: {zipped} ({zipped.stat().st_size} bytes)")
        with zipfile.ZipFile(zipped) as zf:
            assert sorted(zf.namelist()) == [
                "README.md",
                "dubbed.srt",
                "dubbed.wav",
                "manifest.json",
                "report.json",
            ]

        # A tampered copy must fail verification — the hash is the point.
        tampered = work / "tampered"
        export_bundle(manifest, sources, tampered)
        (tampered / "dubbed.wav").write_bytes(b"not the dub")
        problems = verify_bundle(tampered)
        print(f"tampered bundle: {problems}")
        assert len(problems) == 2
        assert problems[0].startswith("changed: dubbed.wav")
        assert problems[1].startswith("size changed: dubbed.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
