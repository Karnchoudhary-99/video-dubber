"""End-to-end demo of batch mode.

Builds a two-job manifest in a temp dir (two short English transcripts,
one with a ground-truth reference so its transcription WER is scored),
clones a voice from a reference tone, and runs the batch through the CLI.
Prints the per-job lines and the batch summary.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_batch.py``.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import make_sine
from video_dubber.cli import main as cli_main
from video_dubber.transcript import Segment, Transcript, Word
from video_dubber.voice import ScriptedVoiceCloner


def _transcript(words_with_timings: list[tuple[str, float, float]]) -> Transcript:
    words = [Word(text, start, end) for text, start, end in words_with_timings]
    return Transcript(segments=[Segment(words=words)], language="en")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="video-dubber-batch-"))
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Asha", [make_sine(180.0, 1.0)])
    profile_path = work / "asha.json"
    profile_path.write_text(profile.to_json(), encoding="utf-8")

    clip1 = _transcript(
        [
            ("Hello", 0.0, 0.4),
            ("and", 0.4, 0.6),
            ("welcome", 0.6, 1.1),
            ("to", 1.1, 1.3),
            ("Mars", 1.3, 1.7),
        ]
    )
    (work / "clip1.json").write_text(clip1.to_json(), encoding="utf-8")
    # Ground truth for clip1: one word differs, so its WER is 1/5 = 0.2.
    reference = _transcript(
        [
            ("Hello", 0.0, 0.4),
            ("and", 0.4, 0.6),
            ("welcome", 0.6, 1.1),
            ("to", 1.1, 1.3),
            ("Venus", 1.3, 1.7),
        ]
    )
    (work / "clip1-ref.json").write_text(reference.to_json(), encoding="utf-8")

    clip2 = _transcript(
        [
            ("Good", 0.0, 0.4),
            ("morning", 0.4, 0.9),
            ("Mars", 0.9, 1.3),
        ]
    )
    (work / "clip2.json").write_text(clip2.to_json(), encoding="utf-8")

    manifest = {
        "defaults": {"profile": "asha.json"},
        "jobs": [
            {
                "name": "clip1",
                "transcript": "clip1.json",
                "reference": "clip1-ref.json",
                "target": "hi",
            },
            {"name": "clip2", "transcript": "clip2.json", "target": "es"},
        ],
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out_dir = work / "out"
    print(f"manifest: {manifest_path}")
    code = cli_main(["batch", str(manifest_path), "--out-dir", str(out_dir)])
    summary = json.loads((out_dir / "batch-report.json").read_text(encoding="utf-8"))
    print(f"\nmean transcription WER: {summary['mean_transcription_wer']}")
    print(f"mean MOS proxy: {summary['mean_mos_proxy']}")
    print(f"reports: {out_dir / 'batch-report.json'}, {out_dir / 'batch-report.md'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
