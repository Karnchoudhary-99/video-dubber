"""End-to-end demo of multi-target dubbing.

Takes one short English transcript and dubs it into Hindi and Spanish
in a single run: per-language bundles (dubbed WAV, quality report,
dub-synced subtitles) plus a summary comparing the two targets.

No model, no network — run it with ``python examples/demo_multilang.py``.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import make_sine
from video_dubber.cli import main as cli_main
from video_dubber.voice import ScriptedVoiceCloner

_WORDS = [
    {"text": "Hello", "start": 0.0, "end": 0.4},
    {"text": "and", "start": 0.4, "end": 0.6},
    {"text": "welcome", "start": 0.6, "end": 1.1},
    {"text": "to", "start": 1.1, "end": 1.3},
    {"text": "Mars", "start": 1.3, "end": 1.7},
]


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="video-dubber-multilang-"))
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Asha", [make_sine(180.0, 1.0)])
    profile_path = work / "asha.json"
    profile_path.write_text(profile.to_json(), encoding="utf-8")
    transcript = {
        "language": "en",
        "segments": [
            {"words": _WORDS, "speaker": None},
        ],
    }
    transcript_path = work / "source.json"
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    out_dir = work / "bundles"
    print(f"work dir: {work}")

    code = cli_main(
        [
            "dub-multi",
            str(transcript_path),
            "--targets",
            "hi,es",
            "--profile",
            str(profile_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0, f"dub-multi exited {code}"
    summary = json.loads((out_dir / "dub.multilang.json").read_text())
    assert summary["succeeded"] == 2, summary
    assert summary["targets"] == ["hi", "es"]
    for target in ("hi", "es"):
        assert (out_dir / f"dub.{target}.wav").exists()
        assert (out_dir / f"dub.{target}.report.json").exists()
        assert (out_dir / f"dub.{target}.srt").exists()
    print((out_dir / "dub.multilang.md").read_text())
    print("multi-target demo OK: hi + es dubbed from one English source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
