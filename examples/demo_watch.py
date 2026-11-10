"""End-to-end demo of watch-folder mode.

Drops two short WAV files into a watch folder, clones a voice from a
reference tone, and runs ``dubber watch --once`` through the CLI: both
files are dubbed, the state file is written, and a second scan finds
nothing new. Then a third file is dropped in and a second scan dubs
only that one.

No model, no network — run it with ``python examples/demo_watch.py``.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import make_sine, save_wav
from video_dubber.cli import main as cli_main
from video_dubber.voice import ScriptedVoiceCloner

_WORDS = [
    {"text": "Hello", "start": 0.0, "end": 0.4},
    {"text": "and", "start": 0.4, "end": 0.6},
    {"text": "welcome", "start": 0.6, "end": 1.1},
    {"text": "to", "start": 1.1, "end": 1.3},
    {"text": "Mars", "start": 1.3, "end": 1.7},
]


def _drop_clip(watch: Path, name: str) -> None:
    save_wav(str(watch / name), make_sine(440.0, 2.0, amplitude=0.5))


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="video-dubber-watch-"))
    watch = work / "watch"
    watch.mkdir()
    cloner = ScriptedVoiceCloner()
    profile = cloner.clone("Demo Asha", [make_sine(180.0, 1.0)])
    profile_path = work / "asha.json"
    profile_path.write_text(profile.to_json(), encoding="utf-8")
    script_path = work / "words.json"
    script_path.write_text(json.dumps(_WORDS), encoding="utf-8")
    out_dir = work / "out"

    _drop_clip(watch, "morning clip.wav")
    _drop_clip(watch, "evening-clip.wav")
    print(f"watch folder: {watch}")

    def run_watch() -> int:
        return cli_main(
            [
                "watch",
                str(watch),
                "--out-dir",
                str(out_dir),
                "--target",
                "hi",
                "--profile",
                str(profile_path),
                "--backend",
                "scripted",
                "--script",
                str(script_path),
                "--once",
                "--no-mux",
            ]
        )

    code = run_watch()
    print(f"\nstate file: {out_dir / '.dubber-watch.json'}")
    print("second scan (nothing new):")
    code = run_watch() or code
    print("\na third file lands in the folder:")
    _drop_clip(watch, "late arrival.wav")
    code = run_watch() or code
    dubbed = sorted(p.parent.name for p in out_dir.glob("*/dubbed.wav"))
    print(f"\ndubbed jobs: {dubbed}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
