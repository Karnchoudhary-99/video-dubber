"""Demo: project config file (dubber.toml).

Writes a small dubber.toml, then shows the CLI picking its defaults
up: ``translate`` without ``--target`` dubs to Hindi, and ``enhance``
normalizes to the config's target instead of the built-in -16 dBFS.
An explicit flag still beats the file.

No model, no network, no ffmpeg needed — run it with
``python examples/demo_config.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dubber.audio import load_wav, make_sine, save_wav
from video_dubber.cli import main
from video_dubber.enhance import measure_loudness
from video_dubber.transcript import Segment, Transcript, Word

CONFIG = """\
[dubber]
target_language = "hi"

[dubber.enhance]
target = -30.0
"""


def main_demo() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "dubber.toml").write_text(CONFIG, encoding="utf-8")
        transcript = root / "en.json"
        transcript.write_text(
            Transcript(
                segments=[
                    Segment(
                        words=[
                            Word(text="Hello", start=0.0, end=0.5),
                            Word(text="world", start=0.5, end=1.0),
                        ]
                    )
                ],
                language="en",
            ).to_json(),
            encoding="utf-8",
        )
        out = root / "hi.json"
        assert (
            main(
                [
                    "--config",
                    str(root / "dubber.toml"),
                    "translate",
                    str(transcript),
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        translated = Transcript.from_json(out.read_text(encoding="utf-8"))
        assert translated.language == "hi"
        print(f"translate with no --target: {translated.segments[0].text}")

        src = root / "in.wav"
        save_wav(str(src), make_sine(440.0, 1.0, amplitude=0.5))
        enhanced = root / "out.wav"
        assert (
            main(
                ["--config", str(root / "dubber.toml"), "enhance", str(src), "--out", str(enhanced)]
            )
            == 0
        )
        rms = measure_loudness(load_wav(str(enhanced))).rms_dbfs
        print(f"enhance with no --target: RMS {rms:.1f} dBFS (config says -30)")
        assert abs(rms - (-30.0)) < 1.0

        # The config subcommand shows the effective configuration.
        assert main(["--config", str(root / "dubber.toml"), "config"]) == 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main_demo())
